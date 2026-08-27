#!/usr/bin/env python
"""End-to-end faithfulness evaluation: baseline RAG vs FaithGuard-guarded RAG.

Runs the QA gold set through two pipelines:
  1. baseline:  retrieve -> generate  (no detection, no mitigation)
  2. guarded:   retrieve -> generate -> detect -> (re-retrieve + regenerate)

Measures on identical questions:
  - answer_correctness (token-F1 vs gold answers)
  - faithfulness (mean claim entailment, via NLI on the final answer)
  - hallucination rate (fraction of answers flagged hallucinated)
  - abstention rate, mitigation success rate, latency

Writes reports/faithfulness_eval.json + a markdown summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithguard.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, get_settings
from faithguard.detection.classifier import HallucinationClassifier
from faithguard.detection.nli import NLIScorer
from faithguard.eval.metrics import answer_correctness, answer_faithfulness, claim_precision
from faithguard.generation.llm import LLMClient
from faithguard.pipeline import GuardedRAGPipeline
from faithguard.retrieval.chunking import Chunker
from faithguard.retrieval.hybrid import HybridRetriever


def load_corpus():
    docs = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    return docs


def build_retriever(docs, settings):
    chunker = Chunker(settings.retrieval.chunk_size, settings.retrieval.chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    print(f"[eval] {len(chunks)} chunks from {len(docs)} documents")
    retriever = HybridRetriever(
        chunks,
        embedding_model=settings.retrieval.embedding_model,
        device=settings.device,
        rrf_k=settings.retrieval.rrf_k,
        bm25_weight=settings.retrieval.bm25_weight,
        dense_weight=settings.retrieval.dense_weight,
        use_dense=False,  # attach dense index below with caching
    )
    from faithguard.retrieval.dense import DenseIndex

    cache = DATA_DIR / "embed_cache.npz"
    dense = DenseIndex(settings.retrieval.embedding_model, device=settings.device)
    if dense.load_cache(cache):
        dense.chunks = chunks
        print(f"[eval] loaded embedding cache from {cache.name}")
    else:
        print("[eval] embedding chunks (first run, cached afterwards) ...")
        dense.index(chunks)
        dense.save_cache(cache)
    retriever.dense = dense
    return retriever


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "mean_correctness": float(np.mean([r["correctness"] for r in rows])),
        "mean_faithfulness": float(np.mean([r["faithfulness"] for r in rows])),
        "mean_claim_precision": float(np.mean([r["claim_precision"] for r in rows])),
        "hallucination_rate": float(np.mean([r["flagged"] for r in rows])),
        "abstention_rate": float(np.mean([r["abstained"] for r in rows])),
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit number of questions (0=all)")
    ap.add_argument("--out", default=str(REPORTS_DIR / "faithfulness_eval.json"))
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds to sleep between questions (throttle free-tier LLM)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="resume from checkpoint if present")
    args = ap.parse_args()

    settings = get_settings()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    qa_gold = json.loads((DATA_DIR / "qa_gold.json").read_text(encoding="utf-8"))
    if args.n:
        qa_gold = qa_gold[: args.n]
    print(f"[eval] {len(qa_gold)} gold questions")

    docs = load_corpus()
    retriever = build_retriever(docs, settings)

    print(f"[eval] loading NLI {settings.detection.nli_model} ...")
    nli = NLIScorer(settings.detection.nli_model, device=settings.device)
    clf = HallucinationClassifier(
        MODELS_DIR / "hallucination_classifier.pkl",
        unsupported_threshold=settings.detection.unsupported_claim_threshold,
        hallucinated_fraction=settings.detection.hallucinated_answer_fraction,
    )
    llm = LLMClient.from_settings(settings)
    if not llm.api_key:
        print("[eval] WARNING: no LLM API key found; generation will fail")

    pipeline = GuardedRAGPipeline(retriever, llm, nli, clf, settings)

    # ---- checkpoint: resume across crashes / endpoint outages
    ckpt_path = REPORTS_DIR / "faithfulness_eval.ckpt.json"
    baseline_rows, guarded_rows, done_ids = [], [], set()
    if args.resume and ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        baseline_rows = ck.get("baseline_rows", [])
        guarded_rows = ck.get("guarded_rows", [])
        done_ids = {r["id"] for r in guarded_rows}
        print(f"[eval] resumed checkpoint: {len(done_ids)} questions already done")

    def save_ckpt():
        ckpt_path.write_text(json.dumps(
            {"baseline_rows": baseline_rows, "guarded_rows": guarded_rows},
            ensure_ascii=False), encoding="utf-8")

    t_start = time.time()
    n_done_this_run = 0
    for i, item in enumerate(qa_gold, 1):
        if item["id"] in done_ids:
            continue
        q = item["question"]
        refs = item["answers"]
        try:
            # ---- baseline: single-shot RAG, no guard
            t_b = time.perf_counter()
            passages = retriever.search(q, top_k=settings.retrieval.top_k)
            ptexts = [p.text for p in passages]
            from faithguard.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
            resp = llm.chat(SYSTEM_PROMPT, build_rag_prompt(q, ptexts, [p.title for p in passages]))
            b_answer = resp.text if resp.ok else ""
            b_verdict = pipeline.detect(b_answer, ptexts, q)
            b_latency = (time.perf_counter() - t_b) * 1000.0
            baseline_rows.append({
                "id": item["id"], "question": q, "answer": b_answer,
                "correctness": answer_correctness(b_answer, refs),
                "faithfulness": answer_faithfulness(b_verdict_claim_scores(b_verdict)),
                "claim_precision": claim_precision(b_verdict_claim_scores(b_verdict)),
                "flagged": int(b_verdict.hallucinated),
                "abstained": 0,
                "latency_ms": b_latency,
            })

            # ---- guarded pipeline
            g = pipeline.ask(q, mitigate=True)
            g_verdict_scores = [
                _ScoreProxy(c) for c in g.verdict["claim_details"]
            ]
            guarded_rows.append({
                "id": item["id"], "question": q, "answer": g.answer,
                "correctness": answer_correctness(g.answer, refs),
                "faithfulness": answer_faithfulness(g_verdict_scores),
                "claim_precision": claim_precision(g_verdict_scores),
                "flagged": int(g.hallucinated_initial),
                "abstained": int(g.abstained),
                "mitigated": int(g.mitigated),
                "mitigation_rounds": g.mitigation_rounds,
                "p_hallucination": g.hallucination_probability,
                "latency_ms": g.latency_ms,
            })
            done_ids.add(item["id"])
            n_done_this_run += 1
        except Exception as e:  # keep the eval running on individual failures
            print(f"[eval] Q{i} failed: {e}")
            continue
        save_ckpt()
        if n_done_this_run % 5 == 0:
            elapsed = time.time() - t_start
            print(f"[eval] {len(done_ids)}/{len(qa_gold)} done "
                  f"({elapsed/max(n_done_this_run,1):.1f}s/q this run)")
        time.sleep(args.sleep)  # throttle: free-tier endpoint rate limits

    base = summarize(baseline_rows)
    guard = summarize(guarded_rows)
    n_mitigated = sum(r.get("mitigated", 0) for r in guarded_rows)
    gains = {
        "faithfulness_delta": round(guard.get("mean_faithfulness", 0) - base.get("mean_faithfulness", 0), 4),
        "claim_precision_delta": round(guard.get("mean_claim_precision", 0) - base.get("mean_claim_precision", 0), 4),
        "correctness_delta": round(guard.get("mean_correctness", 0) - base.get("mean_correctness", 0), 4),
        "hallucination_rate_delta": round(guard.get("hallucination_rate", 0) - base.get("hallucination_rate", 0), 4),
    }
    out = {
        "n_questions": len(qa_gold),
        "n_evaluated": len(guarded_rows),
        "baseline": base,
        "guarded": guard,
        "gains": gains,
        "n_mitigated": n_mitigated,
        "baseline_rows": baseline_rows,
        "guarded_rows": guarded_rows,
        "config": {
            "llm_model": settings.generation.model,
            "nli_model": settings.detection.nli_model,
            "embedding_model": settings.retrieval.embedding_model,
            "top_k": settings.retrieval.top_k,
            "re_top_k": settings.retrieval.re_top_k,
            "max_mitigation_rounds": settings.mitigation.max_rounds,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(json.dumps({"baseline": base, "guarded": guard, "gains": gains}, indent=2))
    print(f"[eval] saved {args.out}")


class _ScoreProxy:
    """Adapt stored claim_details dicts back to the metric interface."""
    def __init__(self, d: dict):
        self.best_entailment = d["best_entailment"]
        self.supported = d["supported"]
        self.contradicted = d["contradicted"]


def b_verdict_claim_scores(verdict):
    return [_ScoreProxy(c) for c in verdict.claim_details]


if __name__ == "__main__":
    main()
