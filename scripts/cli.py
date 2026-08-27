#!/usr/bin/env python
"""FaithGuard CLI: ask questions through the guarded pipeline from the terminal.

Usage:
  python scripts/cli.py ask "Who wrote Hamlet?"
  python scripts/cli.py detect --answer "..." --passages "..." "..."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithguard.config import DATA_DIR, MODELS_DIR, get_settings
from faithguard.detection.classifier import HallucinationClassifier
from faithguard.detection.nli import NLIScorer
from faithguard.generation.llm import LLMClient
from faithguard.pipeline import GuardedRAGPipeline
from faithguard.retrieval.chunking import Chunker
from faithguard.retrieval.hybrid import HybridRetriever


def build_pipeline(settings):
    docs = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    chunker = Chunker(settings.retrieval.chunk_size, settings.retrieval.chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    print(f"[cli] indexing {len(chunks)} chunks ...", file=sys.stderr)
    retriever = HybridRetriever(
        chunks,
        embedding_model=settings.retrieval.embedding_model,
        device=settings.device,
    )
    nli = NLIScorer(settings.detection.nli_model, device=settings.device)
    clf = HallucinationClassifier(MODELS_DIR / "hallucination_classifier.pkl")
    llm = LLMClient.from_settings(settings)
    return GuardedRAGPipeline(retriever, llm, nli, clf, settings)


def main():
    ap = argparse.ArgumentParser(prog="faithguard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="ask through the guarded pipeline")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-mitigate", action="store_true")
    p_ask.add_argument("--json", action="store_true")

    p_det = sub.add_parser("detect", help="score an answer against passages")
    p_det.add_argument("--answer", required=True)
    p_det.add_argument("--passages", nargs="+", required=True)
    p_det.add_argument("--json", action="store_true")

    args = ap.parse_args()
    settings = get_settings()

    if args.cmd == "ask":
        pipe = build_pipeline(settings)
        result = pipe.ask(args.question, mitigate=not args.no_mitigate)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"\nQ: {result.question}")
            print(f"A: {result.answer}\n")
            print(f"p(hallucinated)={result.hallucination_probability:.3f}  "
                  f"initial_flag={result.hallucinated_initial}  "
                  f"mitigated={result.mitigated}  abstained={result.abstained}  "
                  f"rounds={result.mitigation_rounds}  "
                  f"claims={result.n_claims} unsupported={result.n_unsupported} "
                  f"contradicted={result.n_contradicted}  "
                  f"latency={result.latency_ms:.0f}ms")
    elif args.cmd == "detect":
        nli = NLIScorer(settings.detection.nli_model, device=settings.device)
        clf = HallucinationClassifier(MODELS_DIR / "hallucination_classifier.pkl")
        pipe = GuardedRAGPipeline(None, None, nli, clf, settings)
        result = pipe.detect_answer(args.answer, args.passages)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            v = result.verdict
            print(f"hallucinated={v['hallucinated']}  p={v['probability']:.3f}  "
                  f"method={v['method']}  claims={v['n_claims']} "
                  f"unsupported={v['n_unsupported']} contradicted={v['n_contradicted']}")
            for c in v["claim_details"]:
                mark = "OK " if c["supported"] else ("XX " if c["contradicted"] else "?? ")
                print(f"  {mark} ent={c['best_entailment']:.2f}  {c['claim'][:100]}")


if __name__ == "__main__":
    main()
