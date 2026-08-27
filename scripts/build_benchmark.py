#!/usr/bin/env python
"""Build the FaithGuard benchmark from SQuAD v1.1.

Outputs (under data/):
- squad_dev.json            raw SQuAD dev set (downloaded once)
- corpus.json               article corpus (chunked documents) for retrieval
- detection_dataset.jsonl   labeled hallucination detection examples
                            {id, article_id, split, question, answer, passages,
                             label, corruption}
- qa_gold.json              end-to-end QA eval set {question, answers, article_id}

Hallucination labels are created BY CONSTRUCTION:
  label=0 (faithful):  the gold answer, supported by its own article context.
  label=1 (hallucinated): corrupted variants that look plausible but are NOT
    supported by the retrieved context:
      - entity_swap:     named entities/numbers replaced with entities drawn
                         from a DIFFERENT article.
      - cross_answer:    the gold answer of a different question/article.
      - number_perturb:  numbers/dates shifted by a small delta.
      - fabricated:      a plausible-looking fabricated sentence appended.

Article-level train/test split prevents leakage between splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithguard.config import DATA_DIR, get_settings
from faithguard.retrieval.chunking import Chunker

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){0,3})\b")
NUMBER_RE = re.compile(r"\b\d{1,4}(?:\.\d+)?\b")
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20[0-3]\d)\b")


def download_squad(path: Path) -> dict:
    if path.exists():
        print(f"[build] using cached {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"[build] downloading SQuAD dev set from {SQUAD_URL}")
    req = urllib.request.Request(SQUAD_URL, headers={"User-Agent": "faithguard-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    path.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def collect_articles(squad: dict) -> list[dict]:
    """Flatten SQuAD into articles: {title, text, paragraphs, qas}."""
    articles = []
    for art in squad["data"]:
        title = art["title"]
        paras = []
        qas = []
        for p in art["paragraphs"]:
            paras.append(p["context"])
            for qa in p.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                if answers:
                    qas.append({
                        "id": qa["id"],
                        "question": qa["question"],
                        "answers": answers,
                        "context": p["context"],
                    })
        if qas:
            articles.append({
                "title": title,
                "text": " ".join(paras),
                "paragraphs": paras,
                "qas": qas,
            })
    return articles


def assign_split(title: str, test_frac: float = 0.3) -> str:
    h = int(hashlib.md5(title.encode()).hexdigest(), 16) % 1000
    return "test" if h < test_frac * 1000 else "train"


def build_corpus(articles: list[dict], settings) -> tuple[list[dict], dict]:
    """Chunk articles into the retrieval corpus. Returns (documents, title->doc_id)."""
    chunker = Chunker(settings.retrieval.chunk_size, settings.retrieval.chunk_overlap)
    documents = []
    for i, art in enumerate(articles):
        doc_id = f"squad-{i:04d}"
        documents.append({
            "id": doc_id,
            "title": art["title"],
            "text": art["text"],
            "meta": {"split": art["split"]},
        })
    return documents


def retrieve_passages_for(qa: dict, doc_text: str, chunker: Chunker, doc_id: str, title: str, k: int = 5) -> list[str]:
    """Simple deterministic passage selection: the gold paragraph first, then
    nearest-neighbor paragraphs by token overlap with the question."""
    paras = [p for p in doc_text.split(" ") if False]  # noqa - unused
    return []


def entity_pool(articles: list[dict], exclude_title: str, max_entities: int = 400) -> list[str]:
    ents: set[str] = set()
    for art in articles:
        if art["title"] == exclude_title:
            continue
        for m in ENTITY_RE.finditer(art["text"][:20000]):
            e = m.group(0).strip()
            if 3 <= len(e) <= 40:
                ents.add(e)
        if len(ents) > max_entities * 3:
            break
    return sorted(ents)[:max_entities]


def corrupt_entity_swap(answer: str, pool: list[str], rng: random.Random) -> str | None:
    """Replace entities/numbers in the answer with foreign entities."""
    entities = [m.group(0) for m in ENTITY_RE.finditer(answer)]
    numbers = [m.group(0) for m in NUMBER_RE.finditer(answer)]
    targets = entities + numbers
    if not targets or not pool:
        return None
    target = rng.choice(targets)
    replacement = rng.choice(pool)
    if replacement.lower() == target.lower():
        return None
    out = answer.replace(target, replacement, 1)
    return out if out != answer else None


def corrupt_number_perturb(answer: str, rng: random.Random) -> str | None:
    years = YEAR_RE.findall(answer)
    nums = NUMBER_RE.findall(answer)
    if years:
        y = rng.choice(years)
        delta = rng.choice([-3, -2, -1, 1, 2, 3, 10, 50])
        new = str(int(y) + delta)
        out = answer.replace(y, new, 1)
        return out if out != answer else None
    if nums:
        n = rng.choice(nums)
        try:
            val = float(n)
        except ValueError:
            return None
        delta = rng.choice([1, 2, 3, 5, 10]) * (1 if rng.random() > 0.5 else -1)
        new_val = val + delta
        new = str(int(new_val)) if new_val == int(new_val) else f"{new_val:.1f}"
        if new == n:
            return None
        out = answer.replace(n, new, 1)
        return out if out != answer else None
    return None


FABRICATED_TEMPLATES = [
    "This fact was confirmed by a 2019 study at the University of Cambridge.",
    "The event was later featured in a documentary narrated by Morgan Freeman.",
    "Experts estimate the economic impact exceeded two billion dollars.",
    "A related treaty was signed in Geneva the following year.",
    "The discovery was initially rejected by three peer-reviewed journals.",
    "Over 40,000 visitors attended the ceremony each year.",
]


def corrupt_fabricated(answer: str, rng: random.Random) -> str:
    return answer.rstrip(". ") + ". " + rng.choice(FABRICATED_TEMPLATES)


def build_detection_dataset(
    articles: list[dict],
    retriever_search,
    rng: random.Random,
    max_examples_per_split: int,
) -> list[dict]:
    """Build labeled examples. retriever_search(question, article, gold_context) -> passages."""
    examples: list[dict] = []
    by_split = {"train": [], "test": []}
    for art in articles:
        by_split[art["split"]].append(art)

    # global entity pools per split (foreign entities from other articles)
    pools = {
        "train": entity_pool(by_split["train"], exclude_title="", max_entities=400),
        "test": entity_pool(by_split["test"], exclude_title="", max_entities=400),
    }

    cross_answers: dict[str, list[tuple[str, str]]] = {"train": [], "test": []}
    for split, arts in by_split.items():
        for art in arts:
            for qa in art["qas"]:
                cross_answers[split].append((art["title"], qa["answers"][0]))

    eid = 0
    for split, arts in by_split.items():
        pool = [e for e in pools[split] if e]
        count = 0
        for art in arts:
            if count >= max_examples_per_split:
                break
            for qa in art["qas"]:
                if count >= max_examples_per_split:
                    break
                gold = qa["answers"][0]
                if len(gold) < 15:
                    continue
                passages = retriever_search(qa["question"], art, qa.get("context"))
                if not passages:
                    continue
                base = {
                    "article_id": art["title"],
                    "split": split,
                    "question": qa["question"],
                    "passages": passages,
                }
                # faithful example
                examples.append({**base, "id": f"ex-{eid:06d}", "answer": gold,
                                 "label": 0, "corruption": "none"})
                eid += 1
                count += 1

                # corrupted examples (try several strategies, keep 2 per question)
                made = 0
                strategies = ["entity_swap", "number_perturb", "fabricated", "cross_answer"]
                rng.shuffle(strategies)
                for strat in strategies:
                    if made >= 2:
                        break
                    bad = None
                    if strat == "entity_swap":
                        bad = corrupt_entity_swap(gold, pool, rng)
                    elif strat == "number_perturb":
                        bad = corrupt_number_perturb(gold, rng)
                    elif strat == "fabricated":
                        bad = corrupt_fabricated(gold, rng)
                    elif strat == "cross_answer":
                        cands = [(t, a) for (t, a) in cross_answers[split]
                                 if t != art["title"] and len(a) >= 15]
                        if cands:
                            bad = rng.choice(cands)[1]
                    if bad and bad != gold:
                        examples.append({**base, "id": f"ex-{eid:06d}", "answer": bad,
                                         "label": 1, "corruption": strat})
                        eid += 1
                        made += 1
                        count += 1
    return examples


def build_qa_gold(articles: list[dict], rng: random.Random, n: int) -> list[dict]:
    """End-to-end QA evaluation set from the TEST split (unseen articles)."""
    test_arts = [a for a in articles if a["split"] == "test"]
    qas = []
    for art in test_arts:
        for qa in art["qas"]:
            if len(qa["answers"][0]) >= 3:
                qas.append({
                    "id": qa["id"],
                    "article_id": art["title"],
                    "question": qa["question"],
                    "answers": qa["answers"][:5],
                })
    rng.shuffle(qas)
    return qas[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-articles", type=int, default=90)
    ap.add_argument("--max-detection-per-split", type=int, default=450)
    ap.add_argument("--qa-gold-size", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    squad = download_squad(DATA_DIR / "squad_dev.json")
    articles = collect_articles(squad)
    rng.shuffle(articles)
    articles = articles[: args.max_articles]
    for art in articles:
        art["split"] = assign_split(art["title"])
    print(f"[build] {len(articles)} articles "
          f"(train={sum(1 for a in articles if a['split']=='train')}, "
          f"test={sum(1 for a in articles if a['split']=='test')})")

    # corpus
    documents = build_corpus(articles, settings)
    (DATA_DIR / "corpus.json").write_text(
        json.dumps(documents, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[build] corpus: {len(documents)} documents -> data/corpus.json")

    # lightweight passage selection for dataset construction:
    # the paragraph containing the gold answer FIRST (this is what a good
    # retriever must surface), then top token-overlap paragraphs of the article.
    chunker = Chunker(settings.retrieval.chunk_size, settings.retrieval.chunk_overlap)

    def passage_selector(question: str, art: dict, gold_context: str | None = None) -> list[str]:
        chunks = chunker.chunk_text(art["text"], doc_id=art["title"])
        if not chunks:
            return []
        qtok = set(re.findall(r"[a-z0-9]+", question.lower()))
        scored = []
        for c in chunks:
            ctok = set(re.findall(r"[a-z0-9]+", c.text.lower()))
            overlap = len(qtok & ctok)
            scored.append((overlap, c.text))
        scored.sort(key=lambda x: x[0], reverse=True)
        passages: list[str] = []
        if gold_context:
            passages.append(gold_context)
        for _, t in scored:
            if len(passages) >= 5:
                break
            if t not in passages:
                passages.append(t)
        return passages

    examples = build_detection_dataset(
        articles, passage_selector, rng, args.max_detection_per_split
    )
    rng.shuffle(examples)
    out = DATA_DIR / "detection_dataset.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    n_pos = sum(1 for e in examples if e["label"] == 1)
    print(f"[build] detection dataset: {len(examples)} examples "
          f"({n_pos} hallucinated / {len(examples) - n_pos} faithful) -> {out.name}")
    from collections import Counter
    print("        corruption mix:", dict(Counter(e['corruption'] for e in examples)))

    qa_gold = build_qa_gold(articles, rng, args.qa_gold_size)
    (DATA_DIR / "qa_gold.json").write_text(json.dumps(qa_gold, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"[build] QA gold set: {len(qa_gold)} questions -> data/qa_gold.json")
    print("[build] done.")


if __name__ == "__main__":
    main()
