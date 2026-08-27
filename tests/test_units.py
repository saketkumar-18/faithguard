"""Unit tests: chunking, claims, features, metrics, prompts, RRF logic."""
from __future__ import annotations

import numpy as np
import pytest

from faithguard.retrieval.chunking import Chunker
from faithguard.retrieval.bm25 import BM25Index, tokenize
from faithguard.detection.claims import extract_claims
from faithguard.detection.features import build_features, FEATURE_NAMES
from faithguard.eval.metrics import token_f1, answer_correctness, detection_report
from faithguard.generation.prompts import build_rag_prompt, build_corrective_prompt
from faithguard.mitigation.engine import MitigationEngine


# ------------------------------------------------------------------ chunking
class TestChunker:
    def test_basic_split_and_overlap(self):
        text = "Sentence one is here. " * 60
        c = Chunker(chunk_size=200, chunk_overlap=50)
        chunks = c.chunk_text(text.strip(), doc_id="d")
        assert len(chunks) > 1
        # overlap: end of chunk i appears in chunk i+1
        assert chunks[0].text[-40:] in chunks[1].text or chunks[1].text[:60] in chunks[0].text

    def test_empty_text(self):
        assert Chunker().chunk_text("   ", doc_id="d") == []

    def test_uid(self):
        chunks = Chunker(chunk_size=50, chunk_overlap=10).chunk_text(
            "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.", doc_id="x"
        )
        assert chunks[0].uid == "x#0"

    def test_overlap_must_be_smaller(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, chunk_overlap=100)


# ---------------------------------------------------------------------- bm25
class TestBM25:
    def test_search_ranks_relevant(self, tiny_chunks):
        idx = BM25Index(tiny_chunks)
        hits = idx.search("Who built the Eiffel Tower?", top_k=3)
        assert hits
        top = tiny_chunks[hits[0][0]]
        assert top.doc_id == "doc-eiffel"

    def test_tokenize(self):
        assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


# -------------------------------------------------------------------- claims
class TestClaims:
    def test_splits_sentences(self):
        ans = "The Taj Mahal is in Agra. It was built by Shah Jahan in 1631."
        claims = extract_claims(ans)
        assert len(claims) == 2

    def test_strips_filler_prefix(self):
        ans = "Based on the provided context, the tower is 330 metres tall."
        claims = extract_claims(ans)
        assert claims and not claims[0]["text"].lower().startswith("based on")

    def test_hedge_detection(self):
        claims = extract_claims("The population may have been approximately 5 million people.")
        assert claims and claims[0]["hedged"] is True

    def test_empty(self):
        assert extract_claims("") == []
        assert extract_claims("   ") == []

    def test_short_fragments_merge(self):
        claims = extract_claims("The answer is 42, which is a famous number in mathematics.")
        assert all(len(c["text"]) >= 15 for c in claims)


# ------------------------------------------------------------------ features
class TestFeatures:
    def _scores(self):
        from faithguard.detection.nli import ClaimEvidenceScore

        return [
            ClaimEvidenceScore("c1", False, 0.9, 0.05, 0.6, 0),
            ClaimEvidenceScore("c2", True, 0.3, 0.4, 0.2, 1),
        ]

    def test_shape_and_names(self):
        f = build_features(self._scores(), "some answer text", 5)
        assert f.shape == (len(FEATURE_NAMES),)
        assert f[0] == 2  # n_claims

    def test_empty_claims(self):
        f = build_features([], "x", 3)
        assert f.shape == (len(FEATURE_NAMES),)
        assert f[4] == 1.0  # frac_supported vacuously 1


# ------------------------------------------------------------------- metrics
class TestMetrics:
    def test_token_f1_exact(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_token_f1_partial(self):
        f = token_f1("the cat", "the cat sat on mat")
        assert 0 < f < 1

    def test_token_f1_disjoint(self):
        assert token_f1("apple", "banana") == 0.0

    def test_correctness_multi_ref(self):
        assert answer_correctness("Shah Jahan", ["Shah Jahan", "the emperor"]) == 1.0

    def test_detection_report(self):
        rep = detection_report([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
        assert rep["f1"] == 1.0 and rep["auc"] == 1.0
        assert rep["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


# ------------------------------------------------------------------- prompts
class TestPrompts:
    def test_rag_prompt_contains_passages_and_question(self):
        p = build_rag_prompt("Who built it?", ["passage A", "passage B"], ["T1", "T2"])
        assert "[1] (T1) passage A" in p and "Who built it?" in p

    def test_corrective_prompt_lists_bad_claims(self):
        p = build_corrective_prompt(
            "Q?", ["ctx"], "bad answer", ["unsup claim"], ["contra claim"]
        )
        assert "unsup claim" in p and "contra claim" in p and "INSUFFICIENT_CONTEXT" in p


# ------------------------------------------------------------- query expansion
class TestQueryExpansion:
    def test_expand_includes_failed_claims(self):
        from faithguard.detection.classifier import AnswerVerdict

        v = AnswerVerdict(
            hallucinated=True, probability=0.9, method="rules", n_claims=2,
            n_unsupported=1, n_contradicted=1,
            unsupported_claims=["it was built in 1999"],
            contradicted_claims=["the height is 50m"],
            claim_details=[],
        )
        q = MitigationEngine.expand_query("When was it built?", v)
        assert "1999" in q and "50m" in q
