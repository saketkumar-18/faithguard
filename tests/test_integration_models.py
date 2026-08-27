"""Integration tests with real (small) models — marked 'models'.

Run:  pytest -m models
These download ~300MB of models on first run, then cache in ~/.cache.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def nli():
    from faithguard.detection.nli import NLIScorer

    return NLIScorer("cross-encoder/nli-deberta-v3-small", device="cpu")


@pytest.fixture(scope="module")
def retriever(tiny_chunks):
    from faithguard.retrieval.hybrid import HybridRetriever

    return HybridRetriever(tiny_chunks, embedding_model="BAAI/bge-small-en-v1.5", device="cpu")


class TestNLIScorer:
    def test_entailed_claim(self, nli):
        scores = nli.score_claims(
            [{"text": "The Eiffel Tower is in Paris, France.", "hedged": False}],
            ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."],
        )
        assert scores[0].best_entailment > 0.7
        assert scores[0].supported

    def test_unsupported_claim(self, nli):
        scores = nli.score_claims(
            [{"text": "The Eiffel Tower was completed in 1955.", "hedged": False}],
            ["The Eiffel Tower was built from 1887 to 1889 by Gustave Eiffel's company."],
        )
        assert scores[0].best_entailment < 0.5

    def test_contradicted_claim(self, nli):
        scores = nli.score_claims(
            [{"text": "The tower is located in London.", "hedged": False}],
            ["The Eiffel Tower stands on the Champ de Mars in Paris, France."],
        )
        assert scores[0].best_contradiction > scores[0].best_entailment


class TestHybridRetrieval:
    def test_dense_retrieval_finds_right_doc(self, retriever):
        hits = retriever.search("Which emperor commissioned the Taj Mahal?", top_k=2)
        assert hits[0].doc_id == "doc-taj"

    def test_rrf_fusion_scores(self, retriever):
        hits = retriever.search("visitors per year", top_k=2)
        assert len(hits) == 2
        assert hits[0].score >= hits[1].score > 0


class TestDetectionEndToEnd:
    def test_faithful_answer_passes(self, nli, retriever):
        from faithguard.detection.classifier import HallucinationClassifier
        from faithguard.detection.claims import extract_claims

        passages = retriever.search("When was construction of the Taj Mahal completed?", top_k=3)
        answer = "Construction of the Taj Mahal was essentially completed in 1648."
        claims = extract_claims(answer)
        scores = nli.score_claims(claims, [p.text for p in passages])
        verdict = HallucinationClassifier().verdict(answer, scores, len(passages))
        assert verdict.hallucinated is False

    def test_hallucinated_answer_flagged(self, nli, retriever):
        from faithguard.detection.classifier import HallucinationClassifier
        from faithguard.detection.claims import extract_claims

        passages = retriever.search("When was construction of the Taj Mahal completed?", top_k=3)
        answer = (
            "The Taj Mahal was completed in 1899 under British supervision. "
            "It was designed by the architect Edwin Lutyens."
        )
        claims = extract_claims(answer)
        scores = nli.score_claims(claims, [p.text for p in passages])
        verdict = HallucinationClassifier().verdict(answer, scores, len(passages))
        assert verdict.hallucinated is True
        assert verdict.n_unsupported >= 1
