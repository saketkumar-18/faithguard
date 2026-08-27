"""API tests using FastAPI TestClient with stubbed models (no downloads)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """App with no default corpus and stubbed model loading."""
    import faithguard.api.app as appmod

    # Stub the heavy model classes so lifespan doesn't download anything
    class FakeNLI:
        def score_claims(self, claims, passages, batch_size=32):
            from faithguard.detection.nli import ClaimEvidenceScore
            return [
                ClaimEvidenceScore(c["text"], c.get("hedged", False), 0.9, 0.05, 0.7, 0)
                for c in claims
            ]

    class FakeClassifier:
        model = None
        def verdict(self, answer, claim_scores, n_passages, passages=None, question=None, decision_threshold=None):
            from faithguard.detection.classifier import AnswerVerdict
            return AnswerVerdict(
                hallucinated=False, probability=0.1, method="rules",
                n_claims=len(claim_scores), n_unsupported=0, n_contradicted=0,
                unsupported_claims=[], contradicted_claims=[],
                claim_details=[],
            )

    orig_nli, orig_clf = appmod.NLIScorer, appmod.HallucinationClassifier
    appmod.NLIScorer = lambda *a, **k: FakeNLI()
    appmod.HallucinationClassifier = lambda *a, **k: FakeClassifier()
    try:
        app = appmod.create_app(load_default_corpus=False)
        with TestClient(app) as c:
            yield c
    finally:
        appmod.NLIScorer, appmod.HallucinationClassifier = orig_nli, orig_clf


class TestAPI:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["nli_loaded"] is True

    def test_ask_without_corpus_409(self, client):
        r = client.post("/ask", json={"question": "anything here?"})
        assert r.status_code == 409

    def test_load_corpus_then_detect(self, client):
        docs = [
            {"id": "d1", "text": "The sky is blue because of Rayleigh scattering.", "title": "Sky"},
            {"id": "d2", "text": "Water boils at 100 degrees Celsius at sea level.", "title": "Water"},
        ]
        r = client.post("/corpus/load", json={"documents": docs})
        assert r.status_code == 200
        assert r.json()["docs"] == 2

        r = client.post("/detect", json={
            "answer": "Water boils at 100 degrees Celsius at sea level.",
            "passages": ["Water boils at 100 degrees Celsius at sea level."],
        })
        assert r.status_code == 200
        assert r.json()["hallucinated"] is False

    def test_detect_validation(self, client):
        r = client.post("/detect", json={"answer": "", "passages": []})
        assert r.status_code == 422

    def test_corpus_info(self, client):
        r = client.get("/corpus")
        assert r.status_code == 200
        assert "chunks" in r.json()
