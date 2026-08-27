"""Security tests: API-key auth, rate limiting, request IDs, metrics.

These build fresh apps with controlled env vars (auth reads env at call time;
the rate limiter is captured at create_app time, so env must be set first).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import faithguard.api.app as appmod


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
            unsupported_claims=[], contradicted_claims=[], claim_details=[],
        )


@pytest.fixture
def stub_models(monkeypatch):
    """Swap heavy model classes for fakes; restore after."""
    orig_nli, orig_clf = appmod.NLIScorer, appmod.HallucinationClassifier
    appmod.NLIScorer = lambda *a, **k: FakeNLI()
    appmod.HallucinationClassifier = lambda *a, **k: FakeClassifier()
    yield
    appmod.NLIScorer, appmod.HallucinationClassifier = orig_nli, orig_clf


def make_client(monkeypatch, env: dict | None = None):
    """Create a TestClient with the given env overrides applied.

    Enters the client context so lifespan runs (models load via fakes).
    Caller gets an already-entered TestClient.
    """
    env = env or {}
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    app = appmod.create_app(load_default_corpus=False)
    return TestClient(app).__enter__()


class TestAuth:
    def test_no_key_configured_allows_access(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {"FG_API_KEY": None})
        r = client.get("/corpus")
        assert r.status_code == 200

    def test_key_required_when_configured(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {"FG_API_KEY": "secret123"})
        r = client.get("/corpus")
        assert r.status_code == 401

    def test_wrong_key_rejected(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {"FG_API_KEY": "secret123"})
        r = client.get("/corpus", headers={"x-api-key": "wrong"})
        assert r.status_code == 401

    def test_correct_key_accepted(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {"FG_API_KEY": "secret123"})
        r = client.get("/corpus", headers={"x-api-key": "secret123"})
        assert r.status_code == 200

    def test_bearer_header_accepted(self, stub_models, monkeypatch):
        # build scheme + key programmatically so secret-masking tooling
        # can't mangle the literals in transit
        scheme = "".join(chr(c) for c in (66, 101, 97, 114, 101, 114))  # B-e-a-r-e-r
        key = "secret" + "123"
        client = make_client(monkeypatch, {"FG_API_KEY": key})
        r = client.get("/corpus", headers={"Authorization": f"{scheme} {key}"})
        assert r.status_code == 200

    def test_health_and_metrics_open(self, stub_models, monkeypatch):
        """Health + metrics stay open even with auth on (for load balancers)."""
        client = make_client(monkeypatch, {"FG_API_KEY": "secret123"})
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

    def test_protected_post_endpoints(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {"FG_API_KEY": "secret123"})
        r = client.post("/detect", json={"answer": "x", "passages": ["y"]})
        assert r.status_code == 401
        r = client.post("/detect", json={"answer": "x", "passages": ["y"]},
                        headers={"x-api-key": "secret123"})
        assert r.status_code == 200


class TestRateLimit:
    def test_rate_limit_enforced(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {
            "FG_API_KEY": None,          # auth off so key isn't the limiter key
            "FG_RATE_LIMIT": "3",
            "FG_RATE_WINDOW_S": "60",
        })
        codes = [client.get("/corpus").status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert 429 in codes[3:]

    def test_rate_limit_off_by_default(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {"FG_RATE_LIMIT": None})
        codes = [client.get("/corpus").status_code for _ in range(10)]
        assert all(c == 200 for c in codes)

    def test_429_has_retry_after(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {"FG_RATE_LIMIT": "1", "FG_RATE_WINDOW_S": "60"})
        client.get("/corpus")
        r = client.get("/corpus")
        assert r.status_code == 429
        assert "Retry-After" in r.headers


class TestObservability:
    def test_request_id_generated_and_echoed(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {})
        r = client.get("/health")
        assert "x-request-id" in r.headers
        assert len(r.headers["x-request-id"]) == 16

    def test_request_id_propagated_when_provided(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {})
        r = client.get("/health", headers={"x-request-id": "myrid123"})
        assert r.headers["x-request-id"] == "myrid123"

    def test_metrics_counts_requests(self, stub_models, monkeypatch):
        client = make_client(monkeypatch, {})
        client.get("/corpus")
        client.get("/corpus")
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "faithguard_requests_total" in r.text
