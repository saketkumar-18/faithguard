"""Detection layer: claim extraction, NLI scoring, features, classifier."""
from .claims import extract_claims
from .nli import NLIScorer, ClaimEvidenceScore
from .features import build_features, FEATURE_NAMES
from .classifier import HallucinationClassifier, AnswerVerdict

__all__ = [
    "extract_claims",
    "NLIScorer",
    "ClaimEvidenceScore",
    "build_features",
    "FEATURE_NAMES",
    "HallucinationClassifier",
    "AnswerVerdict",
]
