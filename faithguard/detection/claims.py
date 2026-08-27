"""Claim extraction: split an answer into atomic, checkable statements.

Rule-based (no LLM dependency) so detection stays fast, deterministic, and
free. Handles the common answer shapes seen in RAG: short factual answers,
multi-sentence explanations, and list-style answers.
"""
from __future__ import annotations

import re

_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONJUNCTION = re.compile(r"\s+(?:and|;|, and|, which|, where|, while)\s+", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_FILLER = re.compile(
    r"^(?:based on the (?:provided |given )?(?:context|passage|document)s?[,]?\s*"
    r"|according to the (?:provided |given )?(?:context|passage|document)s?[,]?\s*"
    r"|(?:the )?(?:context|passage|document)s? (?:states?|says?|mentions?|indicates?|provides?) that\s*"
    r"|it is (?:stated|mentioned|noted) that\s*)",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(?:may|might|could|possibly|perhaps|probably|likely|approximately|about|around|roughly)\b",
    re.IGNORECASE,
)


def _clean(sentence: str) -> str:
    s = _LIST_ITEM.sub("", sentence.strip())
    s = _FILLER.sub("", s).strip()
    return s.strip()


def extract_claims(answer: str, min_chars: int = 15) -> list[dict]:
    """Split an answer into atomic claims.

    Returns a list of dicts: {"text": str, "hedged": bool}.
    Hedged claims (may/might/approximately) are still returned but flagged —
    the classifier can treat them as lower-risk.
    """
    if not answer or not answer.strip():
        return []
    claims: list[dict] = []
    for sentence in _CLAUSE_SPLIT.split(answer.strip()):
        sentence = _clean(sentence)
        if not sentence:
            continue
        # split long compound sentences into sub-claims
        parts = _CONJUNCTION.split(sentence) if len(sentence) > 120 else [sentence]
        for part in parts:
            part = part.strip(" ,;:")
            if len(part) < min_chars:
                # too short to check on its own — merge with previous if possible
                if claims and len(claims[-1]["text"]) < 200:
                    claims[-1]["text"] = claims[-1]["text"] + " " + part
                    claims[-1]["hedged"] = claims[-1]["hedged"] or bool(_HEDGE.search(part))
                continue
            claims.append({"text": part, "hedged": bool(_HEDGE.search(part))})
    if not claims:
        # Very short answers ("McCrary", "1976") produce no claims above
        # min_chars. Fall back to the whole cleaned answer as one claim so
        # the detector still gets real evidence instead of a degenerate
        # zero-claim input (which the classifier reads as hallucination).
        whole = _clean(answer.strip()).strip(" ,;:.")
        if whole:
            claims.append({"text": whole, "hedged": bool(_HEDGE.search(whole))})
    return claims
