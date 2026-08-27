"""Prompt construction for grounded RAG generation and corrective regeneration."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer ONLY using the "
    "information in the provided context passages. Do not use outside knowledge. "
    "If the context does not contain the answer, say exactly: INSUFFICIENT_CONTEXT. "
    "Keep answers concise and factual. Never invent names, dates, or numbers."
)


def _format_passages(passages: list[str], titles: list[str] | None = None) -> str:
    lines = []
    for i, p in enumerate(passages, start=1):
        title = f" ({titles[i - 1]})" if titles and i - 1 < len(titles) and titles[i - 1] else ""
        lines.append(f"[{i}]{title} {p.strip()}")
    return "\n\n".join(lines)


def build_rag_prompt(question: str, passages: list[str], titles: list[str] | None = None) -> str:
    return (
        f"Context passages:\n{_format_passages(passages, titles)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. If the answer is not in the context, "
        "reply INSUFFICIENT_CONTEXT."
    )


def build_corrective_prompt(
    question: str,
    passages: list[str],
    previous_answer: str,
    unsupported_claims: list[str],
    contradicted_claims: list[str],
    titles: list[str] | None = None,
) -> str:
    """Re-generation prompt after re-retrieval: tells the model exactly what was wrong."""
    parts = [
        f"Context passages:\n{_format_passages(passages, titles)}",
        f"Question: {question}",
        f"A previous answer was generated but failed a faithfulness check:\n\"{previous_answer}\"",
    ]
    if unsupported_claims:
        parts.append(
            "These statements were NOT supported by any retrieved passage and must be "
            "removed or corrected:\n- " + "\n- ".join(unsupported_claims)
        )
    if contradicted_claims:
        parts.append(
            "These statements CONTRADICT the retrieved passages and must be corrected:\n- "
            + "\n- ".join(contradicted_claims)
        )
    parts.append(
        "Write a corrected answer using ONLY the context passages above. "
        "Drop anything not supported. If the context still does not contain the answer, "
        "reply INSUFFICIENT_CONTEXT."
    )
    return "\n\n".join(parts)
