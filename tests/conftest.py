"""Shared fixtures: a tiny corpus + pipeline pieces that don't need model downloads."""
from __future__ import annotations

import pytest

from faithguard.retrieval.chunking import Chunker, Chunk


TINY_DOCS = [
    {
        "id": "doc-taj",
        "title": "Taj Mahal",
        "text": (
            "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the "
            "river Yamuna in Agra, India. It was commissioned in 1631 by the Mughal "
            "emperor Shah Jahan to house the tomb of his favourite wife, Mumtaz Mahal. "
            "Construction of the mausoleum was essentially completed in 1648. The Taj "
            "Mahal was designated a UNESCO World Heritage Site in 1983. It attracts "
            "around 8 million visitors a year."
        ),
    },
    {
        "id": "doc-eiffel",
        "title": "Eiffel Tower",
        "text": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
            "Paris, France. It is named after the engineer Gustave Eiffel, whose company "
            "designed and built the tower from 1887 to 1889. The tower is 330 metres "
            "tall and was the world's tallest man-made structure until 1930. It receives "
            "about 7 million visitors annually."
        ),
    },
]


@pytest.fixture(scope="session")
def tiny_chunks() -> list[Chunk]:
    chunker = Chunker(chunk_size=400, chunk_overlap=50)
    return chunker.chunk_documents(TINY_DOCS)


@pytest.fixture(scope="session")
def bm25_only_retriever(tiny_chunks):
    from faithguard.retrieval.hybrid import HybridRetriever

    return HybridRetriever(tiny_chunks, use_dense=False)
