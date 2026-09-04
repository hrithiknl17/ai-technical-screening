"""Unit tests for the RAG building blocks."""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.chunking import chunk_document
from app.rag.embeddings import HashingEmbedder
from app.rag.ingest import chunk_rejection_reason, is_useful_chunk
from app.rag.loaders import LoadedDocument, LoadedPage, clean_text
from app.rag.retriever import BM25Index, HybridRetriever
from app.rag.vector_store import ChunkRecord, NumpyVectorStore


def test_clean_text_repairs_pdf_artifacts():
    raw = "gener-\nalisation is the goal­ of ﬁtting"
    assert clean_text(raw) == "generalisation is the goal of fitting"


def test_chunking_respects_size_and_overlap():
    page_text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(12))
    document = LoadedDocument(source="x.pdf", pages=[LoadedPage(number=3, text=page_text)])
    chunks = chunk_document(document, chunk_size=800, chunk_overlap=120)

    assert len(chunks) > 1
    assert all(len(c.text) <= 1100 for c in chunks)
    assert all(c.page_start == 3 and c.page_end == 3 for c in chunks)
    # consecutive chunks share text, i.e. the window really overlaps
    assert any(chunks[0].text[-60:] in chunks[1].text for _ in [0])


def test_quality_filter_rejects_front_matter_and_keeps_prose():
    toc = "Introduction .......... 4\nChapter 2 .......... 19\nChapter 3 .......... 44\n" * 3
    assert not is_useful_chunk(toc)
    assert not is_useful_chunk("too short")
    prose = (
        "A hypothesis overfits the training data when another hypothesis fits the "
        "training examples less well yet performs better over the entire distribution "
        "of instances, which is the situation practitioners try hard to avoid when "
        "they grow decision trees to full depth without pruning them afterwards."
    )
    assert is_useful_chunk(prose)


def test_quality_filter_names_the_reason_it_rejected_a_chunk():
    assert chunk_rejection_reason("too short") == "too_short"
    # equation soup: enough words, but mostly digits and operators
    assert chunk_rejection_reason("xy = (3*4)/(7-2)^2 + " * 40) == "low_alpha_ratio"

    contents = (
        "Introduction to statistical learning theory .......... 4\n"
        "Decision tree learning and inductive bias .......... 19\n"
        "Evaluating hypotheses with confidence intervals .......... 44\n"
        "Artificial neural networks and backpropagation .......... 81\n"
        "Bayesian learning and minimum description length .......... 154\n"
        "Computational learning theory and sample complexity .......... 201\n"
        "Instance based learning and locally weighted regression .......... 230\n"
        "Genetic algorithms and the hypothesis space search .......... 249\n"
    )
    assert chunk_rejection_reason(contents) == "toc_dot_leader"

    bibliography = (
        "Breiman, L. (1996). Bagging predictors. Machine Learning, 24.\n"
        "Mitchell, T. (1997). Machine Learning. McGraw Hill, New York.\n"
        "Quinlan, J. R. (1986). Induction of decision trees. Machine Learning.\n"
        "Cover, T. and Hart, P. (1967). Nearest neighbor pattern classification.\n"
        "Vapnik, V. (1995). The nature of statistical learning theory. Springer.\n"
        "Freund, Y. and Schapire, R. (1997). A decision theoretic generalization.\n"
    )
    assert chunk_rejection_reason(bibliography) == "reference_list"

    prose = (
        "Backpropagation searches a hypothesis space of continuous weight values by "
        "gradient descent on a differentiable error surface, and is guaranteed only "
        "to converge to some local minimum rather than the global one, so momentum "
        "and multiple random restarts are the usual practical remedies."
    )
    assert chunk_rejection_reason(prose) is None


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(64)
    first = embedder.embed(["decision tree entropy"])
    second = embedder.embed(["decision tree entropy"])
    assert np.allclose(first, second)
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(first[0]))
    other = embedder.embed(["completely unrelated cooking recipe"])
    assert float(first[0] @ other[0]) < float(first[0] @ second[0])


def test_bm25_ranks_the_document_containing_the_term():
    index = BM25Index(
        [
            "gradient descent minimises the error surface",
            "decision trees split on information gain",
            "bayesian inference uses conjugate priors",
        ]
    )
    scores = index.scores("information gain splitting")
    assert int(np.argmax(scores)) == 1


def test_vector_store_round_trip(tmp_path):
    store = NumpyVectorStore(tmp_path)
    records = [
        ChunkRecord(id="a", text="alpha", source="book.pdf", page_start=1, page_end=1),
        ChunkRecord(id="b", text="beta", source="book.pdf", page_start=2, page_end=2),
    ]
    vectors = np.eye(2, dtype=np.float32)
    store.upsert("role", records, vectors)

    assert store.count("role") == 2
    hits = store.search("role", np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert hits[0].record.id == "a"
    assert hits[0].score > hits[1].score
    assert "book.pdf, p. 1" == hits[0].citation()


def test_hybrid_retrieval_returns_grounded_diverse_chunks(seeded_role):
    retriever = HybridRetriever()
    result = retriever.retrieve(seeded_role, "how to avoid overfitting a decision tree", top_k=2)

    assert result.chunks, "retrieval returned nothing"
    assert "overfit" in result.chunks[0].record.text.lower()
    # MMR must not return the same chunk twice
    assert len({c.record.id for c in result.chunks}) == len(result.chunks)
    trace = result.trace()
    assert trace["query"] and trace["chunks"][0]["citation"]


def test_retrieve_many_never_repeats_a_chunk(seeded_role):
    retriever = HybridRetriever()
    results = retriever.retrieve_many(
        seeded_role,
        ["overfitting in decision trees", "cross validation confidence intervals"],
        per_query_k=2,
    )
    seen = [c.record.id for r in results for c in r.chunks]
    assert len(seen) == len(set(seen))
