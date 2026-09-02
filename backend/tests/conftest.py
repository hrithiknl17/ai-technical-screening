"""Test fixtures.

Tests run fully offline: `OFFLINE_MODE=1` swaps in the hashing embedder and the
stub LLM, so the same code paths execute without an API key or network access.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# Set before anything imports `app.*`: `app.db.session` builds its engine at
# import time, and pytest imports every test module during collection - so an
# environment set from a fixture would arrive too late and the suite would run
# against the developer's real database.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="interviewer-tests-"))
os.environ.update(
    {
        "OFFLINE_MODE": "1",
        "GEMINI_API_KEY": "",
        "DATABASE_URL": f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}",
        "KNOWLEDGE_BASE_DIR": str(_TEST_ROOT / "kb"),
        "VECTOR_STORE_DIR": str(_TEST_ROOT / "vectors"),
        "EMBEDDING_PAUSE_SECONDS": "0",
    }
)


@pytest.fixture(scope="session")
def settings():
    from app.core.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def seeded_role(settings) -> str:
    """Index a tiny in-memory corpus for a real role slug."""
    role = "ai_ml_engineer"
    corpus = settings.knowledge_base_dir / role
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "notes.md").write_text(
        "\n\n".join(
            [
                "Overfitting and generalisation. A hypothesis overfits the training data "
                "when another hypothesis exists that fits the training examples less well "
                "but performs better over the entire distribution of instances. Decision "
                "tree learning avoids overfitting by stopping growth early or by "
                "post-pruning the fully grown tree using a separate validation set, which "
                "reduces variance at the cost of a small amount of bias in the estimate.",
                "Evaluating hypotheses. The sample error of a hypothesis over a sample is "
                "an estimator of its true error over the whole distribution. Because the "
                "estimate is a random variable, confidence intervals quantify how far the "
                "true error may lie from the observed one, and k-fold cross validation "
                "reduces the variance of that estimate by averaging over several folds of "
                "held out data drawn from the same distribution.",
                "Neural networks and gradient descent. Backpropagation searches a "
                "hypothesis space of continuous weight values by gradient descent on a "
                "differentiable error surface. It is guaranteed only to converge to a local "
                "minimum, and momentum, weight decay and multiple random restarts are the "
                "usual practical remedies employed by practitioners training such networks.",
            ]
        ),
        encoding="utf-8",
    )
    from app.db.session import init_db
    from app.rag.ingest import IngestionPipeline

    init_db()
    IngestionPipeline().ingest_role(role, progress=False)
    return role


@pytest.fixture
def client(seeded_role):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


RESUME_TEXT = """
Asha Menon
asha.menon@example.com | github.com/ashamenon

Machine Learning Engineer with production experience 2021 - 2024.
Built retrieval-augmented search over 4M documents using Python, PyTorch and FAISS.
Trained gradient boosted trees with scikit-learn for churn prediction and ran
A/B testing to validate lift. Deployed models with FastAPI and Docker on AWS.
Comfortable with cross validation, feature engineering and model evaluation.
"""
