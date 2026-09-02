"""Role registry.

A role is the unit that ties together (a) the corpus used for retrieval,
(b) the competency areas an interviewer is expected to cover, and (c) the tone
of the questions. Adding a role means adding an entry here and dropping the
corresponding documents in `data/knowledge_base/<slug>/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.errors import NotFoundError


@dataclass(frozen=True)
class Role:
    slug: str
    title: str
    description: str
    #: Competency areas the interview should try to cover, in rough priority order.
    competencies: list[str] = field(default_factory=list)
    #: Books/corpus shipped for this role (informational; ingestion globs the folder).
    corpus: list[str] = field(default_factory=list)
    #: Seed queries used when a resume is thin on signal.
    fallback_queries: list[str] = field(default_factory=list)


ROLES: dict[str, Role] = {
    "ai_ml_engineer": Role(
        slug="ai_ml_engineer",
        title="AI/ML Engineer",
        description=(
            "Builds and ships machine-learning systems: framing the learning problem, "
            "choosing and training models, and reasoning about generalisation."
        ),
        competencies=[
            "Learning problem formulation and inductive bias",
            "Supervised learning algorithms and their assumptions",
            "Generalisation, overfitting, bias-variance trade-off",
            "Model evaluation and experiment design",
            "Neural networks and representation learning",
            "Practical ML engineering trade-offs",
        ],
        corpus=[
            "Machine Learning — Tom Mitchell",
            "The Hundred-Page Machine Learning Book — Andriy Burkov",
            "Machine Learning for Absolute Beginners",
        ],
        fallback_queries=[
            "inductive bias and hypothesis space in concept learning",
            "overfitting and how to avoid it in decision tree learning",
            "backpropagation and gradient descent for neural networks",
            "evaluating hypotheses: confidence intervals and cross validation",
        ],
    ),
    "data_scientist": Role(
        slug="data_scientist",
        title="Data Scientist / Applied ML",
        description=(
            "Turns data into decisions: feature engineering, applied modelling with "
            "scikit-learn style tooling, and honest evaluation."
        ),
        competencies=[
            "Data preparation, feature engineering and leakage",
            "Applied supervised and unsupervised modelling",
            "Model selection, hyper-parameter search and validation",
            "Metrics, uncertainty and communicating results",
            "Algorithm internals (linear models, trees, ensembles)",
        ],
        corpus=[
            "Introduction to Machine Learning with Python",
            "Master Machine Learning Algorithms — Jason Brownlee",
        ],
        fallback_queries=[
            "preprocessing, scaling and feature engineering pipelines",
            "cross validation and grid search for model selection",
            "evaluation metrics for imbalanced classification",
            "how gradient descent fits linear and logistic regression",
        ],
    ),
    "ml_researcher": Role(
        slug="ml_researcher",
        title="ML Researcher (Theoretical)",
        description=(
            "Reasons from first principles: probabilistic modelling, the mathematics "
            "of inference, and the theory behind deep architectures."
        ),
        competencies=[
            "Probabilistic modelling and Bayesian inference",
            "Linear models for regression and classification, from theory",
            "Kernel methods, graphical models and latent-variable models",
            "Approximate inference and optimisation theory",
            "Deep learning architectures and their theoretical motivation",
        ],
        corpus=[
            "Pattern Recognition and Machine Learning — Christopher Bishop",
            "Artificial Intelligence, Machine Learning & Deep Learning",
        ],
        fallback_queries=[
            "bayesian inference and conjugate priors",
            "bias variance decomposition for regression",
            "kernel methods and the dual representation",
            "expectation maximisation for latent variable models",
        ],
    ),
}


def list_roles() -> list[Role]:
    return list(ROLES.values())


def get_role(slug: str) -> Role:
    try:
        return ROLES[slug]
    except KeyError as exc:
        raise NotFoundError(
            f"Unknown role '{slug}'.", details={"available": sorted(ROLES)}
        ) from exc
