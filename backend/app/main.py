"""Application entry point: wiring, cross-cutting concerns, nothing else."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import health, knowledge, resumes, roles, sessions
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.db.session import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    logger.info(
        "%s starting (env=%s, llm=%s)",
        settings.app_name,
        settings.app_env,
        settings.llm_model if settings.llm_enabled else "offline stub",
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "RAG-driven technical screening: resume in, grounded interview out. "
            "Every generated question carries the corpus passages it was built from."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "%s %s -> %s in %.0fms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    # --- error handling: domain errors become predictable JSON -------------
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "The request body failed validation.",
                "details": {"errors": exc.errors()[:10]},
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "Something went wrong on our side.",
                "details": {},
            },
        )

    for router in (health.router, roles.router, resumes.router, sessions.router, knowledge.router):
        app.include_router(router, prefix="/api")

    # Single-origin deployment: if a static export of the frontend exists, serve
    # it from "/". Mounted last, so every /api route is matched before it.
    if settings.static_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=settings.static_dir, html=True), name="frontend"
        )
        logger.info("serving the built frontend from %s", settings.static_dir)

    return app


app = create_app()
