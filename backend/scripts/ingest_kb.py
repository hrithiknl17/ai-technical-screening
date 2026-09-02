"""Knowledge-base ingestion CLI.

    python -m scripts.ingest_kb --role ai_ml_engineer
    python -m scripts.ingest_kb --all --max-pages 220 --skip-leading 12
    python -m scripts.ingest_kb --status

`--max-pages` / `--skip-leading` exist because these corpora are whole books:
they let you index the substantive chapters of a 700-page text without paying to
embed its front matter and index. Ingestion rebuilds a role's collection from
whatever is in `data/knowledge_base/<role>/`, so it is safe to re-run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.db.session import init_db  # noqa: E402
from app.domain.roles import ROLES  # noqa: E402
from app.rag.ingest import IngestionPipeline  # noqa: E402
from app.rag.loaders import iter_corpus_files  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

logger = logging.getLogger("ingest")


def show_status() -> None:
    settings = get_settings()
    store = get_vector_store()
    print(f"knowledge base: {settings.knowledge_base_dir}")
    print(f"vector store:   {settings.vector_store_dir}\n")
    for slug, role in ROLES.items():
        files = iter_corpus_files(settings.knowledge_base_dir / slug)
        meta = store.meta(slug)
        print(f"{role.title}  ({slug})")
        print(f"  files on disk : {len(files)}")
        for path in files:
            print(f"      - {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
        print(f"  indexed chunks: {meta.get('count', 0)}  dim={meta.get('dimension', 0)}")
        print(f"  last indexed  : {meta.get('updated_at', 'never')}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest role corpora into the vector store.")
    parser.add_argument("--role", action="append", choices=sorted(ROLES), help="role slug (repeatable)")
    parser.add_argument("--all", action="store_true", help="ingest every configured role")
    parser.add_argument("--status", action="store_true", help="show what is on disk and indexed")
    parser.add_argument("--max-pages", type=int, default=None, help="cap pages per document")
    parser.add_argument("--skip-leading", type=int, default=0, help="skip N pages of front matter")
    args = parser.parse_args()

    configure_logging("INFO")
    init_db()

    if args.status:
        show_status()
        return 0

    roles = sorted(ROLES) if args.all else (args.role or [])
    if not roles:
        parser.error("pass --role <slug>, --all, or --status")

    pipeline = IngestionPipeline()
    for slug in roles:
        started = time.perf_counter()
        logger.info("ingesting role %s ...", slug)
        report = pipeline.ingest_role(
            slug, max_pages_per_doc=args.max_pages, skip_leading_pages=args.skip_leading
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "%s: %s docs, %s pages, %s/%s chunks indexed in %.1fs",
            slug,
            report.documents,
            report.pages,
            report.chunks_indexed,
            report.chunks_seen,
            elapsed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
