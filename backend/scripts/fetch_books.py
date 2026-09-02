"""Download the assignment's reference books into the knowledge-base folders.

    python -m scripts.fetch_books            # everything that is missing
    python -m scripts.fetch_books --role ml_researcher

The PDFs are not committed to the repository (they are ~90 MB and not ours to
redistribute), so this script reconstructs `data/knowledge_base/` from the public
URLs given in the assignment. Run it before `scripts.ingest_kb`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.domain.roles import ROLES  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.microsoft.com/"}

# role -> (filename, [mirrors])
BOOKS: dict[str, list[tuple[str, list[str]]]] = {
    "ai_ml_engineer": [
        (
            "mitchell_machine_learning.pdf",
            ["https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf"],
        ),
        (
            "ml_for_absolute_beginners.pdf",
            [
                "https://www.hlevkin.com/hlevkin/45MachineDeepLearning/ML/"
                "Machine%20Learning%20For%20Absolute%20Beginners.pdf"
            ],
        ),
        (
            # The host named in the assignment serves an invalid TLS certificate,
            # so this one usually fails; drop the PDF in by hand if you have it.
            "burkov_hundred_page_ml.pdf",
            [
                "https://ema.cri-info.cm/wp-content/uploads/2019/07/"
                "2019BurkovTheHundred-pageMachineLearning.pdf"
            ],
        ),
    ],
    "data_scientist": [
        (
            "intro_to_ml_with_python.pdf",
            [
                "https://www.nrigroupindia.com/e-book/Introduction%20to%20Machine%20"
                "Learning%20with%20Python%20%28%20PDFDrive.com%20%29-min.pdf"
            ],
        ),
        (
            "brownlee_master_ml_algorithms.pdf",
            [
                "https://github.com/khurrameycon/Data-Science-Books/raw/main/"
                "Master%20Machine%20Learning%20Algorithms%20-%20Discover%20how%20they%20"
                "work%20and%20Implement%20Them%20From%20Scratch%20by%20Jason%20Brownlee%20"
                "%28z-lib.org%29.pdf"
            ],
        ),
    ],
    "ml_researcher": [
        (
            "bishop_pattern_recognition_ml.pdf",
            [
                "https://www.microsoft.com/en-us/research/uploads/prod/2006/01/"
                "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"
            ],
        ),
        (
            "ai_ml_deep_learning.pdf",
            [
                "https://jcer.in/jcer-docs/E-Learning/Digital%20Library%20/E-Books/"
                "Artificial%20Intelligence%2C%20Machine%20Learning%2C%20and%20Deep%20Learning.pdf"
            ],
        ),
    ],
}

MIN_BYTES = 200_000


def download(url: str, target: Path) -> bool:
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=240, headers=HEADERS) as r:
            r.raise_for_status()
            with target.open("wb") as handle:
                for block in r.iter_bytes(65536):
                    handle.write(block)
    except Exception as exc:
        print(f"    failed: {type(exc).__name__}: {str(exc)[:110]}")
        target.unlink(missing_ok=True)
        return False
    if target.stat().st_size < MIN_BYTES or target.read_bytes()[:5] != b"%PDF-":
        print("    failed: response was not a PDF")
        target.unlink(missing_ok=True)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the reference corpora.")
    parser.add_argument("--role", action="append", choices=sorted(ROLES))
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    settings = get_settings()
    roles = args.role or sorted(BOOKS)
    failures: list[str] = []

    for role in roles:
        directory = settings.knowledge_base_dir / role
        directory.mkdir(parents=True, exist_ok=True)
        print(f"\n{role} -> {directory}")
        for filename, mirrors in BOOKS.get(role, []):
            target = directory / filename
            if target.exists() and target.stat().st_size > MIN_BYTES and not args.force:
                print(f"  {filename}: already present ({target.stat().st_size / 1e6:.1f} MB)")
                continue
            print(f"  {filename}: downloading …")
            if any(download(url, target) for url in mirrors):
                print(f"    ok ({target.stat().st_size / 1e6:.1f} MB)")
            else:
                failures.append(f"{role}/{filename}")

    if failures:
        print("\nCould not fetch:")
        for item in failures:
            print(f"  - {item}")
        print("Add these files manually, then run: python -m scripts.ingest_kb --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
