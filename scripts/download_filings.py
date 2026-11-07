"""Download the public filings used in the evaluation.

The PDFs are public SEC filings, distributed with the FinanceBench benchmark
(https://github.com/patronus-ai/financebench, Islam et al., 2023). They are not stored in this
repository; this script fetches only the files needed with a sparse git checkout.

    python scripts/download_filings.py                       # dev set: 4 filings -> data/filings
    python scripts/download_filings.py --set heldout         # 5 held-out filings -> data/filings_heldout
    python scripts/download_filings.py --docs NETFLIX_2017_10K --out data/filings
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/patronus-ai/financebench"
SETS = {
    "dev": ["AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K", "PEPSICO_2022_10K"],
    "heldout": ["3M_2022_10K", "CVSHEALTH_2022_10K", "JOHNSON_JOHNSON_2022_10K", "BESTBUY_2023_10K", "AES_2022_10K"],
}
ROOT = Path(__file__).resolve().parents[1]


def run(*cmd: str, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="dev")
    ap.add_argument("--docs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    docs = a.docs or SETS[a.set]
    out = Path(a.out or ROOT / "data" / ("filings" if a.set == "dev" else "filings_heldout"))
    out.mkdir(parents=True, exist_ok=True)
    todo = [d for d in docs if not (out / f"{d}.pdf").exists()]
    if not todo:
        print("All filings already present.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "fb"
        run("git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", REPO, str(repo))
        run("git", "sparse-checkout", "set", "--no-cone", *[f"/pdfs/{d}.pdf" for d in todo], cwd=repo)
        run("git", "checkout", "main", cwd=repo)
        for d in todo:
            src = repo / "pdfs" / f"{d}.pdf"
            if not src.exists():
                print(f"Not found in FinanceBench: {d}")
                continue
            shutil.copy(src, out / src.name)
            print(f"Downloaded {src.name} ({src.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
