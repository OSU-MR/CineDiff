#!/usr/bin/env python
"""Download the CineDiff data and model weights from the GitHub release.

Run from the repository folder::

    python download.py

The data is unpacked into data/ and the weights saved as
weights/model-192000.pt. Both downloads are checked against their SHA256
before use; data and weights that are already present are not downloaded
again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE = "https://github.com/OSU-MR/CineDiff/releases/download/data%26weights"
DATA_URL = f"{RELEASE}/data.zip"
DATA_SHA256 = "5f1e963fc1f09b3094c41f88672f625c3d76dbdd9f767210ab34a87fafcea529"
WEIGHTS_URL = f"{RELEASE}/model-192000.pt"
WEIGHTS_SHA256 = "53b9239eb68eeb021115bedd7c21d4d8a94864eef7c05f1069a13a9743dc1535"
WEIGHTS_PATH = ROOT / "weights" / "model-192000.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, target: Path, expected_sha256: str) -> None:
    """Download ``url`` to ``target`` and check its SHA256."""
    print(f"Downloading {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, target.open("wb") as file:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        for block in iter(lambda: response.read(1024 * 1024), b""):
            file.write(block)
            digest.update(block)
            received += len(block)
            if total:
                print(f"\r  {received / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"{url} does not match the expected checksum")


def _data_is_complete() -> bool:
    """Check that every file listed in data/cases.json is present."""
    cases_path = ROOT / "data" / "cases.json"
    if not cases_path.is_file():
        return False
    for case in json.loads(cases_path.read_text()):
        names = ["kspace.npz", "max_mag.npy", "cs_S.mat", "CS_recon.mat"]
        if case.get("reference_available"):
            names.append("CS_recon_full.mat")
        if not all((ROOT / case["prepared"] / name).is_file() for name in names):
            return False
    return True


def download_weights() -> None:
    if WEIGHTS_PATH.is_file() and _sha256(WEIGHTS_PATH) == WEIGHTS_SHA256:
        print("weights/model-192000.pt is already present and correct")
        return
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = WEIGHTS_PATH.with_name(WEIGHTS_PATH.name + ".download")
    try:
        _fetch(WEIGHTS_URL, partial, WEIGHTS_SHA256)
        partial.replace(WEIGHTS_PATH)
    finally:
        partial.unlink(missing_ok=True)
    print("Saved weights/model-192000.pt")


def download_data() -> None:
    if _data_is_complete():
        print("data/ is already present")
        return
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        archive = Path(temporary) / "data.zip"
        _fetch(DATA_URL, archive, DATA_SHA256)
        with zipfile.ZipFile(archive) as zipped:
            print("Unpacking into data/")
            zipped.extractall(ROOT)
    if not _data_is_complete():
        raise RuntimeError("data/ is still incomplete after unpacking data.zip")
    print("data/ is complete")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-only", action="store_true", help="Download only the data")
    parser.add_argument("--weights-only", action="store_true", help="Download only the weights")
    args = parser.parse_args(argv)
    if args.data_only and args.weights_only:
        parser.error("choose at most one of --data-only and --weights-only")
    try:
        if not args.data_only:
            download_weights()
        if not args.weights_only:
            download_data()
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
