#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CBDB Database Sync Script

Checks for updates from HuggingFace and downloads if available.

Usage:
    python scripts/sync_cbdb.py
"""

import json
import hashlib
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

try:
    import requests
except ImportError:
    requests = None


def get_latest_info():
    """Get latest CBDB version info from GitHub."""
    url = "https://raw.githubusercontent.com/cbdb-project/cbdb_sqlite/master/latest.json"
    try:
        if requests:
            response = requests.get(url, timeout=30)
            return response.json()
        else:
            with urlopen(url) as f:
                return json.loads(f.read().decode("utf-8"))
    except Exception as e:
        print(f"Failed to fetch latest info: {e}")
        return None


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    db_dir = Path(__file__).parent.parent / "config" / "classical" / "databases"
    db_path = db_dir / "cbdb.sqlite"
    metadata_path = db_dir / "metadata.json"

    # Load current metadata
    current_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            current_metadata = json.load(f)

    # Get latest info
    print("Checking for CBDB updates...")
    latest = get_latest_info()
    if not latest:
        print("Failed to get latest version info.")
        sys.exit(1)

    latest_filename = latest.get("sqlite_filename", "")
    latest_sha256 = latest.get("sha256", "")
    download_url = latest.get("download_url", "")

    # Compare versions
    current_version = current_metadata.get("version", "")
    if current_version == latest_filename:
        print(f"Already up to date: {current_version}")
        # Verify checksum
        if db_path.exists():
            current_sha256 = compute_sha256(db_path)
            if current_sha256 == latest_sha256:
                print("Checksum verified.")
            else:
                print(f"Checksum mismatch! Current: {current_sha256[:16]}..., Expected: {latest_sha256[:16]}...")
                print("Database may be corrupted, consider re-downloading.")
        return

    print(f"New version available: {latest_filename}")
    print(f"Current version: {current_version}")
    print(f"Download URL: {download_url}")
    print()
    print("To download manually:")
    print(f"  wget -O {db_path}.zip '{download_url}'")
    print(f"  unzip {db_path}.zip -d {db_dir}")
    print(f"  mv {db_dir}/{latest_filename} {db_path}")
    print()
    print("Or use the provided download script:")
    print(f"  python scripts/download_cbdb.py")


if __name__ == "__main__":
    main()