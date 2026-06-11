"""Test datasets for NLP analysis across languages."""

import json
from pathlib import Path


def load_dataset(name: str) -> dict:
    """Load a test dataset by name.

    Args:
        name: Dataset name (e.g., "modern_chinese", "classical_chinese", "english")

    Returns:
        Dict with 'name', 'lang', and 'cases' keys.
    """
    datasets_dir = Path(__file__).parent
    path = datasets_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_datasets() -> list[str]:
    """List available dataset names."""
    datasets_dir = Path(__file__).parent
    return [f.stem for f in datasets_dir.glob("*.json")]