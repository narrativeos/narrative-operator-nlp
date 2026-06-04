#!/usr/bin/env python3
"""
Model Setup & Verification Script.

Ensures all required HanLP pre-trained models are downloaded and verified.

Usage:
    python scripts/setup_models.py              # Download all models
    python scripts/setup_models.py --check      # Check-only, no download
    python scripts/setup_models.py --model MTL  # Download specific model set
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root on path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelSpec:
    """Describes a single HanLP pre-trained model."""

    def __init__(
        self,
        name: str,
        hanlp_ref: str,
        description: str,
        required: bool = True,
        category: str = "modern",
    ):
        self.name = name
        self.hanlp_ref = hanlp_ref  # e.g., hanlp.pretrained.mtl.CLOSE_...
        self.description = description
        self.required = required
        self.category = category


MODELS: list[ModelSpec] = [
    # --- Modern Chinese (required) ---
    ModelSpec(
        "MTL Modern Chinese (ELECTRA-small)",
        "CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH",
        "Multi-task: tokenization, POS, NER, dependency, SRL, constituency. "
        "Default engine for modern Chinese.",
        required=True,
        category="mtl",
    ),
    # --- Classical Chinese (required for hybrid pipeline) ---
    ModelSpec(
        "Classical Chinese (KYOTO-EVAHAN)",
        "KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH",
        "Classical/literary Chinese: tokenization, lemmatization, POS, dependency. "
        "Used by the two-stage hybrid pipeline.",
        required=False,  # Optional: hybrid pipeline degrades gracefully
        category="lzh",
    ),
    # --- Single-task models for Pipeline mode (optional) ---
    ModelSpec(
        "Fine Tokenizer",
        "FINE_ELECTRA_SMALL_ZH",
        "Single-task fine-grained tokenizer for pipeline mode.",
        required=False,
        category="pipeline",
    ),
    ModelSpec(
        "CTB9 POS Tagger",
        "CTB9_POS_ELECTRA_SMALL",
        "Single-task POS tagger (CTB tagset) for pipeline mode.",
        required=False,
        category="pipeline",
    ),
    ModelSpec(
        "MSRA NER",
        "MSRA_NER_ELECTRA_SMALL_ZH",
        "Single-task NER (MSRA tagset) for pipeline mode.",
        required=False,
        category="pipeline",
    ),
    ModelSpec(
        "CTB9 Dependency Parser",
        "CTB9_DEP_ELECTRA_SMALL",
        "Single-task dependency parser (CTB) for pipeline mode.",
        required=False,
        category="pipeline",
    ),
]


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_model(spec: ModelSpec, verbose: bool = True) -> tuple[bool, str]:
    """
    Load a HanLP model by its pretrained reference string.
    Returns (success, message).
    """
    import hanlp

    start = time.time()
    try:
        if verbose:
            print(f"  Loading: {spec.name} ...", end=" ", flush=True)
        hanlp.load(spec.hanlp_ref, verbose=False)
        elapsed = time.time() - start
        msg = f"OK ({elapsed:.1f}s)"
        if verbose:
            print(msg)
        return True, msg
    except Exception as exc:
        elapsed = time.time() - start
        msg = f"FAILED ({elapsed:.1f}s): {exc}"
        if verbose:
            print(msg)
        return False, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_models(
    categories: set[str] | None = None,
    check_only: bool = False,
    verbose: bool = True,
) -> dict[str, bool]:
    """
    Ensure all required models are available.

    Args:
        categories: If set, only load models from these categories
                    (e.g., {"mtl"}, {"mtl","lzh"}, {"mtl","lzh","pipeline"}).
        check_only: If True, only check existence without downloading.
        verbose: Print progress messages.

    Returns:
        {model_name: success} dict.
    """
    results: dict[str, bool] = {}

    if verbose:
        mode = "CHECKING" if check_only else "SETTING UP"
        print(f"\n{'='*60}")
        print(f"  Narrative Operator NLP — Model {mode}")
        print(f"{'='*60}\n")

    if check_only:
        # Quick existence check via hanlp.pretrained module
        try:
            import hanlp.pretrained.mtl as mtl
            import hanlp.pretrained.tok as tok_mod
            import hanlp.pretrained.ner as ner_mod
            import hanlp.pretrained.dep as dep_mod
            import hanlp.pretrained.pos as pos_mod

            _pretrained_map = {
                "CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH": mtl,
                "KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH": mtl,
                "FINE_ELECTRA_SMALL_ZH": tok_mod,
                "CTB9_POS_ELECTRA_SMALL": pos_mod,
                "MSRA_NER_ELECTRA_SMALL_ZH": ner_mod,
                "CTB9_DEP_ELECTRA_SMALL": dep_mod,
            }

            for spec in MODELS:
                if categories and spec.category not in categories:
                    continue
                if spec.hanlp_ref in _pretrained_map:
                    results[spec.name] = True
                    if verbose:
                        print(f"  ✅ {spec.name} — module found")
                else:
                    results[spec.name] = False
                    if verbose:
                        print(f"  ⚠️  {spec.name} — module not found in pretrained")
        except ImportError:
            if verbose:
                print("  ⚠️  hanlp not installed; cannot check models.")
            return {}

        if verbose:
            required_ok = all(
                results.get(s.name, True) for s in MODELS if s.required
            )
            print(f"\n  {'✅ All required models found' if required_ok else '⚠️  Some models missing'}")
        return results

    # --- Full download ---
    for spec in MODELS:
        if categories and spec.category not in categories:
            continue
        ok, _ = load_model(spec, verbose=verbose)
        results[spec.name] = ok

    # Summary
    if verbose:
        print(f"\n{'─'*60}")
        total = len(results)
        ok_count = sum(1 for v in results.values() if v)
        required_ok = all(
            results.get(s.name, True) for s in MODELS if s.required
        )
        print(f"  Downloaded: {ok_count}/{total}")
        print(f"  Required models: {'✅ READY' if required_ok else '❌ MISSING'}")
        print(f"{'─'*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Narrative Operator NLP — Model Setup"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check-only mode: verify model availability without downloading.",
    )
    parser.add_argument(
        "--model",
        choices=["MTL", "LZH", "PIPELINE", "ALL"],
        default="ALL",
        help="Which model set to set up (default: ALL).",
    )
    args = parser.parse_args()

    category_map = {
        "MTL": {"mtl"},
        "LZH": {"lzh"},
        "PIPELINE": {"pipeline"},
        "ALL": {"mtl", "lzh", "pipeline"},
    }

    setup_models(
        categories=category_map[args.model],
        check_only=args.check,
        verbose=True,
    )
