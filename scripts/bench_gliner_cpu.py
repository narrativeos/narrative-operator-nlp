#!/usr/bin/env python3
"""P0 go/no-go gate for the GLiNER second engine (CPU-only deployment).

Measures the four hard gates from docs/gliner_integration_design.md on the
machine that will actually serve traffic:

    1. latency      per-request wall time vs the HanLP MTL main pipeline
    2. memory       resident memory delta of loading the model
    3. concurrency  P95 under N parallel workers vs the single-request time
    4. quality      entity recall/precision on a labeled set

The gate is deliberately strict: if any number misses, the design says do not
ship the engine and fall back to a custom NER task inside the existing HanLP
MTL pipeline (zero new dependencies).

Nothing here imports GLiNER at module level, so the script stays runnable (and
gives an actionable message) on a machine where the optional dependency is not
installed — which is the normal state, since the engine is off by default.

Usage:
    python scripts/bench_gliner_cpu.py                       # latency + memory
    python scripts/bench_gliner_cpu.py --corpus texts.txt    # one text per line
    python scripts/bench_gliner_cpu.py --eval eval.jsonl     # quality gate
    python scripts/bench_gliner_cpu.py --workers 2           # concurrency gate

Eval file: JSON Lines, one object per line
    {"text": "...", "entities": [{"text": "碳纤维", "category": "MATERIAL"}]}
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# NSP category -> GLiNER label prompts (design doc section 4). Chinese labels
# first: on Chinese text the native labels recall better than English ones.
CATEGORY_LABELS: dict[str, list[str]] = {
    "MATERIAL": ["材料", "material"],
    "STANDARD": ["标准", "标准规范", "standard"],
    "PARAMETER": ["参数", "性能指标", "parameter"],
}

# Gate thresholds (design doc section 5). Miss any one -> do not ship.
GATE_LATENCY_RATIO = 1.30     # <= +30% over the MTL main pipeline
GATE_RSS_DELTA_MB = 300.0     # resident memory added by the model
GATE_P95_RATIO = 2.0          # P95 under concurrency vs single-request p95
GATE_RECALL_GAIN = 0.05       # >= +5 points recall on the labeled set
# Below this, wall-clock noise dominates and a ratio means nothing. A real
# model never lands here; a stub or a cached no-op does, and must not pass.
MIN_TIMING_MS = 1.0


def _die(msg: str, hint: str = "") -> int:
    print(f"GATE: NO-GO — {msg}")
    if hint:
        print(hint)
    return 2


def _load_gliner(model_id: str):
    """Import and load the engine, or return None with a clear reason."""
    try:
        from gliner import GLiNER  # type: ignore
    except ImportError:
        return None, ("gliner is not installed (optional dependency; the "
                      "engine is off by default)")
    try:
        return GLiNER.from_pretrained(model_id), ""
    except Exception as exc:  # offline, missing weights, bad id
        return None, f"could not load model '{model_id}': {exc}"


def _rss_mb() -> float:
    """Resident set size of this process in MB, or -1 if unavailable."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux kilobytes
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except Exception:
        return -1.0


def _read_texts(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _baseline_pipeline():
    """The HanLP MTL main pipeline, for the latency ratio."""
    from core.analyzer import _get_modern_pipeline
    return _get_modern_pipeline()


def _predict(engine, labels: list[str], text: str) -> list[dict]:
    return engine.predict_entities(text, labels, threshold=0.5)


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def measure_latency(engine, baseline, texts, labels) -> dict:
    """Warm up, then time both pipelines on the same texts."""
    for t in texts[:3]:
        _predict(engine, labels, t)
    gl_ms = []
    for t in texts:
        t0 = time.perf_counter()
        _predict(engine, labels, t)
        gl_ms.append((time.perf_counter() - t0) * 1000)

    base_ms: list[float] = []
    if baseline is not None:
        for t in texts:
            t0 = time.perf_counter()
            baseline(t)
            base_ms.append((time.perf_counter() - t0) * 1000)
    return {"gliner_p50": statistics.median(gl_ms),
            "gliner_p95": _pct(gl_ms, 0.95),
            "base_p50": statistics.median(base_ms) if base_ms else 0.0,
            "base_p95": _pct(base_ms, 0.95) if base_ms else 0.0}


def measure_concurrency(engine, texts, labels, workers: int) -> float:
    """P95 wall time per request when `workers` requests run at once."""
    def one(t):
        t0 = time.perf_counter()
        _predict(engine, labels, t)
        return (time.perf_counter() - t0) * 1000

    with ThreadPoolExecutor(max_workers=workers) as pool:
        times = list(pool.map(one, texts))
    return _pct(times, 0.95)


def _dictionary_terms(categories: list[str]) -> dict[str, set[str]]:
    """Terms from config/domain_keywords.yaml, keyed by uppercase category.

    Read straight from the YAML (same source KeywordExtractor loads) so the
    gate does not need the token/id_generator plumbing of the live pipeline.
    """
    import yaml

    path = _project_root / "config" / "domain_keywords.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {c: set(config.get(c.lower(), []) or []) for c in categories}


def measure_quality(engine, samples: list[dict], categories: list[str]) -> dict:
    """Span-level recall/precision of GLiNER alone vs dictionary + GLiNER."""
    labels = sorted({l for c in categories for l in CATEGORY_LABELS.get(c, [])})
    dicts = _dictionary_terms(categories)
    tp_g = fp_g = fn_g = 0
    tp_h = fp_h = fn_h = 0
    for s in samples:
        text, gold = s["text"], {e["text"] for e in s.get("entities", [])}
        g_pred = {e["text"] for e in _predict(engine, labels, text)}
        d_pred = {t for terms in dicts.values() for t in terms if t in text}
        both = g_pred | d_pred
        tp_g += len(g_pred & gold); fp_g += len(g_pred - gold); fn_g += len(gold - g_pred)
        tp_h += len(both & gold); fp_h += len(both - gold); fn_h += len(gold - both)

    def prf(tp, fp, fn):
        return (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)

    pg, rg = prf(tp_g, fp_g, fn_g)
    ph, rh = prf(tp_h, fp_h, fn_h)
    return {"gliner_recall": rg, "hybrid_recall": rh,
            "gliner_precision": pg, "hybrid_precision": ph,
            "n_samples": len(samples)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="urchade/gliner_multi-v2.1")
    ap.add_argument("--corpus", type=Path,
                    help="text file, one document per line (default: built-in samples)")
    ap.add_argument("--eval", type=Path, help="labeled JSON Lines set for the quality gate")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the HanLP MTL comparison (no model download)")
    args = ap.parse_args()

    if args.corpus:
        texts = _read_texts(args.corpus)
    else:
        texts = ["碳纤维复合材料在航空结构件上应用广泛，其疲劳性能需按 GB/T 3001 检测。",
                 "采用选择性激光熔化成型工艺时，层间温度控制决定制件致密度。",
                 "高温合金涡轮盘需进行无损检测，渗透检测用于表面裂纹筛查。"] * 7
    texts = texts[:args.rounds]
    labels = sorted({l for cats in CATEGORY_LABELS.values() for l in cats})

    engine, reason = _load_gliner(args.model)
    if engine is None:
        return _die(reason,
                    "hint: pip install gliner (optional dependency, the engine is "
                    "off by default) or point --model at a local ONNX checkpoint. "
                    "This gate must run on the target CPU box, not a GPU laptop.")

    print(f"model={args.model} texts={len(texts)} labels={len(labels)} "
          f"workers={args.workers}")
    ok = True

    baseline = None
    if not args.no_baseline:
        try:
            baseline = _baseline_pipeline()
        except Exception as exc:
            print(f"note: baseline unavailable ({exc}); latency ratio skipped")

    rss_before = _rss_mb()
    lat = measure_latency(engine, baseline, texts, labels)
    print(f"[latency] gliner p50={lat['gliner_p50']:.1f}ms "
          f"p95={lat['gliner_p95']:.1f}ms | baseline p50={lat['base_p50']:.1f}ms")
    if baseline is None:
        ok = False
        print("[latency] SKIPPED — the ratio vs the MTL pipeline is a mandatory "
              "gate (--no-baseline only skips the model download, it cannot pass)")
    elif lat["gliner_p95"] < MIN_TIMING_MS or lat["base_p95"] < MIN_TIMING_MS:
        ok = False
        print(f"[latency] INCONCLUSIVE — p95 below the {MIN_TIMING_MS}ms timing "
              f"floor, so the ratio is noise; not counted as PASS")
    else:
        ratio = lat["gliner_p95"] / lat["base_p95"]
        gate = ratio <= GATE_LATENCY_RATIO
        ok = ok and gate
        print(f"[latency] ratio={ratio:.2f}x vs MTL gate<={GATE_LATENCY_RATIO}x "
              f"{'PASS' if gate else 'FAIL'}")


    rss_after = _rss_mb()
    if rss_before >= 0 and rss_after >= 0:
        delta = max(0.0, rss_after - rss_before)
        gate = delta <= GATE_RSS_DELTA_MB
        ok = ok and gate
        print(f"[memory ] rss delta={delta:.1f}MB gate<={GATE_RSS_DELTA_MB}MB "
              f"{'PASS' if gate else 'FAIL'}")

    conc_p95 = measure_concurrency(engine, texts, labels, args.workers)
    if conc_p95 < MIN_TIMING_MS or lat["gliner_p95"] < MIN_TIMING_MS:
        ok = False
        print(f"[concurr] p95={conc_p95:.1f}ms INCONCLUSIVE — below the "
              f"{MIN_TIMING_MS}ms timing floor; not counted as PASS")
    else:
        c_ratio = conc_p95 / lat["gliner_p95"]
        gate = c_ratio <= GATE_P95_RATIO
        ok = ok and gate
        print(f"[concurr] p95={conc_p95:.1f}ms ratio={c_ratio:.2f}x "
              f"gate<={GATE_P95_RATIO}x {'PASS' if gate else 'FAIL'}")

    if args.eval:
        samples = [json.loads(ln)
                   for ln in args.eval.read_text(encoding="utf-8").splitlines()
                   if ln.strip()]
        q = measure_quality(engine, samples, ["MATERIAL", "STANDARD", "PARAMETER"])
        gain = q["hybrid_recall"] - q["gliner_recall"]
        gate = gain >= GATE_RECALL_GAIN
        ok = ok and gate
        print(f"[quality] n={q['n_samples']} gliner R={q['gliner_recall']:.3f} "
              f"P={q['gliner_precision']:.3f} | hybrid R={q['hybrid_recall']:.3f} "
              f"P={q['hybrid_precision']:.3f} | dict gain={gain:+.3f} "
              f"gate>=+{GATE_RECALL_GAIN} {'PASS' if gate else 'FAIL'}")
    else:
        print("[quality] SKIPPED — the quality gate is mandatory before shipping "
              "(--eval <jsonl>), see design doc section 5")
        ok = False

    print(f"\nGATE: {'GO' if ok else 'NO-GO'}")
    if not ok:
        print("fallback (design doc section 6): add a custom NER task to the "
              "existing HanLP MTL pipeline instead of a second engine")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

