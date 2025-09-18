#!/usr/bin/env python3
"""Utility script to enforce evaluation thresholds after retraining."""

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--min_auc", type=float, default=0.7)
    parser.add_argument("--drop_tol", type=float, default=0.02)
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found at {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    test_auc = metrics.get("test", {}).get("auc")
    if test_auc is None:
        raise ValueError("metrics missing test.auc")
    if test_auc < args.min_auc:
        raise SystemExit(f"test AUC {test_auc:.3f} below minimum {args.min_auc:.3f}")

    state_path = Path(args.state)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        prev_auc = state.get("last_promoted", {}).get("test", {}).get("auc")
        if prev_auc is not None and test_auc + args.drop_tol < prev_auc:
            raise SystemExit(
                f"test AUC dropped more than tolerance: {test_auc:.3f} (prev {prev_auc:.3f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
