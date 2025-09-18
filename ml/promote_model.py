#!/usr/bin/env python3
"""Promote a trained model directory into the serving location."""

import argparse
import json
import shutil
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to freshly trained artifacts")
    parser.add_argument("--target", required=True, help="Path to serving artifacts")
    parser.add_argument("--state", required=True, help="State JSON to update")
    args = parser.parse_args(argv)

    src = Path(args.source)
    dest = Path(args.target)
    state_path = Path(args.state)

    if not src.exists():
        raise FileNotFoundError(f"source directory missing: {src}")
    metrics_path = src / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json missing in {src}")
    metrics = json.loads(metrics_path.read_text())

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
    state.update({"last_promoted": metrics})
    state_path.write_text(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
