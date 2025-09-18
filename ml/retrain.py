#!/usr/bin/env python3
"""Simple orchestration script to trigger training when enough feedback arrives."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

FEEDBACK_CLASSES = ("no", "yes")


def count_feedback(weak_dir: Path) -> int:
    total = 0
    if not weak_dir.exists():
        return 0
    for cls in FEEDBACK_CLASSES:
        for path in (weak_dir / cls).glob("*"):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                total += 1
    return total


def load_state(path: Path) -> dict:
    if path.exists():
        with path.open() as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                pass
    return {"feedback_seen": 0, "best_run": None, "best_metrics": None}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(state, fh, indent=2)


def launch_training(train_script: Path, out_dir: Path, base_args: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(train_script), "--out_dir", str(out_dir)] + base_args
    print("[retrain] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
    with metrics_path.open() as fh:
        return json.load(fh)


def metric_ok(new_metrics: dict, old_metrics: Optional[dict], metric: str, drop_tol: float, min_abs: float) -> bool:
    new_val = new_metrics.get("test", {}).get(metric)
    if new_val is None:
        raise ValueError(f"new metrics missing test->{metric}")
    if new_val < min_abs:
        print(f"[retrain] reject: {metric} {new_val:.3f} < min {min_abs:.3f}")
        return False
    if not old_metrics:
        return True
    old_val = old_metrics.get("test", {}).get(metric)
    if old_val is None:
        return True
    if new_val + drop_tol < old_val:
        print(f"[retrain] reject: {metric} dropped from {old_val:.3f} to {new_val:.3f}")
        return False
    return True


def promote_run(run_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(run_dir, dest_dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_script", default="ml/train.py")
    parser.add_argument("--weak_dir", default="data/weak_feedback")
    parser.add_argument("--state_file", default="ml/artifacts/state.json")
    parser.add_argument("--runs_dir", default="ml/artifacts/runs")
    parser.add_argument("--current_dir", default="ml/artifacts/current")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--metric", default="auc")
    parser.add_argument("--metric_drop_tol", type=float, default=0.02)
    parser.add_argument("--metric_min", type=float, default=0.7)
    parser.add_argument("--train_arg", action="append", default=[])
    args = parser.parse_args(argv)

    weak_dir = Path(args.weak_dir)
    state_path = Path(args.state_file)
    runs_dir = Path(args.runs_dir)
    current_dir = Path(args.current_dir)
    train_script = Path(args.train_script)

    state = load_state(state_path)
    feedback_count = count_feedback(weak_dir)
    print(f"[retrain] feedback_seen={state['feedback_seen']} current={feedback_count}")
    new_samples = max(0, feedback_count - int(state.get("feedback_seen", 0)))
    if new_samples < args.threshold:
        print(f"[retrain] not enough new samples (need {args.threshold}, have {new_samples})")
        return 0

    ts = int(time.time())
    run_dir = runs_dir / f"run_{ts}"
    base_args = []
    for item in args.train_arg:
        base_args.extend(item.split())
    if "--weak_dir" not in base_args:
        base_args.extend(["--weak_dir", str(weak_dir)])
    launch_training(train_script, run_dir, base_args)

    metrics = load_metrics(run_dir / "metrics.json")
    best_metrics = state.get("best_metrics")
    if not metric_ok(metrics, best_metrics, args.metric, args.metric_drop_tol, args.metric_min):
        print("[retrain] training rejected; keeping current model")
        state["feedback_seen"] = feedback_count
        save_state(state_path, state)
        return 1

    promote_run(run_dir, current_dir)
    state.update({
        "feedback_seen": feedback_count,
        "best_run": str(run_dir),
        "best_metrics": metrics,
    })
    save_state(state_path, state)
    print(f"[retrain] promoted {run_dir} -> {current_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
