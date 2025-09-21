#!/usr/bin/env python3
"""Lightweight helpers to interact with Postgres for logging predictions and feedback."""

import os
import time  
from typing import Optional

try:
    from sqlalchemy import create_engine, text
except ImportError:  # pragma: no cover - optional dependency
    create_engine = None  # type: ignore
    text = None  # type: ignore

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")
_engine = None


def get_engine():
    global _engine
    if DATABASE_URL is None or create_engine is None:
        return None
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def init_db(retries=5, delay=3) -> None:
    """Initialize database with retry logic."""
    engine = get_engine()
    if engine is None or text is None:
        return
    
    ddl_predictions = """
        CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            probability DOUBLE PRECISION,
            predicted_label TEXT,
            threshold DOUBLE PRECISION
        )
    """
    ddl_feedback = """
        CREATE TABLE IF NOT EXISTS feedback (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            predicted_label TEXT,
            correct_label TEXT,
            probability DOUBLE PRECISION,
            image_path TEXT
        )
    """
    
    for i in range(retries):
        try:
            with engine.begin() as conn:
                conn.execute(text(ddl_predictions))
                conn.execute(text(ddl_feedback))
            print("[DB] Database initialized successfully.")
            return  # Success!
        except Exception as e:
            print(f"[DB] Connection failed (Attempt {i + 1}/{retries}). Retrying in {delay}s...")
            print(f"[DB] Error: {e}")
            time.sleep(delay)
    
    print("[DB] FATAL: Could not initialize database after all retries.")
    raise SystemExit("Failed to connect to database.")


def log_prediction(probability: float, predicted_label: str, threshold: float) -> None:
    engine = get_engine()
    if engine is None or text is None:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions (probability, predicted_label, threshold)"
                " VALUES (:prob, :label, :thr)"
            ),
            {"prob": probability, "label": predicted_label, "thr": threshold},
        )


def store_feedback(predicted_label: str, correct_label: str, probability: float, image_path: str) -> None:
    engine = get_engine()
    if engine is None or text is None:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO feedback (predicted_label, correct_label, probability, image_path)"
                " VALUES (:pred, :corr, :prob, :path)"
            ),
            {
                "pred": predicted_label,
                "corr": correct_label,
                "prob": probability,
                "path": image_path,
            },
        )