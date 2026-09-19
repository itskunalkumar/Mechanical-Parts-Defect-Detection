import os
import sqlite3
import sys
from datetime import datetime

from src.exception import CustomException
from src.logger import logging

DB_PATH = os.path.join("artifacts", "predictions.db")

LOW_CONFIDENCE_THRESHOLD = 70.0  # below this, flag for manual review


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    try:
        conn = _get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                needs_review INTEGER NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        raise CustomException(e, sys)


def log_prediction(label: str, confidence: float):
    try:
        init_db()
        needs_review = 1 if confidence < LOW_CONFIDENCE_THRESHOLD else 0
        conn = _get_connection()
        conn.execute(
            "INSERT INTO predictions (timestamp, label, confidence, needs_review) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), label, confidence, needs_review),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # Monitoring must never break the core prediction flow
        logging.info(f"Prediction logging skipped: {e}")


def get_stats() -> dict:
    try:
        init_db()
        conn = _get_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]

        cur.execute("SELECT label, COUNT(*) FROM predictions GROUP BY label")
        by_label = dict(cur.fetchall())

        cur.execute("SELECT AVG(confidence) FROM predictions")
        avg_conf = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM predictions WHERE needs_review = 1")
        needs_review_count = cur.fetchone()[0]

        cur.execute(
            "SELECT timestamp, label, confidence FROM predictions ORDER BY id DESC LIMIT 20"
        )
        recent = cur.fetchall()

        conn.close()

        return {
            "total": total,
            "by_label": by_label,
            "avg_confidence": round(avg_conf, 2),
            "needs_review_count": needs_review_count,
            "recent": recent,
        }
    except Exception as e:
        raise CustomException(e, sys)
