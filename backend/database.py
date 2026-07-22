"""
Neon PostgreSQL database operations for benchmark storage.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from config import DATABASE_URL

logger = logging.getLogger(__name__)


def _get_connection():
    """Create a database connection. Returns None if not configured."""
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set. Benchmarks will not be persisted.")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None


def init_db():
    """Initialize the database schema."""
    conn = _get_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    task VARCHAR(100) NOT NULL,
                    accuracy DOUBLE PRECISION NOT NULL,
                    f1_score DOUBLE PRECISION NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_benchmarks_task
                    ON benchmarks(task);

                CREATE INDEX IF NOT EXISTS idx_benchmarks_model
                    ON benchmarks(model_name);
            """)
        conn.commit()
        logger.info("Database initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False
    finally:
        conn.close()


def save_benchmark(
    model_name: str,
    task: str,
    accuracy: float,
    f1_score: float,
    latency_ms: float,
) -> Optional[int]:
    """
    Save a benchmark run to the database.

    Returns the inserted row id, or None on failure.
    """
    conn = _get_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmarks (model_name, task, accuracy, f1_score, latency_ms)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (model_name, task, accuracy, f1_score, latency_ms),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        logger.info(
            f"Benchmark saved: {model_name}/{task} "
            f"accuracy={accuracy:.4f}, f1={f1_score:.4f}, "
            f"latency={latency_ms:.1f}ms"
        )
        return row_id
    except Exception as e:
        logger.error(f"Failed to save benchmark: {e}")
        return None
    finally:
        conn.close()


def get_benchmarks(task: Optional[str] = None) -> List[dict]:
    """
    Retrieve benchmark entries, optionally filtered by task.

    Returns list of dicts with keys:
        id, model_name, task, accuracy, f1_score, latency_ms, created_at
    """
    conn = _get_connection()
    if not conn:
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if task:
                cur.execute(
                    """
                    SELECT id, model_name, task, accuracy, f1_score, latency_ms, created_at
                    FROM benchmarks
                    WHERE task = %s
                    ORDER BY created_at DESC
                    LIMIT 100
                    """,
                    (task,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, model_name, task, accuracy, f1_score, latency_ms, created_at
                    FROM benchmarks
                    ORDER BY created_at DESC
                    LIMIT 100
                    """
                )
            rows = cur.fetchall()

        # Convert datetime objects to strings
        result = []
        for row in rows:
            entry = dict(row)
            if isinstance(entry.get("created_at"), datetime):
                entry["created_at"] = entry["created_at"].isoformat()
            result.append(entry)

        return result
    except Exception as e:
        logger.error(f"Failed to fetch benchmarks: {e}")
        return []
    finally:
        conn.close()
