"""
local_storage.py — SQLite-based local storage.
Stores prediction history in data/predictions_history.db.
"""

import os
import json
import uuid
import sqlite3
from datetime import datetime

DB_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "predictions_history.db"
)


def _get_conn():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            input_json TEXT,
            result_json TEXT,
            timestamp TEXT
        )
    ''')
    return conn


def save_prediction(student_input: dict, prediction_result: dict) -> str:
    """Save a prediction record and return its ID."""
    record_id = str(uuid.uuid4())[:8]
    ts = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute('INSERT INTO history (id, input_json, result_json, timestamp) VALUES (?, ?, ?, ?)',
                 (record_id, json.dumps(student_input), json.dumps(prediction_result), ts))
    conn.commit()
    conn.close()
    return record_id


def get_history(limit: int = 20) -> list:
    """Retrieve the most recent prediction records."""
    conn = _get_conn()
    cursor = conn.execute('SELECT id, input_json, result_json, timestamp FROM history ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "input": json.loads(r[1]),
            "result": json.loads(r[2]),
            "timestamp": r[3]
        })
    return history


def clear_history():
    """Clear all prediction history."""
    conn = _get_conn()
    conn.execute('DELETE FROM history')
    conn.commit()
    conn.close()
