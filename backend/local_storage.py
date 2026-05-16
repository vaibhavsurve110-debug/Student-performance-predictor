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
    conn.execute('''
        CREATE TABLE IF NOT EXISTS shared_reports (
            id TEXT PRIMARY KEY,
            prediction_id TEXT,
            owner_role TEXT,
            created_at TEXT,
            FOREIGN KEY(prediction_id) REFERENCES history(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS share_links (
            id TEXT PRIMARY KEY,
            shared_report_id TEXT,
            token TEXT UNIQUE,
            expires_at TEXT,
            allow_download INTEGER,
            view_count INTEGER DEFAULT 0,
            FOREIGN KEY(shared_report_id) REFERENCES shared_reports(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS email_logs (
            id TEXT PRIMARY KEY,
            shared_report_id TEXT,
            recipient TEXT,
            subject TEXT,
            status TEXT,
            sent_at TEXT,
            FOREIGN KEY(shared_report_id) REFERENCES shared_reports(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS share_activity_logs (
            id TEXT PRIMARY KEY,
            shared_report_id TEXT,
            action TEXT,
            ip_address TEXT,
            timestamp TEXT,
            FOREIGN KEY(shared_report_id) REFERENCES shared_reports(id)
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


def create_shared_report(prediction_id: str, owner_role: str) -> str:
    """Create a shared report entry."""
    shared_id = str(uuid.uuid4())
    ts = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute('INSERT INTO shared_reports (id, prediction_id, owner_role, created_at) VALUES (?, ?, ?, ?)',
                 (shared_id, prediction_id, owner_role, ts))
    conn.commit()
    conn.close()
    return shared_id


def generate_share_link(shared_report_id: str, expires_at: Optional[str], allow_download: bool) -> str:
    """Generate a tokenized public link."""
    link_id = str(uuid.uuid4())
    token = str(uuid.uuid4().hex)
    conn = _get_conn()
    conn.execute('INSERT INTO share_links (id, shared_report_id, token, expires_at, allow_download) VALUES (?, ?, ?, ?, ?)',
                 (link_id, shared_report_id, token, expires_at, 1 if allow_download else 0))
    conn.commit()
    conn.close()
    return token


def log_email(shared_report_id: str, recipient: str, subject: str, status: str):
    """Log an email sharing event."""
    log_id = str(uuid.uuid4())
    ts = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute('INSERT INTO email_logs (id, shared_report_id, recipient, subject, status, sent_at) VALUES (?, ?, ?, ?, ?, ?)',
                 (log_id, shared_report_id, recipient, subject, status, ts))
    conn.commit()
    conn.close()


def log_activity(shared_report_id: str, action: str, ip_address: str = "unknown"):
    """Log a sharing activity (view/download)."""
    log_id = str(uuid.uuid4())
    ts = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute('INSERT INTO share_activity_logs (id, shared_report_id, action, ip_address, timestamp) VALUES (?, ?, ?, ?, ?)',
                 (log_id, shared_report_id, action, ip_address, ts))
    # Update view count if action is VIEW
    if action == "VIEW":
        conn.execute('UPDATE share_links SET view_count = view_count + 1 WHERE shared_report_id = ?', (shared_report_id,))
    conn.commit()
    conn.close()


def get_shared_report_by_token(token: str) -> Optional[dict]:
    """Retrieve shared report data by token if not expired."""
    conn = _get_conn()
    cursor = conn.execute('''
        SELECT h.input_json, h.result_json, sl.expires_at, sl.allow_download, sl.shared_report_id
        FROM share_links sl
        JOIN shared_reports sr ON sl.shared_report_id = sr.id
        JOIN history h ON sr.prediction_id = h.id
        WHERE sl.token = ?
    ''', (token,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        expires_at = row[2]
        if expires_at and datetime.fromisoformat(expires_at) < datetime.now():
            return None # Expired
            
        return {
            "input": json.loads(row[0]),
            "result": json.loads(row[1]),
            "allow_download": bool(row[3]),
            "shared_report_id": row[4]
        }
    return None


def get_share_history(limit: int = 50, role: str = "Administrator") -> list:
    """Retrieve recent sharing activity history."""
    conn = _get_conn()
    query = '''
        SELECT sr.id, sr.prediction_id, sr.owner_role, sr.created_at, h.result_json
        FROM shared_reports sr
        JOIN history h ON sr.prediction_id = h.id
        ORDER BY sr.created_at DESC LIMIT ?
    '''
    cursor = conn.execute(query, (limit,))
    rows = cursor.fetchall()
    
    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "prediction_id": r[1],
            "owner": r[2],
            "timestamp": r[3],
            "label": json.loads(r[4]).get("predicted_label", "Unknown")
        })
    conn.close()
    return history
