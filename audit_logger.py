"""
audit_logger.py — Structured audit trail and governance logging.
Every agent decision, user action, and system event is recorded
with full metadata for explainability and compliance readiness.
"""

import logging
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid
import os

# Console logger — also writes to pints.log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pints.log", mode="a", encoding="utf-8"),
    ],
)

DB_PATH = "audit_trail.db"


class AuditLogger:
    """
    Persistent audit trail backed by SQLite.
    Each event captures: WHO did WHAT, on WHICH claim, with WHAT outcome,
    WHICH model was used, WHAT sources were cited, and WHAT the confidence was.
    This gives you full explainability for every AI decision.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.logger = logging.getLogger("pints.audit")
        self._init_db()

    def _init_db(self):
        """Create the audit table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id            TEXT PRIMARY KEY,
                timestamp     TEXT NOT NULL,
                event_type    TEXT NOT NULL,
                session_id    TEXT NOT NULL,
                user_action   TEXT,
                claim_id      TEXT,
                input_summary TEXT,
                output_summary TEXT,
                model_used    TEXT,
                rag_sources   TEXT,
                decision      TEXT,
                confidence_score REAL,
                flags         TEXT,
                metadata      TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def log_event(
        self,
        event_type: str,
        session_id: str,
        user_action: str = "",
        claim_id: Optional[str] = None,
        input_summary: str = "",
        output_summary: str = "",
        model_used: str = "",
        rag_sources: Optional[List[str]] = None,
        decision: str = "",
        confidence_score: float = 0.0,
        flags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record one audit event. Returns the generated event ID.

        event_type examples:
            SYSTEM_INIT, RAG_UPDATE, CLAIM_ANALYSIS,
            CLAIM_REANALYSIS, EMAIL_GENERATED, EMAIL_SENT, EXPORT
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    event_id,
                    timestamp,
                    event_type,
                    session_id,
                    user_action,
                    claim_id or "",
                    input_summary[:500],  # cap to avoid huge DB rows
                    output_summary[:500],
                    model_used,
                    json.dumps(rag_sources or []),
                    decision,
                    confidence_score,
                    json.dumps(flags or []),
                    json.dumps(metadata or {}, default=str),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to write audit event: {e}")

        # Also emit to log file / console
        self.logger.info(
            f"[{event_type}] action='{user_action}' "
            f"claim={claim_id or 'N/A'} decision={decision or 'N/A'} "
            f"confidence={confidence_score:.2f} session={session_id[:8]}"
        )

        return event_id

    # ------------------------------------------------------------------ #
    #  Query helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_all_logs(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM audit_log ORDER BY timestamp DESC")
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_session_logs(self, session_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT * FROM audit_log WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,),
        )
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_claim_logs(self, claim_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT * FROM audit_log WHERE claim_id=? ORDER BY timestamp ASC",
            (claim_id,),
        )
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_summary_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        stats = {}
        stats["total_events"] = conn.execute(
            "SELECT COUNT(*) FROM audit_log"
        ).fetchone()[0]
        stats["total_claims"] = conn.execute(
            "SELECT COUNT(DISTINCT claim_id) FROM audit_log WHERE claim_id != ''"
        ).fetchone()[0]
        stats["total_sessions"] = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM audit_log"
        ).fetchone()[0]
        stats["emails_sent"] = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE event_type='EMAIL_SENT'"
        ).fetchone()[0]
        stats["out_of_scope"] = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE decision='out_of_scope'"
        ).fetchone()[0]
        conn.close()
        return stats
    