import sqlite3
import os
import numpy as np
from datetime import datetime, date

DB_PATH = os.path.join("data", "attendance.db")

class Storage:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            day TEXT NOT NULL,
            punch_in TEXT,
            punch_out TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            UNIQUE(user_id, day)
        );
        """)
        self.conn.commit()

    def upsert_user(self, name: str, embedding: np.ndarray):
        emb_blob = embedding.astype(np.float32).tobytes()
        now = datetime.utcnow().isoformat()
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO users(name, embedding, created_at)
        VALUES(?,?,?)
        ON CONFLICT(name) DO UPDATE SET embedding=excluded.embedding
        """, (name, emb_blob, now))
        self.conn.commit()

    def get_users(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, embedding FROM users")
        rows = cur.fetchall()
        users = []
        for uid, name, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            users.append((uid, name, emb))
        return users

    def mark_attendance(self, user_id: int):
        today = date.today().isoformat()
        cur = self.conn.cursor()

        cur.execute("SELECT punch_in, punch_out FROM attendance WHERE user_id=? AND day=?",
                    (user_id, today))
        row = cur.fetchone()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if row is None:
            # Punch-in
            cur.execute("""
            INSERT INTO attendance(user_id, day, punch_in, punch_out)
            VALUES(?,?,?,NULL)
            """, (user_id, today, now))
            self.conn.commit()
            return "PUNCH_IN", now

        punch_in, punch_out = row
        if punch_in and not punch_out:
            # Punch-out
            cur.execute("""
            UPDATE attendance SET punch_out=? WHERE user_id=? AND day=?
            """, (now, user_id, today))
            self.conn.commit()
            return "PUNCH_OUT", now

        return "DONE_FOR_DAY", now