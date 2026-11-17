"""Very small user-account layer: SQLite + salted password hashes (werkzeug)."""
import re
import sqlite3
import time

from werkzeug.security import check_password_hash, generate_password_hash

from . import config

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
MIN_PASSWORD = 6
MAX_FAILS, LOCK_SECONDS = 5, 60
_fails = {}   # (username_lower) -> (count, first_fail_time)


def _conn():
    config.INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")


def create_user(username: str, password: str):
    """Returns (ok, error_message)."""
    if not USERNAME_RE.match(username or ""):
        return False, "Username must be 3-30 letters, numbers, dot, dash or underscore."
    if len(password or "") < MIN_PASSWORD:
        return False, f"Password must be at least {MIN_PASSWORD} characters."
    try:
        with _conn() as c:
            c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                      (username, generate_password_hash(password)))
    except sqlite3.IntegrityError:
        return False, "This username is already taken."
    return True, ""


def locked(username: str) -> bool:
    count, first = _fails.get(username.lower(), (0, 0))
    if count >= MAX_FAILS and time.time() - first < LOCK_SECONDS:
        return True
    if count >= MAX_FAILS:
        _fails.pop(username.lower(), None)
    return False


def verify_user(username: str, password: str):
    """Returns the user row or None."""
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?", (username or "",)).fetchone()
    ok = bool(row) and check_password_hash(row["password_hash"], password or "")
    key = (username or "").lower()
    if ok:
        _fails.pop(key, None)
        return row
    count, first = _fails.get(key, (0, time.time()))
    _fails[key] = (count + 1, first)
    return None
