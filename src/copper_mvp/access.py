"""Local access keys and browser sessions; credentials stay out of model context."""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass

from copper_mvp.common import WorkbenchError

PROJECT = "copper-research"
PERMISSIONS = {
    "owner": {"read", "compute", "approve", "manage", "mock_control"},
    "researcher": {"read", "compute"},
    "viewer": {"read"},
}


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str
    project_id: str = PROJECT
    key_hash: str = ""

    def require(self, permission):
        if permission not in PERMISSIONS.get(self.role, set()):
            raise WorkbenchError("当前账号没有这项权限", "FORBIDDEN")


def token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AccessControl:
    def __init__(self, store):
        self.store = store
        self.owner_key_path = store.root / "owner_access.key"
        with store.connection() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS access_keys (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, role TEXT NOT NULL,
                    project_id TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS browser_sessions (
                    token_hash TEXT PRIMARY KEY, key_hash TEXT NOT NULL,
                    expires_at REAL NOT NULL);
            """)
        if self.owner_key_path.exists():
            key = self.owner_key_path.read_text(encoding="utf-8").strip()
        else:
            key = secrets.token_urlsafe(32)
            with self.owner_key_path.open("x", encoding="utf-8") as stream:
                stream.write(key + "\n")
        with store.connection() as c:
            c.execute(
                "INSERT OR IGNORE INTO access_keys VALUES(?,?,?,?,0)", (token_hash(key), "owner", "owner", PROJECT)
            )

    def authenticate(self, key=None, cookie=None):
        hashed = token_hash(key or cookie or "")
        with self.store.connection() as c:
            if key:
                row = c.execute("SELECT * FROM access_keys WHERE token_hash=? AND revoked=0", (hashed,)).fetchone()
            else:
                row = c.execute(
                    """SELECT k.* FROM browser_sessions s JOIN access_keys k ON k.token_hash=s.key_hash
                                   WHERE s.token_hash=? AND s.expires_at>? AND k.revoked=0""",
                    (hashed, time.time()),
                ).fetchone()
        return Principal(row["user_id"], row["role"], row["project_id"], row["token_hash"]) if row else None

    def current(self, key_hash, user_id, role):
        with self.store.connection() as c:
            row = c.execute(
                "SELECT * FROM access_keys WHERE token_hash=? AND user_id=? AND role=? AND revoked=0",
                (key_hash, user_id, role),
            ).fetchone()
        if not row:
            raise WorkbenchError("任务发起账号的授权已失效", "FORBIDDEN")
        return Principal(row["user_id"], row["role"], row["project_id"], row["token_hash"])

    def login(self, key):
        principal = self.authenticate(key=key)
        if principal is None:
            raise WorkbenchError("本机访问码无效", "UNAUTHENTICATED")
        cookie = secrets.token_urlsafe(32)
        with self.store.connection() as c:
            c.execute(
                "INSERT INTO browser_sessions VALUES(?,?,?)",
                (token_hash(cookie), token_hash(key), time.time() + 30 * 86400),
            )
        return principal, cookie

    def logout(self, cookie):
        with self.store.connection() as c:
            c.execute("DELETE FROM browser_sessions WHERE token_hash=?", (token_hash(cookie or ""),))

    def issue(self, actor, user_id, role):
        actor.require("manage")
        if role not in PERMISSIONS or not user_id or len(user_id) > 80:
            raise WorkbenchError("账号或角色无效", "ACCESS_INPUT")
        key = secrets.token_urlsafe(32)
        with self.store.connection() as c:
            c.execute("INSERT INTO access_keys VALUES(?,?,?,?,0)", (token_hash(key), user_id, role, PROJECT))
        return key

    def revoke(self, actor, key):
        actor.require("manage")
        with self.store.connection() as c:
            c.execute("UPDATE access_keys SET revoked=1 WHERE token_hash=?", (token_hash(key),))
