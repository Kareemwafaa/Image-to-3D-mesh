"""API keys and the request principal.

Key format: `i3d_<prefix>_<secret>`. The prefix (8 hex chars) is stored in
clear for lookup and display; only SHA-256(key) is stored. A fast hash is fine
here because keys carry ~190 bits of randomness, so there is nothing to
brute-force (unlike passwords, which need bcrypt/argon2).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from ..core.config import Settings
from ..core.errors import AppError

KEY_RE = re.compile(r"^i3d_([0-9a-f]{8})_([A-Za-z0-9_-]{20,})$")


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


def generate_key() -> tuple[str, str, str]:
    """Returns (plaintext, prefix, hash). The plaintext is shown once, never stored."""
    prefix = secrets.token_hex(4)
    plaintext = f"i3d_{prefix}_{secrets.token_urlsafe(24)}"
    return plaintext, prefix, hash_key(plaintext)


@dataclass(frozen=True)
class Principal:
    owner: str  # "key:<uuid>", "key:admin" or "ip:<address>"
    kind: str  # "key" | "anon"
    is_admin: bool = False
    api_key_id: Optional[uuid.UUID] = None
    submit_per_minute: Optional[float] = None
    max_inflight: Optional[int] = None

    def submit_limits(self, s: Settings) -> tuple[float, int]:
        if self.kind == "anon":
            return s.anon_submit_per_minute, s.anon_submit_burst
        return (self.submit_per_minute or s.key_submit_per_minute), s.key_submit_burst

    def inflight_cap(self, s: Settings) -> Optional[int]:
        if self.is_admin:
            return None
        if self.kind == "anon":
            return s.anon_max_inflight
        return self.max_inflight or s.key_max_inflight


def client_ip(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _presented_key(request: Request) -> Optional[str]:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


async def resolve_principal(request: Request) -> Principal:
    c = request.app.state.container
    s: Settings = c.settings
    presented = _presented_key(request)
    if presented is None:
        if s.auth_mode == "required":
            raise AppError(
                401, "unauthenticated", "An API key is required (X-API-Key header).", {"WWW-Authenticate": "Bearer"}
            )
        return Principal(owner=f"ip:{client_ip(request, s.trust_proxy_headers)}", kind="anon")

    if s.admin_api_key and hmac.compare_digest(presented, s.admin_api_key):
        return Principal(owner="key:admin", kind="key", is_admin=True)

    m = KEY_RE.match(presented)
    if not m:
        raise AppError(401, "invalid_api_key", "Malformed API key.")
    record = await c.api_keys.by_prefix(m.group(1))
    if record is None or record.revoked_at is not None or not hmac.compare_digest(record.key_hash, hash_key(presented)):
        raise AppError(401, "invalid_api_key", "Unknown or revoked API key.")
    return Principal(
        owner=f"key:{record.id}",
        kind="key",
        is_admin=record.is_admin,
        api_key_id=record.id,
        submit_per_minute=record.submit_per_minute,
        max_inflight=record.max_inflight,
    )
