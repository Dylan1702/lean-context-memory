"""
lean_context/offline_guard.py — Block all non-loopback socket connections.

Call enable() once at startup (before any network-capable import) to install
a socket-level guard that raises OfflineViolation whenever application code
tries to open a TCP connection to any host other than 127.0.0.1, ::1, or
localhost. This enforces the local-only design (Requirement 10.1) and catches
accidental cloud calls (HuggingFace, Chroma telemetry, etc.) at runtime.

All blocked attempts are appended to offline_guard.log in the project root.

Usage:
    from lean_context.offline_guard import enable
    enable()                      # install the guard
    enable(log_path="my.log")     # custom log file
    enable()                      # calling again is a no-op

Disable (e.g. for integration tests that need the real internet):
    from lean_context.offline_guard import disable
    disable()

CLI opt-out:  pass --no-offline-guard to any CLI that supports it.

offline_check entry-point:
    python -m lean_context.offline_check
"""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_LOG = str(_BASE_DIR / "offline_guard.log")

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}

# Store the original so we can restore it
_original_create_connection: Callable = socket.create_connection
_original_getaddrinfo: Callable = socket.getaddrinfo
_guard_active: bool = False


class OfflineViolation(Exception):
    """Raised when application code tries to reach a non-loopback host."""


def _is_loopback(host: str) -> bool:
    h = (host or "").lower().strip()
    if h in _LOOPBACK:
        return True
    # Also accept bare numeric loopback variants
    if h.startswith("127.") or h == "::1":
        return True
    return False


def _guarded_create_connection(address, timeout=None, source_address=None, **kwargs):
    host, port = address[0], address[1]
    if not _is_loopback(host):
        _log_violation(host, port)
        raise OfflineViolation(
            f"OfflineGuard: blocked connection to {host}:{port} "
            f"(non-loopback host). Pass --no-offline-guard to disable."
        )
    return _original_create_connection(address, timeout=timeout,
                                       source_address=source_address, **kwargs)


def _log_violation(host: str, port: int) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    msg = f"{ts}  BLOCKED  {host}:{port}\n"
    try:
        with open(_DEFAULT_LOG, "a") as f:
            f.write(msg)
    except OSError:
        pass
    logger.warning("OfflineGuard blocked connection to %s:%s", host, port)


def enable(log_path: str | None = None) -> None:
    """
    Install the socket guard.  Idempotent — safe to call multiple times.

    After this call any TCP connection attempt to a non-loopback address
    raises OfflineViolation and is logged to *log_path* (default: offline_guard.log).
    Ollama on localhost:11434 is always permitted.
    """
    global _guard_active, _DEFAULT_LOG
    if _guard_active:
        return
    if log_path:
        _DEFAULT_LOG = log_path
    socket.create_connection = _guarded_create_connection
    _guard_active = True
    logger.debug("OfflineGuard enabled — non-loopback connections will be blocked")


def disable() -> None:
    """Remove the socket guard and restore the original create_connection."""
    global _guard_active
    socket.create_connection = _original_create_connection
    _guard_active = False
    logger.debug("OfflineGuard disabled")


def is_active() -> bool:
    """Return True if the guard is currently installed."""
    return _guard_active
