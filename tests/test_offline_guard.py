"""tests/test_offline_guard.py — Tests for lean_context/offline_guard.py."""

import socket
import pytest

from lean_context.offline_guard import (
    enable, disable, is_active, OfflineViolation,
    _original_create_connection,
)


@pytest.fixture(autouse=True)
def _restore_guard():
    """Always restore the original socket after each test."""
    was_active = is_active()
    yield
    # Restore regardless of test outcome
    socket.create_connection = _original_create_connection
    import lean_context.offline_guard as og
    og._guard_active = False


def test_enable_installs_guard():
    enable()
    assert is_active()


def test_enable_is_idempotent():
    enable()
    enable()  # second call must not raise
    assert is_active()


def test_disable_removes_guard():
    enable()
    disable()
    assert not is_active()
    assert socket.create_connection is _original_create_connection


def test_guard_blocks_non_loopback():
    enable()
    with pytest.raises(OfflineViolation):
        socket.create_connection(("8.8.8.8", 80), timeout=0.1)


def test_guard_allows_loopback_127():
    """create_connection to 127.0.0.1 must not raise OfflineViolation.
    It will raise a connection-refused OSError, but NOT OfflineViolation."""
    enable()
    try:
        socket.create_connection(("127.0.0.1", 1), timeout=0.05)
    except OfflineViolation:
        pytest.fail("Guard should allow 127.0.0.1")
    except OSError:
        pass   # expected — nothing is listening on port 1


def test_guard_allows_localhost():
    enable()
    try:
        socket.create_connection(("localhost", 1), timeout=0.05)
    except OfflineViolation:
        pytest.fail("Guard should allow 'localhost'")
    except OSError:
        pass


def test_guard_blocks_external_domain():
    enable()
    with pytest.raises(OfflineViolation):
        socket.create_connection(("huggingface.co", 443), timeout=0.1)


def test_guard_not_active_by_default():
    """Fresh import — guard must NOT be active until enable() is called."""
    # (autouse fixture already disabled it before this test runs)
    assert not is_active()
