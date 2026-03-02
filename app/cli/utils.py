"""Shared CLI utilities: signal handling, thread pools."""

from collections.abc import Generator
from contextlib import contextmanager
import signal
from concurrent.futures import ThreadPoolExecutor


def setup_signal_handler(
    message: str = "\nInterrupt received, shutting down...",
) -> list[bool]:
    """Register SIGINT/SIGBREAK handler. Returns mutable shutdown flag (check shutdown[0])."""
    shutdown_requested: list[bool] = [False]

    def _on_sigint(*_args: object) -> None:
        shutdown_requested[0] = True
        print(message)

    signal.signal(signal.SIGINT, _on_sigint)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _on_sigint)

    return shutdown_requested


@contextmanager
def managed_thread_pool(max_workers: int) -> Generator[ThreadPoolExecutor, None, None]:
    """Context manager for ThreadPoolExecutor with graceful shutdown on Ctrl+C.

    On exit, calls shutdown(wait=False, cancel_futures=True) so blocked workers
    don't cause hangs or stale DB connections.
    """
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
