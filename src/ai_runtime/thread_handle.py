"""``ThreadHandle`` — ``threading.Thread`` wrapper that surfaces exceptions.

A bare ``threading.Thread`` swallows uncaught exceptions, so a crashed
worker silently stops producing while its owner keeps waiting.
``ThreadHandle`` captures the exception and re-raises it on ``join()``.
"""

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class ThreadHandle:
    """Wrap a target callable in a ``threading.Thread`` with exception capture."""

    def __init__(self, target: Callable[[], None], name: str,
                 daemon: bool = False) -> None:
        """Construct a non-started thread bound to ``target``.

        ``daemon`` is for targets that never return — a server accept loop
        blocks forever, so the process could not exit while it is joined.
        """
        self._target = target
        self._name = name
        self._exception: BaseException | None = None
        self._thread = threading.Thread(target=self._wrapped, name=name,
                                        daemon=daemon)

    def _wrapped(self) -> None:
        """Invoke ``target`` and capture any exception for later re-raise on join.

        Catches ``BaseException`` (not just ``Exception``) so that
        ``SystemExit`` and ``KeyboardInterrupt`` raised inside the
        worker also surface to ``join`` rather than silently killing
        the thread.
        """
        try:
            self._target()
        except BaseException as exc:  # noqa: BLE001
            self._exception = exc
            logger.exception(
                "Thread %r raised %s; captured for join()",
                self._name,
                type(exc).__name__,
            )

    def start(self) -> None:
        """Start the underlying thread."""
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Join the thread, re-raising any captured exception from ``target``.

        If ``timeout`` elapses without the thread finishing, return
        without raising — the caller can poll ``is_alive()`` and decide
        whether to retry or escalate.
        """
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            return  # timed out; nothing to re-raise yet
        if self._exception is not None:
            raise self._exception

    def is_alive(self) -> bool:
        """Return whether the wrapped thread is still running."""
        return self._thread.is_alive()

    @property
    def name(self) -> str:
        """The thread name given at construction."""
        return self._name

    @staticmethod
    def interruptible_sleep(event: threading.Event, seconds: float) -> None:
        """Sleep up to ``seconds``, waking immediately when ``event`` is set.

        Thin wrapper around ``event.wait(timeout)``. Used by worker
        loops that need to back off briefly between polls but still
        respond promptly to shutdown signals.
        """
        event.wait(timeout=seconds)
