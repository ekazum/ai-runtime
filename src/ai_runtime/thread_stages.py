"""``ThreadStages`` — thread stages started in one order and torn down in another.

Each stage is a group of threads, optionally with the queue they consume.
``stop`` sets the shutdown event, wakes every consumer with one sentinel
per thread, and joins the stages in the order given. A stage whose producer
must finish first gets its sentinels only after that producer is joined.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .thread_handle import ThreadHandle

logger = logging.getLogger(__name__)

# Put on a stage's inbox at shutdown; consumers compare by identity.
SHUTDOWN_SENTINEL: Any = object()


@dataclass
class _Stage:
    """One stage: its threads, its inbox and when the inbox is woken."""

    handles: list[ThreadHandle]
    inbox: "queue.Queue | None"
    sentinel_after: "str | None"


class ThreadStages:
    """Starts stages in the order added; stops them in a given join order."""

    def __init__(self, name: str, shutdown_event: threading.Event | None = None,
                 join_timeout: float = 30.0) -> None:
        self._name = name
        self.shutdown_event = shutdown_event or threading.Event()
        self._join_timeout = join_timeout
        self._stages: dict[str, _Stage] = {}

    def stage(self, name: str,
              targets: "Callable[[], None] | list[Callable[[], None]]",
              inbox: "queue.Queue | None" = None,
              sentinel_after: "str | None" = None) -> None:
        """Add one stage: one thread ``name``, or for a list ``name-0``, ``name-1``, ...

        ``inbox`` gets one sentinel per thread at ``stop``, or right after
        stage ``sentinel_after`` has been joined.
        """
        handles = ([ThreadHandle(t, f"{name}-{i}") for i, t in enumerate(targets)]
                   if isinstance(targets, list) else [ThreadHandle(targets, name)])
        self._stages[name] = _Stage(handles, inbox, sentinel_after)

    def __len__(self) -> int:
        """The number of threads across all stages."""
        return sum(len(s.handles) for s in self._stages.values())

    def start(self) -> None:
        """Start every stage's threads, in the order the stages were added."""
        for stage in self._stages.values():
            for handle in stage.handles:
                handle.start()

    def wait(self) -> None:
        """Block until some participant sets the shutdown event."""
        self.shutdown_event.wait()

    def stop(self, join_order: list[str]) -> None:
        """Set the event, wake the consumers, join the stages in ``join_order``.

        A captured thread exception is re-raised by its join.
        """
        self.shutdown_event.set()
        for name in join_order:
            stage = self._stages[name]
            if stage.inbox is not None and stage.sentinel_after is None:
                self._push_sentinels(stage.inbox, len(stage.handles))
        for name in join_order:
            for handle in self._stages[name].handles:
                handle.join(timeout=self._join_timeout)
                if handle.is_alive():
                    logger.warning("%s: thread %r did not exit within %.0fs",
                                   self._name, handle.name, self._join_timeout)
            for later in self._stages.values():
                if later.inbox is not None and later.sentinel_after == name:
                    self._push_sentinels(later.inbox, len(later.handles))

    @staticmethod
    def _push_sentinels(q: queue.Queue, n: int) -> None:
        """Best-effort: one sentinel per consumer; Full is fine (they poll too)."""
        for _ in range(n):
            try:
                q.put_nowait(SHUTDOWN_SENTINEL)
            except queue.Full:
                return
