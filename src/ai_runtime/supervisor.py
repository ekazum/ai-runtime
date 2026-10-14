"""``ServiceSupervisor`` — a service's thread groups under one shutdown event.

Threads are added with an optional ready event, started together, checked for
unasked-for exits, and joined with a bound on shutdown. ``dial`` and ``drain``
cover a service that keeps a connection to a server it may lose.
"""

import queue
import threading
import time
from multiprocessing.connection import Client, Connection

from .thread_handle import ThreadHandle


class ServiceSupervisor:
    """Starts, watches and stops a service's threads."""

    def __init__(self, name: str = "ServiceSupervisor",
                 shutdown_event: threading.Event | None = None) -> None:
        self._name = name
        self.shutdown_event = shutdown_event or threading.Event()
        self._handles: list[ThreadHandle] = []
        self._ready: list[threading.Event] = []

    def add(self, target, name: str, daemon: bool = False,
            ready: threading.Event | None = None) -> None:
        """Register one thread; ``ready`` is set by it once it can serve."""
        self._handles.append(ThreadHandle(target, name, daemon))
        if ready is not None:
            self._ready.append(ready)

    def start(self) -> None:
        """Start every registered thread."""
        for handle in self._handles:
            handle.start()

    def __len__(self) -> int:
        """The number of registered threads."""
        return len(self._handles)

    def wait_ready(self, timeout: float) -> bool:
        """Block until every registered ready event is set; False on timeout or if none.

        Safe to call while another thread is still in ``start``: the events are
        registered by ``add``, before any thread runs.
        """
        if not self._ready:
            return False
        deadline = time.monotonic() + timeout
        return all(event.wait(max(0.0, deadline - time.monotonic()))
                   for event in self._ready)

    def stop(self, join_timeout: float) -> None:
        """Signal shutdown and join each thread for at most ``join_timeout``.

        A thread wedged past the bound is abandoned; give it ``daemon=True``
        so it cannot hold the process open.
        """
        self.shutdown_event.set()
        for handle in self._handles:
            handle.join(timeout=join_timeout)

    def dead_threads(self) -> list[str]:
        """Names of threads that have exited."""
        return [h.name for h in self._handles if not h.is_alive()]

    def dial(self, address: tuple[str, int], timeout: float,
             retry_interval: float = 5.0) -> Connection:
        """Connect to ``address``, retrying while the server finishes starting up."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return Client(address)
            except (ConnectionRefusedError, OSError) as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{self._name}: {address[0]}:{address[1]} unreachable "
                        f"after {timeout:.0f}s — {error}"
                    ) from error
                time.sleep(retry_interval)

    @staticmethod
    def drain(*queues: queue.Queue) -> None:
        """Discard everything queued, e.g. work whose asker is gone."""
        for q in queues:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
