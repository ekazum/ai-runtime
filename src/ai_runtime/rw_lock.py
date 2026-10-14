"""``ReadWriteLock`` — many readers or one writer.

Writer-preferring: once a writer waits, new readers queue behind it, so
writes are not starved by a steady stream of reads. The protected work runs
outside the internal mutex. Not reentrant.
"""

import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ReadWriteLock:
    """Shared/exclusive lock: concurrent readers, one writer, writer-preferring."""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._readers_done = threading.Condition(self._mutex)
        self._writer_done = threading.Condition(self._mutex)
        self._active_readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    @contextmanager
    def read_lock(self):
        """Hold the shared side: concurrent with other readers, excludes writers."""
        with self._mutex:
            # Writer preference: queue behind an active OR waiting writer.
            while self._writer_active or self._writers_waiting > 0:
                self._writer_done.wait()
            self._active_readers += 1
        try:
            yield
        finally:
            with self._mutex:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._readers_done.notify_all()

    @contextmanager
    def write_lock(self):
        """Hold the exclusive side: excludes all readers and other writers."""
        with self._mutex:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._active_readers > 0:
                    self._readers_done.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            with self._mutex:
                self._writer_active = False
                # Wake both queues; writer preference is re-applied at
                # acquisition time by the predicates above.
                self._readers_done.notify_all()
                self._writer_done.notify_all()
