"""``ServingRegistry`` — what a worker notifies as it starts/stops serving."""

from typing import Protocol


class ServingRegistry(Protocol):
    """What a worker notifies as it starts/stops consuming the queue."""

    def worker_started(self) -> None: ...

    def worker_stopped(self) -> None: ...
