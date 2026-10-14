"""``GpuBatchStage`` — length-sorted batching of queued work under a weight budget.

Items accumulate into a sort window, flushed when the window is full or the
producer has been idle for ``idle_timeout_s``; the window is then sorted by
weight and cut greedily into batches under ``max_weight``, so similar-weight
items share a batch. Each batch's results go out as it finishes.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Items awaiting a result, and the queue to deliver them on.

    The stage puts ``(key, result)`` on ``sink`` per item as its batch finishes.
    """

    keys: list
    items: list
    sink: queue.Queue
    # Per-item budget handed to the model with the item (e.g. a reply length).
    budgets: list


class BatchModel(Protocol):
    """What a project supplies: the model the stage loads and runs."""

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def weight(self, item: Any) -> int: ...

    def process(self, items: list, budgets: list) -> list: ...


class GpuBatchStage:
    """Sorts pending work by weight and runs it through a ``BatchModel``."""

    def __init__(
        self,
        model: BatchModel,
        request_queue: queue.Queue,
        shutdown_event: threading.Event,
        window_size: int,
        idle_timeout_s: float,
        max_weight: int,
        name: str = "GpuBatchStage",
    ) -> None:
        self._model = model
        self._request_queue = request_queue
        self._shutdown_event = shutdown_event
        self._window_size = window_size
        self._idle_timeout_s = idle_timeout_s
        self._max_weight = max_weight
        self._name = name
        # Set once the model is loaded; loading can take minutes.
        self.ready = threading.Event()

    def run(self) -> None:
        """Load the model, then window, batch and process until shutdown."""
        self._model.load()
        self.ready.set()
        window: list = []
        try:
            while not self._shutdown_event.is_set():
                try:
                    request = self._request_queue.get(timeout=self._idle_timeout_s)
                    window += self._entries(request)
                    if len(window) < self._window_size:
                        continue
                except queue.Empty:
                    if not window:
                        continue
                self._process_window(window)
                window = []
        finally:
            self._model.unload()
        logger.info("%s: done.", self._name)

    @staticmethod
    def _entries(request: BatchRequest) -> list:
        """One ``(item, key, sink, budget)`` entry per item of the request."""
        return [
            (item, key, request.sink, budget)
            for key, item, budget in zip(request.keys, request.items,
                                         request.budgets)
        ]

    def form_batches(self, window: list) -> list[list]:
        """Sort the window by weight and cut it into batches under ``max_weight``."""
        window.sort(key=lambda entry: self._model.weight(entry[0]))
        batches: list[list] = []
        batch: list = []
        weight = 0
        for entry in window:
            item_weight = self._model.weight(entry[0])
            if batch and weight + item_weight > self._max_weight:
                batches.append(batch)
                batch, weight = [], 0
            batch.append(entry)
            weight += item_weight
        if batch:
            batches.append(batch)
        return batches

    def _process_window(self, window: list) -> None:
        """Run every batch, delivering each item's result as it finishes."""
        batches = self.form_batches(window)
        logger.info("%s: %d items in %d batches.",
                    self._name, len(window), len(batches))
        for batch in batches:
            results = self._model.process([entry[0] for entry in batch],
                                          [entry[3] for entry in batch])
            for (_, key, sink, _), result in zip(batch, results):
                sink.put((key, result))
