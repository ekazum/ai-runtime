"""Minimal ai-runtime app: two producers share one batching stage over a CPU dummy model.

Run from anywhere: ``python examples/minimal/app.py``. The model upper-cases
words and sleeps in proportion to a batch's weight, standing in for a GPU.
"""

import logging
import queue
import random
import threading
import time

from ai_runtime import BatchRequest, GpuBatchStage, ServiceSupervisor

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


class EchoModel:
    """A ``BatchModel``: weight is the word length, the result its upper case."""

    @staticmethod
    def load() -> None:
        time.sleep(0.2)            # a real model loads its weights here

    def unload(self) -> None:
        pass

    @staticmethod
    def weight(item: str) -> int:
        return len(item)

    @staticmethod
    def process(items: list, budgets: list) -> list:
        time.sleep(0.001 * sum(len(item) for item in items))
        return [item.upper()[:budget] for item, budget in zip(items, budgets)]


class Producer:
    """Sends a few requests to the stage and waits for every answer."""

    def __init__(self, name: str, requests: queue.Queue, n_requests: int) -> None:
        self.name = name
        self._requests = requests
        self._n_requests = n_requests
        self._sink: queue.Queue = queue.Queue()
        self.answers: dict = {}

    def run(self) -> None:
        rng = random.Random(self.name)
        expected = 0
        for r in range(self._n_requests):
            words = rng.sample(WORDS, 3)
            keys = [f"{self.name}-{r}-{i}" for i in range(len(words))]
            self._requests.put(BatchRequest(keys=keys, items=words, sink=self._sink,
                                            budgets=[8] * len(words)))
            expected += len(words)
        while len(self.answers) < expected:
            key, answer = self._sink.get()
            self.answers[key] = answer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    requests: queue.Queue = queue.Queue()
    supervisor = ServiceSupervisor("example")
    stage = GpuBatchStage(EchoModel(), requests, supervisor.shutdown_event,
                          window_size=16, idle_timeout_s=0.05, max_weight=40)
    producers = [Producer(name, requests, n_requests=10) for name in ("p0", "p1")]
    supervisor.add(stage.run, "batch-stage", daemon=True, ready=stage.ready)
    supervisor.start()
    if not supervisor.wait_ready(5):
        raise SystemExit("model never became ready")
    threads = [threading.Thread(target=p.run, name=p.name) for p in producers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    supervisor.stop(join_timeout=2)
    for producer in producers:
        sample = dict(list(producer.answers.items())[:3])
        print(f"{producer.name}: {len(producer.answers)} answers, e.g. {sample}")
    print("dead threads after stop:", supervisor.dead_threads())


if __name__ == "__main__":
    main()
