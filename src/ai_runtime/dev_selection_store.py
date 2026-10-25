"""``DevSelectionStore`` — an eval store whose dev rows pick the champion."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ai_runtime.sqlite_store import SqliteStore


class DevSelectionStore(SqliteStore):
    """Train, dev and meta rows, plus best-so-far selection over the dev rows.

    Each subclass names its ``FILENAME`` and ``SCHEMA_PATH`` and defines
    ``scored()``, whose third value is the baseline the bar starts at.
    """

    SCHEMA_PATH: Path      # each store names its own schema

    def __init__(self, eval_dir: Path) -> None:
        """Open (creating if absent) the store under the run's eval dir."""
        super().__init__(eval_dir)

    def add_train(self, epoch: int, metrics: dict) -> None:
        """One ``train_metrics`` row."""
        self.append("train_metrics", [{"epoch": int(epoch), **metrics}], key="epoch")

    def add_dev(self, epoch: int, checkpoint: Path, metrics: dict) -> None:
        """One ``dev_metrics`` row: the score task's dict, composite included."""
        row = {
            "epoch": int(epoch),
            "checkpoint": str(checkpoint),
            "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **metrics,
        }
        self.append("dev_metrics", [row], key="epoch")

    def add_meta(self, key: str, value: object) -> None:
        """One ``eval_meta`` pair."""
        self.set_pairs("eval_meta", {key: value})

    def scored(self) -> list[tuple[int, float, float]]:
        """``(epoch, composite, baseline composite)`` of every dev row, by epoch."""
        raise NotImplementedError

    def selection_state(self) -> tuple[Optional[float], int]:
        """``(best composite, epochs without improvement)`` from the dev rows.

        The bar starts at the baseline composite, the same every epoch; a dev
        composite above it becomes the new bar. ``(None, 0)`` before any row.
        """
        rows = self.scored()
        if not rows:
            return None, 0
        best_composite, epochs_without_improvement = rows[0][2], len(rows)
        for index, (_epoch, composite, _baseline) in enumerate(rows):
            if composite > best_composite:
                best_composite = composite
                epochs_without_improvement = len(rows) - 1 - index
        return best_composite, epochs_without_improvement
