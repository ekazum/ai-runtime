"""Checkpoint managers: shared file mechanics, and a run's latest/best pair.

``BaseCheckpointManager`` does atomic torch saves, loads, snapshot listing,
promotion and deletion. ``LatestBestCheckpointManager`` keeps ``latest.pt``
(model + optimizer + step + meta) for resume and ``best.pt`` for the champion.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

class BaseCheckpointManager:
    """Shared file mechanics; nothing here knows what a payload holds."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def atomic_save(self, payload: dict, final_path: Path) -> None:
        """Write to ``<final_path>.tmp``, fsync, then ``os.replace`` onto the name."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        try:
            with open(tmp_path, "rb") as fh:
                os.fsync(fh.fileno())
        except (OSError, AttributeError):
            # fsync may fail on networked filesystems; the rename is atomic anyway.
            pass
        os.replace(tmp_path, final_path)

    @staticmethod
    def load(path: Path, map_location: "str | torch.device" = "cpu") -> dict:
        """A torch-saved dict; payloads hold plain Python objects beyond tensors."""
        return torch.load(path, map_location=map_location, weights_only=False)

    def snapshots(self, pattern: str) -> list[Path]:
        """The files under the dir matching ``pattern``, sorted by name."""
        return sorted(self.checkpoint_dir.glob(pattern))

    def promote(self, path: Path, best_path: Path) -> Path:
        """Rename one snapshot onto ``best_path``; idempotent."""
        source = Path(path)
        if source == best_path:
            return best_path
        source.replace(best_path)
        logger.info("%s: promoted %s -> %s", type(self).__name__, source, best_path)
        return best_path

    def delete(self, path: Path) -> None:
        """Remove one snapshot file."""
        Path(path).unlink(missing_ok=True)
        logger.info("%s: removed %s", type(self).__name__, path)


class LatestBestCheckpointManager(BaseCheckpointManager):
    """``latest.pt`` for resume, ``best.pt`` for the champion."""

    LATEST = "latest.pt"
    BEST = "best.pt"
    # Payload key of the training step: "epoch" or "update".
    STEP_KEY = "epoch"
    # Constructor arguments of the network, stamped into every payload's meta.
    _MODEL_META: tuple[str, ...] = ()

    def __init__(self, checkpoint_dir: Path,
                 run_meta: Optional[dict] = None) -> None:
        """``run_meta``: what the run was trained against, stamped into every payload."""
        super().__init__(checkpoint_dir)
        self._run_meta = dict(run_meta or {})

    def save_latest(self, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, step: int,
                    meta: dict) -> Path:
        """Overwrite ``latest.pt`` atomically with the resume state."""
        path = self.checkpoint_dir / self.LATEST
        self.atomic_save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            self.STEP_KEY: int(step),
            "meta": self._meta(model, meta),
        }, path)
        return path

    def save_best(self, model: torch.nn.Module, step: int,
                  meta: dict) -> Path:
        """Overwrite ``best.pt`` atomically with the champion's weights."""
        path = self.checkpoint_dir / self.BEST
        self.atomic_save({
            "model_state_dict": model.state_dict(),
            self.STEP_KEY: int(step),
            "meta": self._meta(model, meta),
        }, path)
        logger.info("%s: best.pt <- %s %d", type(self).__name__,
                    self.STEP_KEY, step)
        return path

    def find_latest(self) -> Optional[Path]:
        """``latest.pt`` when it exists — the resume point."""
        path = self.checkpoint_dir / self.LATEST
        return path if path.exists() else None

    def best(self) -> Optional[Path]:
        """``best.pt`` when it exists."""
        path = self.checkpoint_dir / self.BEST
        return path if path.exists() else None

    def _meta(self, model: torch.nn.Module, meta: dict) -> dict:
        """The run's meta, the network's constructor arguments, then the caller's."""
        return {**self._run_meta,
                **{key: getattr(model, key) for key in self._MODEL_META},
                **meta}
