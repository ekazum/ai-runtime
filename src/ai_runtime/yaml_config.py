"""``YamlConfig`` — shared YAML-loading base for small typed config models.

Every config model inherits ``from_yaml`` from here (directly, or via
``BaseConfig``) so the shared file/parse/mapping handling lives in one place.
A subclass supplies its fields, a static ``_validate(raw)`` for type/range
checks, an optional ``_constructor_kwargs(raw)`` that adds computed fields,
and an optional ``log_loaded`` load summary.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="YamlConfig")


class YamlConfig(BaseModel):
    """Base for the typed config models loaded from a single YAML mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        """Load, validate, and return an instance from a YAML file.

        Raises ``FileNotFoundError`` if ``path`` is missing, ``ValueError`` on any
        type/range failure. An empty file yields the all-default config.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{cls.__name__} YAML not found: {path}")
        raw = yaml.safe_load(path.read_bytes())
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"{cls.__name__} YAML must parse to a mapping; got {type(raw).__name__}"
            )
        cls._validate(raw)
        cfg = cls(**cls._constructor_kwargs(raw))
        cfg.log_loaded()
        return cfg

    @staticmethod
    def _validate(raw: dict[str, Any]) -> None:
        """Type + range checks; overridden by subclasses (default no-op)."""

    @classmethod
    def _constructor_kwargs(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Constructor kwargs from the validated ``raw``; subclasses add computed ones."""
        return raw

    def log_loaded(self) -> None:
        """Emit a one-line load summary; overridden by subclasses (default no-op)."""
