"""``SqliteStore`` — a SQLite file shared between a writer and readers on other hosts.

Each subclass names its ``FILENAME``; the directory is the caller's. Journal mode
is DELETE because WAL is not safe across hosts on network filesystems.
``append`` derives a table from its first row and widens it for later rows;
``set_pairs`` is the key/value form, for totals and counters.
"""

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SqliteStore:
    """A SQLite file with cross-host-safe defaults and dynamic tables."""

    FILENAME: str          # each store names its own file
    _TIMEOUT = 60.0

    def __init__(self, directory: Path, fresh: bool = False) -> None:
        """Bind the store inside ``directory``; ``fresh`` starts it empty."""
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / self.FILENAME
        self._lock = threading.Lock()
        self._columns: dict[str, list[str]] = {}
        if fresh and self._path.exists():
            self._path.unlink()
        logger.info("%s: %s ready.", type(self).__name__, self._path)

    @property
    def path(self) -> Path:
        """Where the store lives."""
        return self._path

    @contextmanager
    def _connect(self) -> "Iterator[sqlite3.Connection]":
        """A connection in the only journal mode safe across hosts here.

        Closed on the way out: a ``sqlite3.Connection`` used as a context
        manager commits the transaction but never closes, so descriptors
        accumulate until a long-running writer exhausts them.
        """
        con = sqlite3.connect(str(self._path), timeout=self._TIMEOUT)
        con.execute("PRAGMA journal_mode=DELETE")
        # Parallel tasks write their own rows; a contended write waits.
        con.execute("PRAGMA busy_timeout=30000")
        try:
            with con:
                yield con
        finally:
            con.close()

    def append(self, table: str, rows: list[dict], key: str) -> None:
        """Insert ``rows`` into ``table``, keyed by ``key``. Empty is a no-op."""
        if not rows:
            return
        with self._lock, self._connect() as db:
            for row in rows:
                self._ensure(db, table, row, key)
                cols = list(row)
                db.execute(
                    f"INSERT OR REPLACE INTO {table} "
                    f"({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [self._value(v) for v in row.values()],
                )

    def set_pairs(self, table: str, values: dict) -> None:
        """Upsert a key/value table — totals, counters, single-row results."""
        with self._lock, self._connect() as db:
            db.execute(f"CREATE TABLE IF NOT EXISTS {table} "
                       f"(key TEXT PRIMARY KEY, value REAL, text TEXT)")
            for key, value in values.items():
                num = (value if isinstance(value, (int, float))
                       and not isinstance(value, bool) else None)
                db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?)",
                           (str(key), num,
                            None if num is not None else str(value)))

    def read(self, table: str, order_by: "str | None" = None) -> list[dict]:
        """One table as dicts; empty when the producer never wrote it."""
        sql = f"SELECT * FROM {table}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        with self._connect() as db:
            try:
                cur = db.execute(sql)
            except sqlite3.OperationalError:
                return []
            names = [d[0] for d in cur.description]
            return [dict(zip(names, row)) for row in cur]

    def read_pairs(self, table: str) -> dict:
        """One key/value table as a dict; empty until it exists."""
        with self._connect() as db:
            try:
                rows = db.execute(
                    f"SELECT key, value, text FROM {table}").fetchall()
            except sqlite3.OperationalError:
                return {}
        return {k: (v if v is not None else t) for k, v, t in rows}

    def _ensure(self, db: sqlite3.Connection, table: str,
                row: dict, key: str) -> None:
        """Create the table from this row, or widen it for unseen keys."""
        known = self._columns.get(table)
        if known is None:
            known = [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
            if not known:
                cols = ", ".join(
                    f"{c} {'INTEGER PRIMARY KEY' if c == key else self._type(row[c])}"
                    for c in row
                )
                db.execute(f"CREATE TABLE {table} ({cols})")
                known = list(row)
            self._columns[table] = known
        for col in row:
            if col not in known:
                db.execute(f"ALTER TABLE {table} ADD COLUMN "
                           f"{col} {self._type(row[col])}")
                known.append(col)

    @staticmethod
    def _type(value: Any) -> str:
        """The column type a first value implies."""
        if isinstance(value, bool) or value is None:
            return "TEXT"
        if isinstance(value, int):
            return "INTEGER"
        return "REAL" if isinstance(value, float) else "TEXT"

    @staticmethod
    def _value(value: Any) -> Any:
        """SQLite takes numbers, strings and bytes; the rest becomes text."""
        if value is None or isinstance(value, (int, float, str, bytes)):
            return value
        return str(value)
