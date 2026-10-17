"""ai-runtime: threads, queues, GPU batching and service supervision for ML pipelines."""

from .batching import BatchModel, BatchRequest, GpuBatchStage
from .counters import LockedCounter
from .dev_selection_store import DevSelectionStore
from .pipeline import SHUTDOWN_SENTINEL, Pipeline
from .rw_lock import ReadWriteLock
from .service_endpoint import ServiceEndpoint
from .serving_registry import ServingRegistry
from .sqlite_store import SqliteStore
from .supervisor import ServiceSupervisor
from .thread_handle import ThreadHandle

__version__ = "0.3.1"

__all__ = [
    "BatchModel", "BatchRequest", "DevSelectionStore", "GpuBatchStage",
    "LockedCounter", "Pipeline", "ReadWriteLock", "SHUTDOWN_SENTINEL",
    "ServiceEndpoint", "ServiceSupervisor", "ServingRegistry", "SqliteStore",
    "ThreadHandle",
]
