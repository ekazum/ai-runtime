"""ai-runtime: threads, queues, GPU batching and service supervision for ML pipelines."""

from .batching import BatchModel, BatchRequest, GpuBatchStage
from .counters import LockedCounter
from .rw_lock import ReadWriteLock
from .service_endpoint import ServiceEndpoint
from .serving_registry import ServingRegistry
from .supervisor import ServiceSupervisor
from .thread_handle import ThreadHandle

__version__ = "0.1.0"

__all__ = [
    "BatchModel", "BatchRequest", "GpuBatchStage", "LockedCounter",
    "ReadWriteLock", "ServiceEndpoint", "ServiceSupervisor", "ServingRegistry",
    "ThreadHandle",
]
