# ai-runtime

Threads, queues, GPU batching and service supervision for ML pipelines. The core is standard library only;
`[torch]` adds checkpoints, `[config]` adds YAML configs.

## Parts

- **Batching** (`GpuBatchStage`, `BatchModel`, `BatchRequest`): queued work is collected into a window, sorted by
  weight, cut into batches under a weight budget and run through the project's model; each result goes back to its
  caller's queue as its batch finishes.
- **Supervision** (`ServiceSupervisor`): a service's threads under one shutdown event — ready events, dead-thread
  checks, bounded joins, a retrying `dial` and `drain` for a service connected to a server it may lose.
- **Primitives**: `ThreadHandle` (a thread whose exception re-raises on `join`), `ReadWriteLock` (writer-preferring),
  `LockedCounter`, `ServingRegistry`.
- **Discovery** (`ServiceEndpoint`): a running service publishes host and port in a JSON file on shared storage;
  clients on any node read it.
- **Pipeline** (`Pipeline`, `SHUTDOWN_SENTINEL`): named thread stages under one shutdown event, started in order;
  `stop(join_order)` sends each stage's sentinel and joins in that order, late sentinels after a named stage joins.
- **Stores** (`SqliteStore`, `DevSelectionStore`): a SQLite file safe for one writer and readers on other hosts
  (DELETE journal), tables that widen as new keys arrive; `DevSelectionStore` adds train/dev/meta rows and
  best-so-far selection over the dev rows.
- **Checkpoints** (`ai_runtime.checkpoints`, `[torch]`): atomic saves, snapshots, promote, delete;
  `LatestBestCheckpointManager` keeps `latest.pt` for resume and `best.pt` for the champion.
- **Config** (`ai_runtime.yaml_config`, `[config]`): `YamlConfig`, a frozen pydantic model loaded from one YAML
  mapping, with `_validate`, `_constructor_kwargs` and `log_loaded` hooks.

## Use

A project supplies its model as a `BatchModel` — four methods — and the runtime does the rest:

```python
from ai_runtime import BatchRequest, GpuBatchStage, ServiceSupervisor

class MyModel:
    def load(self): ...                      # weights onto the GPU
    def unload(self): ...
    def weight(self, item) -> int: ...       # e.g. prompt tokens
    def process(self, items, budgets) -> list: ...

supervisor = ServiceSupervisor("my-service")
stage = GpuBatchStage(MyModel(), requests, supervisor.shutdown_event,
                      window_size=4096, idle_timeout_s=0.2, max_weight=500_000)
supervisor.add(stage.run, "batch-stage", daemon=True, ready=stage.ready)
supervisor.start()
supervisor.wait_ready(timeout=1800)
requests.put(BatchRequest(keys=keys, items=items, sink=my_results, budgets=budgets))
```

## Example

`python examples/minimal/app.py` runs two producers against one batching stage over a CPU dummy model.

`AGENTS.md` lists the signatures, the assembly order and the rules to keep.

## License

Apache-2.0
