# ai-runtime — notes for AI coding tools

ai-runtime gives the mechanics of a threaded ML service or training pipeline: batching work onto a model, running
threads under one shutdown event, stopping them in order, SQLite stores, checkpoint files and YAML configs. Nothing in
it knows what an item, a model or a result is; the project supplies those.

Install: `pip install "ai-runtime[torch,config] @ git+https://github.com/ekazum/ai-runtime.git@v0.4.2"`. The core is
standard library only; `[torch]` is needed for `ai_runtime.checkpoints`, `[config]` for `ai_runtime.yaml_config`.
Everything else imports from the top level: `from ai_runtime import ...`.

## What ai-runtime provides, what the project writes

| Area | ai-runtime provides | The project writes |
|---|---|---|
| Batching | `GpuBatchStage`, `BatchRequest` | a model meeting `BatchModel` |
| Threads | `ThreadHandle`, `ServiceSupervisor`, `ThreadStages`, `SHUTDOWN_SENTINEL` | the thread targets (`run` methods) |
| Primitives | `ReadWriteLock`, `LockedCounter`, `ServingRegistry` (protocol) | — |
| Logging | `logging.setup_logging`, `logging.LOG_FORMAT` | one `setup_logging()` call per entry point |
| Discovery | `ServiceEndpoint` | the service and its port |
| Stores | `SqliteStore`, `DevSelectionStore` | the rows |
| Checkpoints | `checkpoints.BaseCheckpointManager`, `checkpoints.LatestBestCheckpointManager` | a subclass when it needs more files |
| Config | `yaml_config.YamlConfig` | a subclass with the fields |

## Signatures

```python
class BatchModel(Protocol):                       # the project's model; inherits nothing
    def load(self) -> None: ...                   # once, before the stage's ready is set
    def unload(self) -> None: ...                 # when the stage's loop exits
    def weight(self, item) -> int: ...            # sorts and cuts a window, e.g. prompt tokens
    def process(self, items: list, budgets: list) -> list: ...   # one result per item

BatchRequest(keys: list, items: list, sink: queue.Queue, budgets: list)
GpuBatchStage(model, request_queue, shutdown_event, window_size, idle_timeout_s, max_weight, name="GpuBatchStage")
    .run()  .ready  .form_batches(window)

ThreadHandle(target, name, daemon=False)          .start() .join(timeout) .is_alive()
    ThreadHandle.interruptible_sleep(event, seconds)
ServiceSupervisor(name="ServiceSupervisor", shutdown_event=None)
    .add(target, name, daemon=False, ready=None) .start() .wait_ready(timeout) -> bool
    .dead_threads() .stop(join_timeout) .dial(address, timeout, retry_interval=5.0) .drain(*queues)
ThreadStages(name, shutdown_event=None, join_timeout=30.0)
    .stage(name, targets, inbox=None, sentinel_after=None) .start() .wait() .stop(join_order, raise_errors=True)

ReadWriteLock()        with lock.read_lock(): ...   with lock.write_lock(): ...
LockedCounter()        .add(n) -> int  .value()  .load(value)
ServiceEndpoint(path)  .write(port, extra=None)  .read() -> dict | None  .clear()

SqliteStore(directory, fresh=False)    # runs SCHEMA_PATH, when a subclass sets it, on every open
    .append(table, rows: list[dict], key) .set_pairs(table, values) .read(table, order_by=None) .read_pairs(table)
DevSelectionStore(eval_dir)
    .add_train(epoch, metrics) .add_dev(epoch, checkpoint, metrics) .add_meta(key, value)
    .scored() .selection_state() -> (best composite, epochs without improvement)

checkpoints.BaseCheckpointManager       .atomic_save(payload, path) .load(path, map_location="cpu")
                                        .snapshots(pattern) .promote(path, best_path) .delete(path)
checkpoints.LatestBestCheckpointManager(checkpoint_dir, run_meta=None)
    .save_latest(model, optimizer, step, meta) .save_best(model, step, meta) .find_latest() .best()

yaml_config.YamlConfig                  # frozen pydantic model, extra fields forbidden
    .from_yaml(path)  hooks: _validate(raw) (static), _constructor_kwargs(raw) (class), log_loaded()
```

## Assembly

A batching service (see `examples/minimal/app.py`):

1. `supervisor = ServiceSupervisor("name")`; `requests = queue.Queue()`.
2. `stage = GpuBatchStage(MyModel(), requests, supervisor.shutdown_event, window_size, idle_timeout_s, max_weight)`.
3. `supervisor.add(stage.run, "batch-stage", daemon=True, ready=stage.ready)`; `supervisor.start()`;
   `supervisor.wait_ready(timeout)`.
4. Producers put `BatchRequest(keys, items, sink, budgets)` and read `(key, result)` tuples from their own `sink`.
5. `supervisor.stop(join_timeout)`.

A training pipeline:

1. `stages = ThreadStages("train")`; one `stage(name, target_or_targets, inbox=q)` per thread group, in start order.
2. Each target loops `while not shutdown_event.is_set()` with `q.get(timeout=...)`, and breaks when it gets
   `SHUTDOWN_SENTINEL`.
3. `stages.start()`; `stages.wait()` returns once any thread sets the shutdown event.
4. `stages.stop(join_order)` puts one sentinel per thread on each inbox and joins producers before consumers; a
   stage with `sentinel_after="x"` gets its sentinels only after stage `x` has joined, so it drains what `x` sent.
   A thread's exception re-raises at its join; `raise_errors=False` logs it and joins the rest.

## Rules that are easy to break

- Compare the sentinel by identity: `item is SHUTDOWN_SENTINEL`.
- Thread loops wait with a timeout (`queue.get(timeout=...)`, `ThreadHandle.interruptible_sleep`) so the shutdown
  event is seen; a bare `get()` hangs at stop.
- Start threads through `ThreadHandle`, `ServiceSupervisor` or `ThreadStages`; a target's exception then re-raises on
  `join` instead of vanishing.
- A thread that may block past the join bound is added with `daemon=True`.
- `BatchModel.process` returns exactly one result per item, in item order.
- `SqliteStore` is for one writer process; readers on other hosts are safe (DELETE journal).
- A `SCHEMA_PATH` script runs on every open, so it uses `CREATE TABLE IF NOT EXISTS` (and the same for indexes).
- `YamlConfig` subclasses that add a field validator give it a name the base class does not use; the same name
  replaces the base validator.
- `ReadWriteLock` is not reentrant.

## Examples

- `examples/minimal/app.py` — two producers share one `GpuBatchStage` over a CPU model;
  `python examples/minimal/app.py`.

## The five repos

| Repo | Gives | Depends on |
|---|---|---|
| ai-runtime | threads, `SHUTDOWN_SENTINEL`, `ThreadStages`, GPU batching, service supervision, SQLite stores, checkpoint managers, `YamlConfig`, logging | standard library; `[torch]`, `[config]` |
| [ai-llm](https://github.com/ekazum/ai-llm) | an LLM generation service: engine, GPU worker, server, client, stats store | ai-runtime |
| [ai-ui](https://github.com/ekazum/ai-ui) | a Flask web shell and a runs dashboard over a SQLite runs DB | ai-runtime |
| [rl-core](https://github.com/ekazum/rl-core) | RL threads and data types: PPO, prioritized replay, Q-network inference, DQN learner, checkpoint writer | ai-runtime |
| [ai-pipeline](https://github.com/ekazum/ai-pipeline) | data preparation stages | ai-runtime; `[embed]`, `[llm]`, `[reduce]` |
