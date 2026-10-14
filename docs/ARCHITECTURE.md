# ai-runtime Architecture

## Table of Contents

1. [Overview](#1-overview)
2. [Modules](#2-modules)
3. [Batching](#3-batching)
4. [Supervision](#4-supervision)
5. [Primitives](#5-primitives)
6. [Service Discovery](#6-service-discovery)
7. [Design Principles](#7-design-principles)
8. [Directory Structure](#8-directory-structure)

---

## 1. Overview

The mechanics of a threaded ML pipeline: moving work between threads, batching it for a model, and starting,
watching and stopping the threads that do it. A project supplies what the work means — its model, its items and
their weights — and the runtime owns the threads and queues around it.

The package depends on the standard library only. The project's model, whatever it is built on, enters through the
`BatchModel` protocol.

---

## 2. Modules

| File                  | Class                                         | Role                                                        |
|-----------------------|-----------------------------------------------|-------------------------------------------------------------|
| `batching.py`         | `GpuBatchStage`, `BatchModel`, `BatchRequest` | Weight-sorted batching of queued work under a budget        |
| `supervisor.py`       | `ServiceSupervisor`                           | A service's threads under one shutdown event                |
| `thread_handle.py`    | `ThreadHandle`                                | Thread whose exception re-raises on `join`                  |
| `rw_lock.py`          | `ReadWriteLock`                               | Many readers or one writer, writer-preferring               |
| `counters.py`         | `LockedCounter`                               | An int that many threads advance                            |
| `serving_registry.py` | `ServingRegistry`                             | Protocol a worker notifies as it starts and stops serving   |
| `service_endpoint.py` | `ServiceEndpoint`                             | Host and port of a running service, in a shared JSON file   |

---

## 3. Batching

`GpuBatchStage.run` loads the model, sets `ready`, and loops until the shutdown event:

1. Take a `BatchRequest` from the request queue and add one entry per item to the window.
2. Flush when the window holds `window_size` entries, or when the queue has been idle for `idle_timeout_s`.
3. Sort the window by `model.weight(item)` and cut it greedily into batches whose total weight stays within
   `max_weight`, so similar-weight items share a batch.
4. Run each batch through `model.process(items, budgets)` and put `(key, result)` on each item's `sink` as soon as its
   batch finishes.

The model is unloaded when the loop exits, also on an exception. A request carries its own `sink`, so many producers
share one stage and each reads only its own results. `budgets` passes a per-item limit to the model, such as a reply
length.

| `BatchModel` method        | Called                                   |
|----------------------------|------------------------------------------|
| `load()`                   | once, before `ready` is set              |
| `weight(item) -> int`      | when sorting and cutting a window        |
| `process(items, budgets)`  | once per batch; one result per item      |
| `unload()`                 | when the loop exits                      |

---

## 4. Supervision

`ServiceSupervisor` holds a service's threads and one shutdown event:

| Method                               | Behaviour                                                                      |
|--------------------------------------|--------------------------------------------------------------------------------|
| `add(target, name, daemon, ready)`   | Registers a thread; `ready` is the event it sets once it can serve             |
| `start()`                            | Starts every registered thread                                                 |
| `wait_ready(timeout)`                | Blocks until every registered ready event is set; safe while `start` runs      |
| `dead_threads()`                     | Names of threads that have exited                                              |
| `stop(join_timeout)`                 | Sets the shutdown event and joins each thread for at most `join_timeout`       |
| `dial(address, timeout)`             | Connects with `multiprocessing.connection`, retrying while the server starts   |
| `drain(*queues)`                     | Discards queued work, e.g. when the asker is gone with the server              |

A thread that may be wedged past the join bound is added with `daemon=True`, so it cannot hold the process open. An
exception in a thread re-raises on its join (`ThreadHandle`).

---

## 5. Primitives

- `ThreadHandle` wraps `threading.Thread`, captures any exception the target raises and re-raises it on `join`, so a
  crashed worker cannot be ignored. `interruptible_sleep(event, seconds)` wakes early when the event is set.
- `ReadWriteLock` gives concurrent readers or one writer. Once a writer waits, new readers queue behind it. The
  protected work runs outside the internal mutex. It is not reentrant.
- `LockedCounter` advances under a lock; `add(n)` returns the new total, from which a caller can derive unique indices.

---

## 6. Service Discovery

`ServiceEndpoint(path)` publishes a running service: `write(port, extra)` stores the node's FQDN, the port, the start
time and any extra fields as JSON; `read()` returns that dict, or `None` when the file is missing, unreadable or not an
endpoint; `clear()` removes it when the service stops. Clients on any machine that shares the path find the service
without querying a scheduler.

---

## 7. Design Principles

- **The project supplies meaning, the runtime supplies mechanics.** Nothing in the package knows what an item, a
  model or a result is.
- **Protocols over base classes.** A project implements `BatchModel` on its own class; it inherits nothing.
- **Plain constructor arguments.** Sizes and timeouts are passed in; the package reads no configuration files.
- **Standard library only.**

---

## 8. Directory Structure

```
ai-runtime/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/
│   └── ARCHITECTURE.md
├── examples/
│   └── minimal/
│       └── app.py
└── src/
    └── ai_runtime/
        ├── __init__.py
        ├── batching.py
        ├── counters.py
        ├── rw_lock.py
        ├── service_endpoint.py
        ├── serving_registry.py
        ├── supervisor.py
        └── thread_handle.py
```
