"""Per-iteration log of the scheduler loop (added for the GRASS project).

Off by default. When the environment variable ``GRASS_ITER_LOG`` names a file, ``BaseScheduler.run``
records one row per loop iteration and writes them to that file as a compressed ``.npz`` when the
run ends. Nothing else in the engine changes: no extra GPU synchronisation, no extra GPU work.

What one row holds (``COLUMNS``):

* six wall-clock times in seconds since the first iteration: loop start, after admission (which
  includes KV offload and restore), after the forward inputs are built, after the forward pass was
  issued, after ``update_requests``, and loop end;
* what the forward pass of this iteration was asked to do, split by request state (NORMAL, DRAFT,
  VERIFY, STALL): number of requests, query tokens, KV tokens loaded (the sparse selection for a
  DRAFT request, the whole cache otherwise), and the sum over requests of query tokens times KV
  tokens loaded;
* queue state: active, pending, offloaded, completed requests; free GPU and host KV pages;
* cumulative counters kept elsewhere in the engine: CUDA-graph token counts (total, captured,
  padded) and speculative tokens (proposed, accepted).

Reading the times. With ``--async_cpu`` the forward pass of iteration ``j`` is only issued in
iteration ``j``; the loop waits for it at the start of the forward section of iteration ``j + 1``.
The GPU time of iteration ``j`` therefore shows up in the row of iteration ``j + 1``. Without
``--async_cpu`` it shows up in the same row. The log does not correct for this; whoever fits a cost
model to it has to.

This module imports only numpy, so it can be tested without a GPU.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

ENV_VAR = "GRASS_ITER_LOG"
STATES = ("normal", "draft", "verify", "stall")   # order of Request.ReqExecType values 0..3

COLUMNS = (
    ["t_start", "t_admit", "t_prepare", "t_forward", "t_update", "t_end", "forward"]
    + ["n_active", "n_pending", "n_offloaded", "n_completed"]
    + [f"{kind}_{s}" for kind in ("n", "q", "kv", "qkv") for s in STATES]
    + ["kv_full_active", "gpu_free_pages", "cpu_free_pages"]
    + ["graph_total", "graph_captured", "graph_padded", "spec_proposed", "spec_accepted"]
)
_COL = {name: i for i, name in enumerate(COLUMNS)}


def state_sums(status: np.ndarray, query_tokens: np.ndarray, kv_loaded: np.ndarray) -> np.ndarray:
    """Per-state counts and sums for one iteration, as a flat array ordered like ``COLUMNS``.

    ``status`` holds the ``ReqExecType`` value of each active request, ``query_tokens`` the number
    of tokens the request sends through the model in this iteration and ``kv_loaded`` the number
    of KV tokens its attention reads. Returns 16 numbers: n, q, kv and q*kv for each of the 4
    states.
    """
    status = np.asarray(status, dtype=np.int64)
    q = np.asarray(query_tokens, dtype=np.float64)
    kv = np.asarray(kv_loaded, dtype=np.float64)
    if not (len(status) == len(q) == len(kv)):
        raise ValueError(f"length mismatch: {len(status)} states, {len(q)} token counts, {len(kv)} KV lengths")
    n_states = len(STATES)
    out = np.zeros(4 * n_states)
    if len(status):
        if status.min() < 0 or status.max() >= n_states:
            raise ValueError(f"unknown request state in {np.unique(status)}")
        out[0:n_states] = np.bincount(status, minlength=n_states)
        out[n_states:2 * n_states] = np.bincount(status, weights=q, minlength=n_states)
        out[2 * n_states:3 * n_states] = np.bincount(status, weights=kv, minlength=n_states)
        out[3 * n_states:4 * n_states] = np.bincount(status, weights=q * kv, minlength=n_states)
    return out


class IterLog:
    """Collects rows in memory and writes them once. Use ``IterLog.from_env()``."""

    def __init__(self, path: str, capacity: int = 4096):
        self.path = path
        self._rows = np.zeros((capacity, len(COLUMNS)))
        self._n = 0
        self._t0: Optional[float] = None
        self._cur = np.zeros(len(COLUMNS))

    @classmethod
    def from_env(cls) -> Optional["IterLog"]:
        path = os.environ.get(ENV_VAR, "").strip()
        return cls(path) if path else None

    def __len__(self) -> int:
        return self._n

    def mark(self, name: str) -> None:
        """Store the current time in column ``name`` of the row being built."""
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self._cur[_COL[name]] = now - self._t0

    def snapshot(self, scheduler, input_metadata, n_pending: int, n_completed: int) -> None:
        """Record what this iteration's forward pass does. Call right after ``prepare_forward_inputs``."""
        cur = self._cur
        active = scheduler.active_requests
        status = np.fromiter((r.status.value for r in active), dtype=np.int64, count=len(active))
        kv_full = np.fromiter((r.len_kv_cache for r in active), dtype=np.int64, count=len(active))
        if input_metadata is None:
            # every active request is waiting for its verification result: no forward pass
            q = np.zeros(len(active))
            kv = np.zeros(len(active))
            cur[_COL["forward"]] = 0
        else:
            q = np.diff(np.frombuffer(input_metadata["token_indptr"], dtype=np.int32))
            kv = np.fromiter((r.len_loading_kv_cache for r in active), dtype=np.int64, count=len(active))
            cur[_COL["forward"]] = 1
        first = _COL[f"n_{STATES[0]}"]
        cur[first:first + 4 * len(STATES)] = state_sums(status, q, kv)
        cur[_COL["n_active"]] = len(active)
        cur[_COL["n_pending"]] = n_pending
        cur[_COL["n_offloaded"]] = len(scheduler.offload_requests)
        cur[_COL["n_completed"]] = n_completed
        cur[_COL["kv_full_active"]] = kv_full.sum()
        pool = scheduler.kv_pool
        cur[_COL["gpu_free_pages"]] = pool.num_free_pages
        cur[_COL["cpu_free_pages"]] = len(pool._free_cpu)
        model = scheduler.model
        cur[_COL["graph_total"]] = getattr(model, "num_total_tokens", 0)
        cur[_COL["graph_captured"]] = getattr(model, "num_captured_tokens", 0)
        cur[_COL["graph_padded"]] = getattr(model, "num_padded_tokens", 0)

    def set_spec_counters(self, proposed: float, accepted: float) -> None:
        self._cur[_COL["spec_proposed"]] = proposed
        self._cur[_COL["spec_accepted"]] = accepted

    def commit(self) -> None:
        """Finish the row being built and start the next one."""
        if self._n == len(self._rows):
            self._rows = np.concatenate([self._rows, np.zeros_like(self._rows)])
        self._rows[self._n] = self._cur
        self._n += 1
        self._cur = np.zeros(len(COLUMNS))

    def save(self, meta: dict) -> None:
        """Write ``rows`` (iterations x columns), ``columns`` and ``meta`` (a JSON string)."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, rows=self._rows[: self._n], columns=np.array(COLUMNS),
                            meta=np.array(json.dumps(meta, default=str)))
        os.replace(tmp, self.path)   # a preempted job never leaves a half-written log behind
