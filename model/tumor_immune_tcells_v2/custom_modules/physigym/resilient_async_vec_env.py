"""
ResilientAsyncSubprocVecEnv
===========================

Drop-in extension of `ResilientSubprocVecEnv` that adds a **true asynchronous,
per-environment** stepping API on top of the existing batched (lock-step) one.

Why this exists
---------------
`SubprocVecEnv.step(actions)` is `step_async(actions)` followed immediately by
`step_wait()`. `step_wait()` blocks until EVERY worker has replied, so every
step is gated by the slowest environment. With PhysiCell episodes of uneven
length, the fast workers sit idle waiting for one straggler — which is why the
machine is only fractionally used.

The async API here removes that barrier. Each worker is kept permanently busy:
the instant a worker returns a result you can hand it its next action, without
waiting for its siblings. This is what actually fills all the cores.

Two APIs, one class
-------------------
1. Batched (inherited, unchanged): `step(actions)` / `step_async` / `step_wait`.
   Existing code keeps working exactly as before.
2. Async (new):
     - `async_reset()`            -> initial obs for every live env
     - `step_send(i, action)`     -> dispatch one env's step, non-blocking
     - `poll_ready(timeout)`      -> list of env indices with a result waiting
     - `recv_env(i)`              -> (obs, reward, done, info) for that env
   Typical loop:
       obs = envs.async_reset()
       for i in range(n): envs.step_send(i, policy(obs[i]))
       while training:
           for i in envs.poll_ready():
               o, r, d, info = envs.recv_env(i)
               store_transition(i, o, r, d, info)
               envs.step_send(i, policy(o))   # immediately re-dispatch

Resilience is preserved: a worker that dies, hangs, or breaks its pipe is
disabled exactly as in the batched path and its `recv_env` returns a dummy
terminal transition so the caller's per-env bookkeeping never stalls.
"""

import numpy as np
from typing import List, Optional
from multiprocessing.connection import wait as mp_wait

from resilient_sub_vec_env import ResilientSubprocVecEnv


class ResilientAsyncSubprocVecEnv(ResilientSubprocVecEnv):
    # ------------------------------------------------------------------
    # Async lifecycle
    # ------------------------------------------------------------------
    def __init__(self, env_fns, start_method="spawn"):
        super().__init__(env_fns, start_method=start_method)
        # Which live envs currently have an outstanding ("step", action) that
        # has been sent but whose reply has not yet been recv()'d. Used so we
        # never recv() from a worker we did not dispatch, and so poll_ready()
        # only reports envs we are actually waiting on.
        self._pending: List[bool] = [False] * self.num_envs
        # Map a raw connection object back to its env index for fast lookup
        # after multiprocessing.connection.wait() returns ready connections.
        self._remote_to_idx = {}
        self._rebuild_remote_index()

    def _rebuild_remote_index(self):
        """Refresh the connection->index map (call after disabling an env)."""
        self._remote_to_idx = {
            id(remote): i
            for i, remote in enumerate(self.remotes)
            if i not in self.dead_envs
        }

    def _disable_env(self, i: int):
        # Keep the base-class disabling behavior, then clear async bookkeeping
        # so this env is never polled or recv'd from again.
        super()._disable_env(i)
        if 0 <= i < len(self._pending):
            self._pending[i] = False
        self._rebuild_remote_index()

    # ------------------------------------------------------------------
    # Async reset: send reset to every live env, collect initial obs.
    # Mirrors the batched reset() but is safe to call before async stepping.
    # ------------------------------------------------------------------
    def async_reset(self):
        return self.reset()

    # ------------------------------------------------------------------
    # Send a single env's step. Non-blocking. Handles a worker that died
    # between steps exactly like the batched step_async does.
    # ------------------------------------------------------------------
    def step_send(self, i: int, action) -> bool:
        """
        Dispatch `action` to env `i`. Returns True if the step was sent to a
        live worker, False if the env is dead/disabled (in which case the
        caller should still call recv_env(i) to get a dummy terminal result).
        """
        if i in self.dead_envs:
            # Nothing sent; mark pending so recv_env returns a dummy result and
            # the caller's request/response accounting stays balanced.
            self._pending[i] = True
            return False

        if not self.processes[i].is_alive():
            print(f"[AsyncVecEnv] Env {i} died before step_send, disabling.")
            self._disable_env(i)
            self._pending[i] = True
            return False

        try:
            self.remotes[i].send(("step", action))
            self._pending[i] = True
            return True
        except (BrokenPipeError, OSError) as e:
            print(f"[AsyncVecEnv] Send failed on env {i}: {e}")
            self._disable_env(i)
            self._pending[i] = True
            return False

    # ------------------------------------------------------------------
    # Return the indices of envs whose step result is ready to recv without
    # blocking. Dead envs with a pending dummy are always "ready". Real envs
    # are checked via multiprocessing.connection.wait so we block at most
    # `timeout` seconds total across ALL pipes (not per-pipe).
    # ------------------------------------------------------------------
    def poll_ready(self, timeout: Optional[float] = 0.0) -> List[int]:
        ready: List[int] = []

        # Dead envs that were "sent" a step resolve immediately to a dummy.
        for i in range(self.num_envs):
            if self._pending[i] and i in self.dead_envs:
                ready.append(i)

        live_pending = [
            self.remotes[i]
            for i in range(self.num_envs)
            if self._pending[i] and i not in self.dead_envs
        ]
        if live_pending:
            ready_conns = mp_wait(live_pending, timeout=timeout)
            for conn in ready_conns:
                idx = self._remote_to_idx.get(id(conn))
                if idx is not None:
                    ready.append(idx)

        return ready

    # ------------------------------------------------------------------
    # Receive one env's step result. Assumes the caller confirmed readiness
    # via poll_ready (or accepts blocking up to STEP_TIMEOUT_S otherwise).
    # Returns (obs, reward, done, info) — single-env, NOT stacked.
    # ------------------------------------------------------------------
    def recv_env(self, i: int):
        if not self._pending[i]:
            raise RuntimeError(
                f"recv_env({i}) called with no pending step. "
                f"Call step_send({i}, action) first."
            )
        self._pending[i] = False

        if i in self.dead_envs:
            obs, reward, done, info = self._dummy_envs[i].step(None)
            return obs, reward, done, info

        remote = self.remotes[i]
        try:
            if remote.poll(timeout=self.STEP_TIMEOUT_S):
                obs, reward, done, info, reset_info = remote.recv()
                return obs, reward, done, info
            else:
                print(f"[AsyncVecEnv] Timeout waiting for env {i}, disabling.")
                self._disable_env(i)
                obs, reward, done, info = self._dummy_envs[i].step(None)
                return obs, reward, done, info
        except (EOFError, BrokenPipeError, OSError) as e:
            print(f"[AsyncVecEnv] Pipe error on env {i}: {e}")
            self._disable_env(i)
            obs, reward, done, info = self._dummy_envs[i].step(None)
            return obs, reward, done, info

    # ------------------------------------------------------------------
    # Drain any outstanding async steps so close()/reset() don't leave
    # unread replies stuck in the pipes. Safe to call any time.
    # ------------------------------------------------------------------
    def drain_pending(self):
        for i in range(self.num_envs):
            if self._pending[i]:
                try:
                    self.recv_env(i)
                except Exception:
                    self._pending[i] = False

    def close(self) -> None:
        # Async steps in flight would otherwise wedge the base close(), which
        # sends ("close", None) and expects clean pipes. Drain first.
        if not self.closed:
            self.drain_pending()
        super().close()
