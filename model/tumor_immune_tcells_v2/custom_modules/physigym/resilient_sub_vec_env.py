import numpy as np
from typing import Set, List
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _stack_obs
import faulthandler

faulthandler.enable()


# ------------------------------------------------------------------
# Dummy env used to replace crashed ones safely
# ------------------------------------------------------------------
class ToyEnv:
    """Safe placeholder for crashed environments."""

    def __init__(self, observation_space):
        self.observation_space = observation_space
        self.action_space = None

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, True, {"crashed": True, "disabled": True, "step_episode": -1}

    def reset(self, seed=None, options=None):
        obs = self.observation_space.sample()
        return obs, {"crashed": True, "disabled": True, "step_episode": -1}


# ------------------------------------------------------------------
# Resilient VecEnv
# ------------------------------------------------------------------
class ResilientSubprocVecEnv(SubprocVecEnv):
    """
    SubprocVecEnv variant that permanently disables crashing environments
    instead of restarting them (PhysiCell-safe).

    Key fixes vs naive implementation:
      - Observation spaces are cached at __init__ time, so _disable_env
        never needs to spawn a new PhysiCell instance (which is forbidden).
      - remote.poll(timeout) is used before recv() so a SIGABRT that does
        not cleanly close the pipe is caught as a timeout rather than
        hanging forever.
      - step_async checks is_alive() and wraps send() so a process that
        died between steps is caught immediately.
    """

    # Tune this to slightly above your longest expected step duration.
    STEP_TIMEOUT_S: float = 120.0
    RESET_TIMEOUT_S: float = 300.0

    def __init__(self, env_fns, start_method="spawn"):
        assert start_method == "spawn", "PhysiCell requires spawn"

        self.env_fns = env_fns
        self.dead_envs: Set[int] = set()
        self._dummy_envs = {}

        super().__init__(env_fns, start_method=start_method)

        # Make mutable (parent stores as tuples)
        self.remotes = list(self.remotes)
        self.processes = list(self.processes)

        # -------------------------------------------------------
        # Cache observation spaces NOW, before any env can crash.
        # This is the critical fix: _disable_env must never call
        # env_fns[i]() because PhysiCell forbids >1 instance.
        # -------------------------------------------------------
        self._obs_spaces = [self.observation_space] * self.num_envs

    def get_modify_observation_space(self, observation_space):
        self.observation_space = observation_space
        self._obs_spaces = [self.observation_space] * self.num_envs

    # ------------------------------------------------------------------
    # Crash handling
    # ------------------------------------------------------------------
    def _disable_env(self, i: int):
        if i in self.dead_envs:
            return

        print(f"[ResilientVecEnv] Disabling env {i}")
        self.dead_envs.add(i)

        try:
            if self.processes[i].is_alive():
                self.processes[i].terminate()
        except Exception:
            pass

        try:
            self.remotes[i].close()
        except Exception:
            pass

        # Use pre-cached obs space — never spawn a new env here
        self._dummy_envs[i] = ToyEnv(self._obs_spaces[i])

    # ------------------------------------------------------------------
    # Safe set_attr: skips dead envs
    # ------------------------------------------------------------------
    def set_attr(self, attr_name, value, indices=None):
        indices = indices if indices is not None else range(self.num_envs)
        for i in indices:
            if i in self.dead_envs:
                continue
            super().set_attr(attr_name, value, [i])

    # ------------------------------------------------------------------
    # Safe env_method: skips dead envs
    # ------------------------------------------------------------------
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        indices = indices if indices is not None else range(self.num_envs)
        safe_indices = [i for i in indices if i not in self.dead_envs]
        if not safe_indices:
            return []
        return super().env_method(
            method_name, *method_args, indices=safe_indices, **method_kwargs
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step_async(self, actions):
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            if i in self.dead_envs:
                continue
            # Catch processes that died silently between steps
            if not self.processes[i].is_alive():
                print(f"[ResilientVecEnv] Env {i} died between steps, disabling.")
                self._disable_env(i)
                continue
            try:
                remote.send(("step", action))
            except (BrokenPipeError, OSError) as e:
                print(f"[ResilientVecEnv] Send failed on env {i}: {e}")
                self._disable_env(i)
        self.waiting = True

    def step_wait(self):
        results = []

        for i, remote in enumerate(self.remotes):
            if i in self.dead_envs:
                obs, reward, done, info = self._dummy_envs[i].step(None)
                results.append((obs, reward, done, info, info))
                continue

            try:
                if remote.poll(timeout=self.STEP_TIMEOUT_S):
                    results.append(remote.recv())
                else:
                    # Subprocess hung or died without closing the pipe (SIGABRT)
                    print(f"[ResilientVecEnv] Timeout waiting for env {i}, disabling.")
                    self._disable_env(i)
                    obs, reward, done, info = self._dummy_envs[i].step(None)
                    results.append((obs, reward, done, info, info))
            except (EOFError, BrokenPipeError, OSError) as e:
                print(f"[ResilientVecEnv] Pipe error on env {i}: {e}")
                self._disable_env(i)
                obs, reward, done, info = self._dummy_envs[i].step(None)
                results.append((obs, reward, done, info, info))

        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)

        return (
            _stack_obs(obs, self.observation_space),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        for i, remote in enumerate(self.remotes):
            if i in self.dead_envs:
                continue
            if not self.processes[i].is_alive():
                print(f"[ResilientVecEnv] Env {i} dead at reset, disabling.")
                self._disable_env(i)
                continue
            try:
                remote.send(("reset", (self._seeds[i], self._options[i])))
            except (OSError, EOFError, BrokenPipeError) as e:
                print(f"[ResilientVecEnv] Send failed at reset on env {i}: {e}")
                self._disable_env(i)

        results = []
        for i, remote in enumerate(self.remotes):
            if i in self.dead_envs:
                obs, reset_info = self._dummy_envs[i].reset()
                results.append((obs, reset_info))
                continue

            try:
                if remote.poll(timeout=self.RESET_TIMEOUT_S):
                    results.append(remote.recv())
                else:
                    print(f"[ResilientVecEnv] Timeout at reset for env {i}, disabling.")
                    self._disable_env(i)
                    obs, reset_info = self._dummy_envs[i].reset()
                    results.append((obs, reset_info))
            except (EOFError, BrokenPipeError, OSError) as e:
                print(f"[ResilientVecEnv] Pipe error at reset on env {i}: {e}")
                self._disable_env(i)
                obs, reset_info = self._dummy_envs[i].reset()
                results.append((obs, reset_info))

        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()

        return _stack_obs(obs, self.observation_space)

    # ------------------------------------------------------------------
    # Safe close: skips dead envs (their remotes are already closed by
    # _disable_env, so the base-class close() would raise
    # "OSError: handle is closed" when it blindly sends ("close", None)
    # to every remote). Guard each send/join so one bad worker can't
    # abort shutdown.
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.closed:
            return

        if self.waiting:
            for i, remote in enumerate(self.remotes):
                if i in self.dead_envs:
                    continue
                try:
                    remote.recv()
                except (OSError, EOFError, BrokenPipeError):
                    pass

        for i, remote in enumerate(self.remotes):
            if i in self.dead_envs:
                continue
            try:
                remote.send(("close", None))
            except (OSError, EOFError, BrokenPipeError):
                pass

        for process in self.processes:
            try:
                process.join()
            except Exception:
                pass

        self.closed = True
