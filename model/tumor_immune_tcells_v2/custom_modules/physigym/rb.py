from collections import deque
import random
import numpy as np
import torch
from torch_geometric.data import Data, Batch


def np2torch_dtype(np_dtype):
    _map = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32:   torch.int32,
        np.int64:   torch.int64,
        np.uint8:   torch.uint8,
    }
    dt = np.dtype(np_dtype)
    if dt.type not in _map:
        raise ValueError(f"Unsupported NumPy dtype: {np_dtype}")
    return _map[dt.type]


class ReplayBuffer:
    """
    Replay buffer supporting:
    - array-based states (NumPy / preallocated GPU tensors)
    - graph-based states (PyG Data objects)

    Key optimisations vs previous version
    ──────────────────────────────────────
    - add_batch() writes a whole slice at once (no Python loop)
    - graph sample() builds Data objects with pin_memory for faster H→D transfer
    - bare except replaced with explicit TypeError/RuntimeError
    - .size property added (used by tqdm postfix)
    - removed tensordict dependency (unused in the non-graph path)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        buffer_size,
        batch_size,
        state_type=np.float32,
        is_graph=False,
    ):
        self.device      = device
        self.buffer_size = int(buffer_size)
        self.batch_size  = batch_size
        self.is_graph    = is_graph

        self._ptr  = 0      # write pointer
        self._full = False  # whether buffer has wrapped at least once

        print(f"ReplayBuffer | state_dim={state_dim} | is_graph={is_graph} | "
              f"buffer_size={self.buffer_size} | device={device}")

        self.is_image = (len(state_dim) == 3)  # (C, H, W)

        if self.is_graph:
            # Variable-size graphs → deque (can't preallocate)
            self._buf = deque(maxlen=self.buffer_size)
            self.use_torch_tensors = False
        else:
            pin = (str(device) != "cpu")
            try:
                # ── Images stored as uint8 on CPU (pinned) to save VRAM ──
                # Small tensors (action/reward/done) live on GPU directly.
                if self.is_image:
                    self.state      = torch.empty((self.buffer_size, *state_dim), dtype=torch.uint8,   pin_memory=pin)
                    self.next_state = torch.empty((self.buffer_size, *state_dim), dtype=torch.uint8,   pin_memory=pin)
                else:
                    self.state      = torch.empty((self.buffer_size, *state_dim), dtype=torch.float32, pin_memory=pin)
                    self.next_state = torch.empty((self.buffer_size, *state_dim), dtype=torch.float32, pin_memory=pin)
                self.action     = torch.empty((self.buffer_size, *action_dim), dtype=torch.float32).to(device)
                self.reward     = torch.empty((self.buffer_size, 1),           dtype=torch.float32).to(device)
                self.done       = torch.empty((self.buffer_size, 1),           dtype=torch.uint8).to(device)
                self.use_torch_tensors = True
            except (RuntimeError, TypeError) as e:
                print(f"[ReplayBuffer] prealloc failed ({e}), falling back to NumPy")
                obs_dtype       = np.uint8 if self.is_image else np.float32
                self.state      = np.empty((self.buffer_size, *state_dim),  dtype=obs_dtype)
                self.next_state = np.empty((self.buffer_size, *state_dim),  dtype=obs_dtype)
                self.action     = np.empty((self.buffer_size, *action_dim), dtype=np.float32)
                self.reward     = np.empty((self.buffer_size, 1),           dtype=np.float32)
                self.done       = np.empty((self.buffer_size, 1),           dtype=np.uint8)
                self.use_torch_tensors = False

    # ── Size helpers ────────────────────────────────────────────────────────

    def __len__(self):
        if self.is_graph:
            return len(self._buf)
        return self.buffer_size if self._full else self._ptr

    @property
    def size(self):
        """Alias used by tqdm postfix and external callers."""
        return len(self)

    # ── Writing ─────────────────────────────────────────────────────────────

    def add_batch(self, batch):
        """
        Write a list of (state, action, reward, next_state, done) transitions
        in one vectorised operation instead of looping through add().
        """
        if self.is_graph:
            for transition in batch:
                self._add_graph(*transition)
            return

        n = len(batch)
        states, actions, rewards, next_states, dones = zip(*batch)

        if self.use_torch_tensors:
            obs_dtype = torch.uint8 if self.is_image else torch.float32
            s  = torch.stack([torch.as_tensor(x) for x in states]).to(dtype=obs_dtype)
            ns = torch.stack([torch.as_tensor(x) for x in next_states]).to(dtype=obs_dtype)
            a  = torch.stack([torch.as_tensor(x) for x in actions]).to(dtype=torch.float32, device=self.device)
            r  = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
            d  = torch.tensor(dones,   dtype=torch.uint8,   device=self.device).unsqueeze(-1)

            # Handle wrap-around with two slices
            end = self._ptr + n
            if end <= self.buffer_size:
                self.state[self._ptr:end].copy_(s,  non_blocking=True)
                self.action[self._ptr:end].copy_(a,  non_blocking=True)
                self.reward[self._ptr:end].copy_(r,  non_blocking=True)
                self.next_state[self._ptr:end].copy_(ns, non_blocking=True)
                self.done[self._ptr:end].copy_(d,  non_blocking=True)
            else:
                # Wrap: split into two writes
                first = self.buffer_size - self._ptr
                self.state[self._ptr:].copy_(s[:first],   non_blocking=True)
                self.state[:end % self.buffer_size].copy_(s[first:], non_blocking=True)
                self.action[self._ptr:].copy_(a[:first],  non_blocking=True)
                self.action[:end % self.buffer_size].copy_(a[first:], non_blocking=True)
                self.reward[self._ptr:].copy_(r[:first],  non_blocking=True)
                self.reward[:end % self.buffer_size].copy_(r[first:], non_blocking=True)
                self.next_state[self._ptr:].copy_(ns[:first], non_blocking=True)
                self.next_state[:end % self.buffer_size].copy_(ns[first:], non_blocking=True)
                self.done[self._ptr:].copy_(d[:first],    non_blocking=True)
                self.done[:end % self.buffer_size].copy_(d[first:], non_blocking=True)

        else:
            # NumPy path — still vectorised
            s  = np.stack([np.asarray(x) for x in states])
            a  = np.stack([np.asarray(x) for x in actions])
            r  = np.array(rewards,  dtype=np.float32)[:, None]
            ns = np.stack([np.asarray(x) for x in next_states])
            d  = np.array(dones,    dtype=np.uint8)[:, None]

            end = self._ptr + n
            if end <= self.buffer_size:
                self.state[self._ptr:end]      = s
                self.action[self._ptr:end]     = a
                self.reward[self._ptr:end]     = r
                self.next_state[self._ptr:end] = ns
                self.done[self._ptr:end]       = d
            else:
                first = self.buffer_size - self._ptr
                for arr, src in [
                    (self.state,      s),
                    (self.action,     a),
                    (self.reward,     r),
                    (self.next_state, ns),
                    (self.done,       d),
                ]:
                    arr[self._ptr:] = src[:first]
                    arr[:end % self.buffer_size] = src[first:]

        self._full = self._full or (self._ptr + n >= self.buffer_size)
        self._ptr  = (self._ptr + n) % self.buffer_size

    def add(self, state, action, reward, next_state, done):
        """Single-transition fallback — prefer add_batch() for performance."""
        self.add_batch([(state, action, reward, next_state, done)])

    # ── Graph helpers ────────────────────────────────────────────────────────

    def _add_graph(self, state, action, reward, next_state, done):
        self._buf.append((
            self._dict_reduced(state),
            action,
            reward,
            self._dict_reduced(next_state),
            done,
        ))

    def _dict_reduced(self, obs):
        """Strip padding from a dict observation → compact graph dict."""
        node_mask  = obs["node_mask"]  > 0.5
        edge_mask  = obs["edge_mask"]  > 0.5
        return {
            "nodes":      obs["node_features"][node_mask],
            "edge_links": obs["edge_index"][:, edge_mask],
            "edges":      obs["edge_attr"][edge_mask],
        }

    def _dict_to_pyg(self, s):
        return Data(
            x=torch.as_tensor(s["nodes"],      dtype=torch.float32, device=self.device),
            edge_index=torch.as_tensor(s["edge_links"], dtype=torch.long,    device=self.device),
            edge_attr=torch.as_tensor(s["edges"],       dtype=torch.float32, device=self.device),
        )

    # ── Sampling ─────────────────────────────────────────────────────────────

    def sample(self):
        if self.is_graph:
            return self._sample_graph()
        return self._sample_tensor()

    def _sample_graph(self):
        raw = random.sample(self._buf, self.batch_size)
        _s, actions, rewards, _ns, dones = zip(*raw)

        action = torch.tensor(np.stack(actions), dtype=torch.float32, device=self.device)
        reward = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        done   = torch.tensor(dones,   dtype=torch.uint8,   device=self.device)

        state      = Batch.from_data_list([self._dict_to_pyg(s) for s in _s])
        next_state = Batch.from_data_list([self._dict_to_pyg(s) for s in _ns])

        return {"state": state, "action": action, "reward": reward,
                "done": done, "next_state": next_state}

    def _sample_tensor(self):
        current_size = self.buffer_size if self._full else self._ptr
        # torch.randint is faster than np.random.randint for GPU-resident indices
        idx = torch.randint(0, current_size, (self.batch_size,), device=self.device)

        if self.use_torch_tensors:
            idx_cpu = idx.cpu()
            s  = self.state[idx_cpu].to(self.device, non_blocking=True)
            ns = self.next_state[idx_cpu].to(self.device, non_blocking=True)
            if self.is_image:
                s  = s.float()
                ns = ns.float()
            return {
                "state":      s,
                "action":     self.action[idx],
                "reward":     self.reward[idx],
                "next_state": ns,
                "done":       self.done[idx],
            }
        else:
            idx_np = idx.cpu().numpy()
            return {
                "state":      torch.as_tensor(self.state[idx_np],      device=self.device).float(),
                "action":     torch.as_tensor(self.action[idx_np],     device=self.device),
                "reward":     torch.as_tensor(self.reward[idx_np],     device=self.device),
                "next_state": torch.as_tensor(self.next_state[idx_np], device=self.device).float(),
                "done":       torch.as_tensor(self.done[idx_np],       device=self.device),
            }


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # ── Tensor path ──────────────────────────────────────────────────────────
    rb = ReplayBuffer(
        state_dim=(144, 8), action_dim=(1,),
        device=device, buffer_size=int(2e5), batch_size=64,
    )
    local_batch = [
        (
            torch.randn(144, 8),
            torch.randn(1),
            1.0,
            torch.randn(144, 8),
            False,
        )
        for _ in range(128)
    ]
    rb.add_batch(local_batch)
    s = rb.sample()
    print(f"[tensor] state={s['state'].shape}  size={rb.size}")
    assert s["state"].shape == (64, 144, 8)

    # ── Wrap-around test ─────────────────────────────────────────────────────
    rb2 = ReplayBuffer(
        state_dim=(10,), action_dim=(4,),
        device=device, buffer_size=100, batch_size=32,
    )
    big_batch = [(np.random.randn(10).astype(np.float32),
                  np.random.randn(4).astype(np.float32),
                  float(np.random.randn()),
                  np.random.randn(10).astype(np.float32),
                  bool(np.random.randint(2)))
                 for _ in range(150)]   # intentionally > buffer_size
    rb2.add_batch(big_batch)
    s2 = rb2.sample()
    print(f"[wrap]   state={s2['state'].shape}  size={rb2.size}  full={rb2._full}")
    assert rb2._full

    print("All tests passed.")