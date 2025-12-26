import numpy as np
import torch
import cv2
import random


class ReplayMemory(object):
    """Experience replay with optional stride-aware stacking for vector envs.

    When `num_envs > 1`, stacked frames for a state are gathered with a stride
    of `num_envs` so that all frames in a stack come from the same environment
    timeline: indices [e - (k-1)*num_envs, ..., e - num_envs, e]. The
    corresponding next-state stack ends at `e + num_envs`.
    """

    def __init__(self, capacity, nFrame, w, h, num_envs: int = 1):
        self.h = h
        self.w = w
        self.capacity = capacity
        self.nFrame = nFrame
        self.num_envs = int(num_envs) if num_envs is not None else 1
        self.ptr = 0
        self.size = 0
        self.obs = np.empty((self.capacity, h, w), dtype=np.uint8)
        self.act = np.empty((self.capacity, ), dtype=np.uint8)
        self.rew = np.empty((self.capacity, ), dtype=np.float32)
        self.done = np.empty((self.capacity, ), dtype=np.bool_)
        # Track environment id per transition for vectorized rollouts
        self.env_id = np.empty((self.capacity, ), dtype=np.int16)

    def __len__(self):
        return self.size

    def _to_gray_u8(self, obs_rgb):
        if obs_rgb.ndim == 3 and obs_rgb.shape[-1] == 3:
            g = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2GRAY)
        else:
            g = obs_rgb
        if g.shape[:2] != (self.h, self.w):
            g = cv2.resize(g, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        return g

    def append(self, *transition):
        """Saves a transition. Accepts (obs, action, reward, done[, env_id])"""
        if len(transition) == 4:
            obs, action, reward, done = transition
            env_id = 0
        elif len(transition) == 5:
            obs, action, reward, done, env_id = transition
        else:
            raise ValueError(
                "append expects 4 or 5 elements: obs, action, reward, done[, env_id]"
            )

        i = self.ptr
        self.obs[i] = self._to_gray_u8(obs)
        self.act[i] = np.uint8(action)
        self.rew[i] = np.float32(reward)
        self.done[i] = bool(done)
        self.env_id[i] = np.int16(env_id)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _valid_end(self, e):
        stride = self.num_envs if self.num_envs > 1 else 1
        # Need enough elements to form state (nFrame) and a next end (e+stride)
        if self.size < (self.nFrame * stride) + 1:
            return False

        cap = self.capacity
        # window indices: [e-(k-1), ..., e-1, e]
        widx = (e - (np.arange(self.nFrame)[::-1] * stride)) % cap

        # 1) Same env across the window
        ids = self.env_id[widx]
        if not np.all(ids == ids[-1]):
            return False

        # 2) No done within the window except the last transition
        if self.done[widx[:-1]].any():
            return False

        # 3) Do not cross write pointer when full (for state and next-state)
        if self.size == self.capacity:
            if self.ptr in widx:
                return False
            widx_next = (widx + stride) % cap
            if self.ptr in widx_next:
                return False
            if ((e + stride) % cap) == self.ptr:
                return False

        return True

    def _stack(self, end_indices):
        B = len(end_indices)
        out = np.empty((B, self.nFrame, self.h, self.w), dtype=np.uint8)
        stride = self.num_envs if self.num_envs > 1 else 1
        for b, e in enumerate(end_indices):
            for t in range(self.nFrame):
                idx = (e - ((self.nFrame - 1 - t) * stride)) % self.capacity
                out[b, t] = self.obs[idx]
        return out

    def sample(self, batch_size, device):
        """Sample a batch of transitions"""
        idxs = []
        tries, max_tries = 0, batch_size * 1000
        stride = self.num_envs if self.num_envs > 1 else 1
        if self.size < self.capacity:
            # Ensure we can build [state, next_state] without wrap
            low, high = (self.nFrame - 1) * stride, self.size - stride - 1
            while len(idxs) < batch_size and tries < max_tries:
                e = random.randint(low, max(high, low))
                if self._valid_end(e):
                    idxs.append(e)
                tries += 1
        else:
            while len(idxs) < batch_size and tries < max_tries:
                e = random.randint(0, self.capacity - 1)
                if self._valid_end(e):
                    idxs.append(e)
                tries += 1

        if len(idxs) < batch_size:
            raise RuntimeError(
                "Failed to sample enough valid indices; try later.")

        idxs = np.asarray(idxs, dtype=np.int64)
        s_u8 = self._stack(idxs)
        sp_u8 = self._stack((idxs + stride) % self.capacity)
        a = self.act[idxs].astype(np.int64)
        r = self.rew[idxs]
        d = self.done[idxs]

        # Use pinned memory + non_blocking transfers to speed up H2D copies.
        states = (torch.from_numpy(s_u8).pin_memory().float().div_(255.0).to(
            device, non_blocking=True))
        next_states = (
            torch.from_numpy(sp_u8).pin_memory().float().div_(255.0).to(
                device, non_blocking=True))
        actions = torch.from_numpy(a).pin_memory().to(device,
                                                      non_blocking=True)
        rewards = torch.from_numpy(r).pin_memory().to(device,
                                                      non_blocking=True)
        dones = (torch.from_numpy(d.astype(np.bool_)).pin_memory().to(
            device, non_blocking=True))
        return states, actions, rewards, next_states, dones
