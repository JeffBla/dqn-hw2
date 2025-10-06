import numpy as np
import torch
import cv2
import random


class ReplayMemory(object):

    def __init__(self, capacity, nFrame, w, h):
        self.h = h
        self.w = w
        self.capacity = capacity
        self.nFrame = nFrame
        self.ptr = 0
        self.size = 0
        self.obs = np.empty((self.capacity, h, w), dtype=np.uint8)
        self.act = np.empty((self.capacity, ), dtype=np.uint8)
        self.rew = np.empty((self.capacity, ), dtype=np.float32)
        self.done = np.empty((self.capacity, ), dtype=np.bool_)

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
        """Saves a transition"""
        obs, action, reward, done = transition

        i = self.ptr
        self.obs[i] = self._to_gray_u8(obs)
        self.act[i] = np.uint8(action)
        self.rew[i] = np.float32(reward)
        self.done[i] = bool(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _valid_end(self, e):
        if self.size < self.nFrame + 1: return False
        # 避免窗口跨越寫入斷點與 episode 邊界
        next_is_break = (self.size == self.capacity) and (
            (e + 1) % self.capacity == self.ptr)
        if next_is_break: return False
        for t in range(self.nFrame - 1):
            i = (e - t) % self.capacity
            if (self.size == self.capacity) and (i == self.ptr): return False
            if self.done[(i - 1) % self.capacity]: return False
        return True

    def _stack(self, end_indices):
        B = len(end_indices)
        out = np.empty((B, self.nFrame, self.h, self.w), dtype=np.uint8)
        for b, e in enumerate(end_indices):
            for t in range(self.nFrame):
                out[b,
                    t] = self.obs[(e - (self.nFrame - 1 - t)) % self.capacity]
        return out

    def sample(self, batch_size, device):
        """Sample a batch of transitions"""
        idxs = []
        tries, max_tries = 0, batch_size * 200
        if self.size < self.capacity:
            low, high = self.nFrame - 1, self.size - 2
            while len(idxs) < batch_size and tries < max_tries:
                e = random.randint(low, high)
                # prevent cross other episode
                s, ed = e - (self.nFrame - 1), e
                if not self.done[s:ed].any():
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
        sp_u8 = self._stack((idxs + 1) % self.capacity)
        a = self.act[idxs].astype(np.int64)
        r = self.rew[idxs]
        d = self.done[idxs]

        states = torch.from_numpy(s_u8).float().div_(255.0).to(device)
        next_states = torch.from_numpy(sp_u8).float().div_(255.0).to(device)
        actions = torch.from_numpy(a).to(device)
        rewards = torch.from_numpy(r).to(device)
        dones = torch.from_numpy(d.astype(np.bool_)).to(device)
        return states, actions, rewards, next_states, dones
