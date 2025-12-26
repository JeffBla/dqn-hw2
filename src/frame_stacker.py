import cv2
import numpy as np
from collections import deque


class FrameStacker():

    def __init__(self, nFrame, w, h):
        self.frame_stack = deque(maxlen=nFrame)
        self.frame_stack_test = deque(maxlen=nFrame)
        self.isTrain = True
        self.nFrame = nFrame
        self.w = w
        self.h = h

    def eval_mode(self):
        self.isTrain = False

    def train_mode(self):
        self.isTrain = True

    def isFull(self):
        return len(self.frame_stack) == self.nFrame

    def reset(self, init_obs):
        out_target = None
        obs_gray = self.preprocess(init_obs)
        if self.isTrain:
            for _ in range(self.nFrame):
                self.frame_stack.append(obs_gray)
            out_target = self.frame_stack
        else:
            for _ in range(self.nFrame):
                self.frame_stack_test.append(obs_gray)
            out_target = self.frame_stack_test
        return np.concatenate(list(out_target), 0)

    def push(self, new_obs):
        out_target = None
        obs_gray = self.preprocess(new_obs)
        if self.isTrain:
            self.frame_stack.append(obs_gray)
            out_target = self.frame_stack
        else:
            self.frame_stack_test.append(obs_gray)
            out_target = self.frame_stack_test
        return np.concatenate(list(out_target), 0)

    def preprocess(self, obs):
        # Handle both RGB and already-grayscale observations.
        if obs.ndim == 3 and obs.shape[-1] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 2:
            gray = obs
        elif obs.ndim == 3 and obs.shape[0] == 1:
            # (1, H, W) -> (H, W)
            gray = obs[0]
        else:
            # Fallback: try to squeeze any singleton dim
            gray = np.squeeze(obs)
            if gray.ndim != 2:
                raise ValueError(f"Unsupported observation shape for preprocessing: {obs.shape}")

        if gray.shape[:2] != (self.h, self.w):
            gray = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # Normalize to [0,1] float for network input
        img = gray.astype(np.float32) / 255.0
        return img[None, :, :]


class MultiFrameStacker:
    """
    Batched version of FrameStacker (no new file).
    - Maintains per-env deques for train and eval modes.
    - reset(obs_batch) -> (N, C, H, W)
    - push(obs_batch, reset_mask) -> (N, C, H, W)
    """

    def __init__(self, num_envs: int, nFrame: int, w: int, h: int):
        self.num_envs = num_envs
        self.nFrame = nFrame
        self.w = w
        self.h = h
        self.isTrain = True
        self.frame_stacks = [deque(maxlen=nFrame) for _ in range(num_envs)]
        self.frame_stacks_test = [deque(maxlen=nFrame) for _ in range(num_envs)]

    def eval_mode(self):
        self.isTrain = False

    def train_mode(self):
        self.isTrain = True

    def _preprocess(self, obs):
        # Mirror FrameStacker.preprocess
        if obs.ndim == 3 and obs.shape[-1] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 2:
            gray = obs
        elif obs.ndim == 3 and obs.shape[0] == 1:
            gray = obs[0]
        else:
            gray = np.squeeze(obs)
            if gray.ndim != 2:
                raise ValueError(f"Unsupported observation shape for preprocessing: {obs.shape}")

        if gray.shape[:2] != (self.h, self.w):
            gray = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)

        img = gray.astype(np.float32) / 255.0
        return img[None, :, :]

    def _target_stacks(self):
        return self.frame_stacks if self.isTrain else self.frame_stacks_test

    def reset(self, obs_batch):
        assert len(obs_batch) == self.num_envs
        stacks = self._target_stacks()
        states = []
        for i in range(self.num_envs):
            stacks[i].clear()
            proc = self._preprocess(obs_batch[i])
            for _ in range(self.nFrame):
                stacks[i].append(proc)
            states.append(np.concatenate(list(stacks[i]), 0))
        return np.stack(states, axis=0)

    def reset_at(self, idx: int, obs):
        stacks = self._target_stacks()
        stacks[idx].clear()
        proc = self._preprocess(obs)
        for _ in range(self.nFrame):
            stacks[idx].append(proc)
        return np.concatenate(list(stacks[idx]), 0)

    def push(self, obs_batch, reset_mask=None):
        stacks = self._target_stacks()
        if reset_mask is None:
            reset_mask = np.zeros(self.num_envs, dtype=bool)
        states = []
        for i in range(self.num_envs):
            if reset_mask[i]:
                states.append(self.reset_at(i, obs_batch[i]))
            else:
                proc = self._preprocess(obs_batch[i])
                stacks[i].append(proc)
                states.append(np.concatenate(list(stacks[i]), 0))
        return np.stack(states, axis=0)
