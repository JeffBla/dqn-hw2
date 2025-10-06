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

    def preprocess(self, obs_rgb):
        gray = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(gray, (self.w, self.h))
        img = img.astype(np.float32) / 255.0
        return img[None, :, :]
