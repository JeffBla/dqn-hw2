from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import numpy as np
import random
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN


class AtariDQNAgent(DQNBaseAgent):

    def __init__(self, config):
        super(AtariDQNAgent, self).__init__(config)
        ### TODO ###
        self.env = gym.make(config["env_id"], render_mode="rgb_array")

        ### TODO ###
        self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
        # self.test_env = gym.wrappers.RecordVideo(
        #     self.test_env, config["logdir"], episode_trigger=lambda x: True)

        # initialize behavior network and target network
        self.behavior_net = AtariNetDQN(self.env.action_space.n)
        self.behavior_net.to(self.device)
        self.target_net = AtariNetDQN(self.env.action_space.n)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(self.behavior_net.parameters(),
                                      lr=self.lr,
                                      eps=1.5e-4)

    def decide_agent_actions(self,
                             observation,
                             epsilon=0.0,
                             action_space=None):
        ### TODO ###
        # get action from behavior net, with epsilon-greedy selection

        if random.random() < epsilon:
            action = action_space.sample()
        else:
            obs_tensor = torch.tensor(observation,
                                      dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.behavior_net(obs_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size, self.device)

        ### TODO ###
        # calculate the loss and update the behavior network
        # 1. get Q(s,a) from behavior net
        # 2. get max_a Q(s',a) from target net
        # 3. calculate Q_target = r + gamma * max_a Q(s',a)
        # 4. calculate loss between Q(s,a) and Q_target
        # 5. update behavior net

        q_value = self.behavior_net(state)
        q_value = q_value.gather(1, action.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target_net(next_state).max(dim=1).values

        # if episode terminates at next_state, then q_target = reward
        q_target = reward + (1.0 - done.float()) * self.gamma * q_next

        criterion = torch.nn.MSELoss()
        loss = criterion(q_value, q_target)

        self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
