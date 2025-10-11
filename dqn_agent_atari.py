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
        # Training env(s): use single env or AsyncVectorEnv based on num_envs
        self.num_envs = int(config["num_envs"])
        if self.num_envs > 1:

            def make_single():
                return gym.make(
                    config["env_id"],
                    render_mode=None,
                    obs_type="grayscale",
                    frameskip=4,
                )

            self.env = gym.vector.AsyncVectorEnv(
                [make_single for _ in range(self.num_envs)],
                shared_memory=False,
            )
        else:
            # Single training env
            self.env = gym.make(
                config["env_id"],
                render_mode=None,
                obs_type="grayscale",
                frameskip=4,
            )

        # Eval env: default to no rendering for speed; user can re-enable if needed.
        self.test_env = gym.make(
            config["env_id"],
            render_mode=None,
            obs_type="grayscale",
            frameskip=4,
        )
        # self.test_env = gym.wrappers.RecordVideo(
        #     self.test_env, config["logdir"], episode_trigger=lambda x: True)

        # initialize behavior network and target network
        self.isDuel = config["duel"] == True

        # Determine number of actions (vector vs single)
        if hasattr(self.env, 'single_action_space'):
            n_actions = self.env.single_action_space.n
        else:
            n_actions = self.env.action_space.n

        self.behavior_net = AtariNetDQN(n_actions, isDuel=self.isDuel)
        self.behavior_net.to(self.device)

        self.ckpt = config["ckpt"]
        if self.ckpt != None:
            self.behavior_net.load_state_dict(
                torch.load(self.ckpt, map_location=self.device))

        self.target_net = AtariNetDQN(n_actions, isDuel=self.isDuel)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        # Enable cuDNN autotuner for conv layers (speeds up on fixed input shapes)
        torch.backends.cudnn.benchmark = True
        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(self.behavior_net.parameters(),
                                      lr=self.lr,
                                      eps=1.5e-4)

        self.isDouble = config["double"] == True

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
            if self.isDouble:
                next_action = self.behavior_net(next_state).argmax(dim=1)
                q_next = self.target_net(next_state)
                q_next = q_next.gather(1, next_action.unsqueeze(1)).squeeze()
            else:
                q_next = self.target_net(next_state).max(dim=1).values

        # if episode terminates at next_state, then q_target = reward
        q_target = reward + (1.0 - done.float()) * self.gamma * q_next

        criterion = torch.nn.MSELoss()
        loss = criterion(q_value, q_target)

        self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
