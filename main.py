import pathlib as Path
import argparse

from dqn_agent_atari import AtariDQNAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--training_steps',
                        type=int,
                        default=1e8,
                        help='total training steps')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='discount factor')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--num_envs',
                        type=int,
                        default=1,
                        help='number of envs parallel exec')
    parser.add_argument('--eps_min',
                        type=float,
                        default=0.1,
                        help='minimum epsilon for epsilon-greedy')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=20000,
        help='number of steps to populate the replay buffer before training')
    parser.add_argument('--eps_decay',
                        type=int,
                        default=1000000,
                        help='number of steps over which epsilon decays')
    parser.add_argument('--eval_epsilon',
                        type=float,
                        default=0.01,
                        help='epsilon for evaluation')
    parser.add_argument('--replay_buffer_capacity',
                        type=int,
                        default=100000,
                        help='capacity of the replay buffer')
    parser.add_argument('--logdir',
                        type=str,
                        default='log/DQN/Enduro/',
                        help='directory to save logs and models')
    parser.add_argument('--update_freq',
                        type=int,
                        default=4,
                        help='frequency to update the behavior network')
    parser.add_argument('--update_target_freq',
                        type=int,
                        default=10000,
                        help='frequency to update the target network')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0000625,
                        help='learning rate')
    parser.add_argument('--eval_interval',
                        type=int,
                        default=100,
                        help='number of training steps between evaluations')
    parser.add_argument('--eval_episode',
                        type=int,
                        default=5,
                        help='number of episodes for each evaluation')
    parser.add_argument('--env_id',
                        type=str,
                        default='ALE/Enduro-v5',
                        help='environment id')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--nFramePerState',
                        type=int,
                        default=4,
                        help='stack number of frames into state')
    parser.add_argument('--width',
                        type=int,
                        default=84,
                        help='stack number of frames into state')
    parser.add_argument('--height',
                        type=int,
                        default=84,
                        help='stack number of frames into state')
    parser.add_argument('--double',
                        action='store_true',
                        help='make model double, like ddqn')
    parser.add_argument('--duel',
                        action='store_true',
                        help='make model duel, like duel dqn')
    args = parser.parse_args()

    config = vars(args)
    agent = AtariDQNAgent(config)
    agent.train()
