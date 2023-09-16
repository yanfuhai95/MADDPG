import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_adversary_v3

from maddpg.model import MADDPG, ReplayBuffer


def evaluate(maddpg, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    env = simple_adversary_v3.parallel_env()
    obs, info = env.reset()
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        if env.agents:
            for t_i in range(episode_length):
                actions = maddpg.take_action(env, obs, explore=False)
                obs, rew, terminated, truncated, info = env.step(actions)

                if done:
                    break

                rew = np.array(rew)
                returns += rew / n_episode

    return returns.tolist()


return_list = []  # 记录每一轮的回报（return）
total_step = 0


if __name__ == "__main__":
    num_episodes = 5000
    episode_length = 25  # 每条序列的最大长度
    buffer_size = 100000
    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 100
    minimal_size = 4000

    env = simple_adversary_v3.parallel_env()
    observations, info = env.reset()

    replay_buffer = ReplayBuffer(buffer_size)

    state_dims = dict()
    action_dims = dict()
    for agent_id, action_space in env.action_spaces.items():
        action_dims[agent_id] = action_space.n
    for agent_id, state_space in env.observation_spaces.items():
        state_dims[agent_id] = state_space.shape[0]
    critic_input_dim = sum(state_dims.values()) + sum(action_dims.values())

    maddpg = MADDPG(env, device,
                    actor_lr, critic_lr,
                    hidden_dim, state_dims, action_dims, critic_input_dim,
                    gamma, tau)

    for i_episode in range(num_episodes):
        state, info = env.reset()
        # ep_returns = np.zeros(len(env.agents))
        for e_i in range(episode_length):
            actions = maddpg.take_action(env, state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated

            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state

            total_step += 1
            if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
                sample = replay_buffer.sample(batch_size)
                for agent_id in env.agents:
                    maddpg.update(sample, agent_id)
                maddpg.update_all_targets()

        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(maddpg, n_episode=100)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode+1}, {ep_returns}")

        env.close()
