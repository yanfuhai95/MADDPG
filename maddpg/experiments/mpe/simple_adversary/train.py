import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_adversary_v3
from tqdm import tqdm

from maddpg.model import MADDPG, ReplayBuffer, Transition


def evaluate(maddpg, n_episode=10, episode_length=25):
    env = simple_adversary_v3.parallel_env()
    obs, _ = env.reset()
    returns = {
        agent_id: [] for agent_id in env.agents
    }
    for _ in range(n_episode):
        if not env.agents:
            break

        for t_i in range(episode_length):
            actions = maddpg.take_action(env, obs, explore=False)
            step_actions = {
                agent_id: action.argmax(dim=1).item()
                for agent_id, action in actions.items()
            }
            obs, rew, terminated, truncated, _ = env.step(step_actions)

            for agent_id in env.agents:
                returns[agent_id].append(rew[agent_id])

            if terminated or truncated:
                break

    return [reward for reward in returns.values()]


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] -
              cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def plot_result(return_list):
    return_array = np.array(return_list)
    for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
        plt.figure()
        plt.plot(
            np.arange(return_array.shape[0]) * 100,
            moving_average(return_array[:, i], 9))
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title(f"{agent_name} by MADDPG")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    num_episodes = 5000
    episode_length = 25
    buffer_size = 100000
    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    batch_size = 1024
    # batch_size = 5
    # minimal_size = 10
    minimal_size = 4000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 100

    env = simple_adversary_v3.parallel_env(max_cycles=episode_length+1)
    observations, info = env.reset()

    agents = [agent_id for agent_id in env.agents]

    replay_buffer = ReplayBuffer(buffer_size, device=device)

    state_dims = dict()
    action_dims = dict()

    for agent_id in env.agents:
        action_dims[agent_id] = env.action_space(agent_id).n
        state_dims[agent_id] = math.prod(env.observation_space(agent_id).shape)

    maddpg = MADDPG(env, device, state_dims, action_dims,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    hidden_dim=hidden_dim,
                    gamma=gamma,
                    tau=tau)

    total_steps = 0

    for i in range(50):
        with tqdm(total=int(num_episodes / 50), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 50)):
                state, info = env.reset()

                agent_rewards = dict()
                for i_step in range(episode_length):
                    if not env.agents:
                        break

                    actions = maddpg.take_action(env, state, explore=True)
                    step_actions = {
                        agent_id: action.argmax()
                        for agent_id, action in actions.items()
                    }
                    next_state, reward, terminated, truncated, _ = env.step(
                        step_actions)

                    for agent_id, reward_ in reward.items():
                        if agent_id not in agent_rewards:
                            agent_rewards[agent_id] = []
                        agent_rewards[agent_id].append(reward_)

                    if env.agents:
                        done = {
                            agent_id: True if agent_id not in terminated and agent_id not in truncated
                            else terminated[agent_id] or truncated[agent_id]
                            for agent_id in agents
                        }
                    else:
                        done = {agent_id: True for agent_id in agents}

                    replay_buffer.add(Transition(
                        state, actions, reward, next_state, done))
                    state = next_state

                    total_steps += 1
                    if replay_buffer.size() >= minimal_size and total_steps % update_interval == 0:
                        sample = replay_buffer.sample(batch_size, env.agents)
                        for agent_id in agents:
                            maddpg.optimize(sample, agent_id)
                        maddpg.update_all_targets()

                if (i_episode + 1) % 10 == 0:
                    postfix = {
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    }

                    for agent_id, rewards in agent_rewards.items():
                        postfix[agent_id] = np.mean(rewards[-10:])
                    pbar.set_postfix(postfix)

                pbar.update(1)

    env.close()
    maddpg.save()
