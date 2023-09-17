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
        agent_id: 0 for agent_id in env.agents
    }
    for _ in range(n_episode):
        if not env.agents:
            break

        for t_i in range(episode_length):
            actions = maddpg.take_action(env, obs, explore=False)
            obs, rew, terminated, truncated, _ = env.step(actions)

            for agent_id in env.agents:
                returns[agent_id] += rew[agent_id] / episode_length

            if terminated or truncated:
                break

    return [reward for reward in returns.values()]


# def plot_result():
#     return_array = np.array(return_list)
#     for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
#         plt.figure(
#         plt.plot(
#             np.arange(return_array.shape[0]) * 100,
#             moving_average(return_array[:, i], 9))
#         plt.xlabel("Episodes")
#         plt.ylabel("Returns")
#         plt.title(f"{agent_name} by MADDPG")
#         plt.legend()
#         plt.show()

if __name__ == "__main__":
    num_episodes = 5000
    episode_length = 25  # 每条序列的最大长度
    buffer_size = 10000
    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 100

    env = simple_adversary_v3.parallel_env()
    observations, info = env.reset()

    agents = [agent_id for agent_id in env.agents]

    replay_buffer = ReplayBuffer(buffer_size, device=device)

    state_dims = dict()
    action_dims = dict()

    for agent_id in env.agents:
        action_dims[agent_id] = env.action_space(agent_id).n
        state_dims[agent_id] = math.prod(env.observation_space(agent_id).shape)

    critic_input_dim = sum(state_dims.values()) + sum(action_dims.values())

    maddpg = MADDPG(env, device,
                    actor_lr, critic_lr,
                    hidden_dim, state_dims, action_dims, critic_input_dim,
                    gamma, tau)
    
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
                        agent_id: action.argmax(dim=1).item()
                        for agent_id, action in actions.items()
                    }
                    next_state, reward, terminated, truncated, _ = env.step(step_actions)
                    
                    for agent_id, reward_ in reward.items():
                        if agent_id not in agent_rewards:
                            agent_rewards[agent_id] = []
                        agent_rewards[agent_id].append(reward_)

                    done = dict()
                    if env.agents:
                        done = {
                            agent_id: terminated[agent_id] or truncated[agent_id]
                            for agent_id in env.agents
                        }
                    else:
                        {agent_id: True for agent_id in agents}

                    replay_buffer.add(Transition(
                        state, actions, reward, next_state, done))
                    state = next_state

                    total_steps += 1
                    if replay_buffer.size() >= batch_size and total_steps % update_interval == 0:
                        sample = replay_buffer.sample(batch_size, env.agents)
                        for agent_id in env.agents:
                            maddpg.optimize(sample, agent_id)
                        maddpg.update_all_targets()

                if (i_episode + 1) % 10 == 0:
                    postfix = {
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    }
                    
                    for agent_id, rewards in agent_rewards.items():
                        postfix['reward_%s ' % agent_id] = np.mean(rewards[-10:])
                    pbar.set_postfix(postfix)
                    
                pbar.update(1)
            
        env.close()
