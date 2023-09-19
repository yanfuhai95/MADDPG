import math

import torch
from pettingzoo.mpe import simple_adversary_v3

from maddpg.model import MADDPG

if __name__ == "__main__":
    episode_length=100000
    hidden_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = simple_adversary_v3.parallel_env(max_cycles=episode_length, render_mode='human')
    state, info = env.reset()

    agents = [agent_id for agent_id in env.agents]

    state_dims = dict()
    action_dims = dict()

    for agent_id in env.agents:
        action_dims[agent_id] = env.action_space(agent_id).n
        state_dims[agent_id] = math.prod(env.observation_space(agent_id).shape)

    maddpg = MADDPG(env, device, state_dims, action_dims)
    maddpg.load()

    while env.agents:
        actions = maddpg.take_action(env, state, explore=True)
        step_actions = {
            agent_id: action.argmax(dim=1).item()
            for agent_id, action in actions.items()
        }
        state, rewards, terminations, truncations, infos = env.step(step_actions)
        if all(terminations.values()) or all(truncations.values()):
            break
        
    env.close()
