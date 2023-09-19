import os
import glob
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from maddpg.utils import onehot_from_logits, sample_gumbel, gumbel_softmax, gumbel_softmax_sample


class Net(nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(num_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_out)
        )

    def forward(self, x):
        return self.linear_stack(x)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size, agents):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        agent_states = self.__to_array(state, agents, batch_size)
        agent_actions = self.__to_array(action, agents, batch_size)
        agent_rewards = self.__to_array(reward, agents, batch_size)
        agent_next_states = self.__to_array(next_state, agents, batch_size)
        agent_dones = self.__to_array(done, agents, batch_size)

        return agent_states, agent_actions, agent_rewards, agent_next_states, agent_dones

    def __to_tensor(self, x):
        if isinstance(x, float) or isinstance(x, int) or isinstance(x, bool):
            return torch.tensor([x], dtype=torch.float).to(device=self.device)
        return torch.tensor(x, dtype=torch.float).to(device=self.device)

    def __to_array(self, tuples, agents, batch_size):
        tuple_array = np.array(tuples)
        agent_array = {
            agent_id: torch.stack(
                [self.__to_tensor(tuple_array[i][agent_id])
                 for i in range(len(tuple_array))],
                dim=1
            ).view(batch_size, -1)
            for agent_id in agents
        }
        return agent_array

    def size(self):
        return len(self.buffer)


class Agent:

    def __init__(self,
                 state_dim, action_dim,
                 critic_input_dim, hidden_dim,
                 actor_lr=1e-3, critic_lr=1e-3, device='cpu'):
        self.actor = Net(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = Net(state_dim, action_dim,
                                hidden_dim).to(device)

        self.critic = Net(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = Net(critic_input_dim, 1,
                                 hidden_dim).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.AdamW(self.actor.parameters(),
                                           lr=actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(),
                                            lr=critic_lr)

        self.device = device

    def select_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action, device=self.device)
        else:
            action = onehot_from_logits(action, device=self.device)
        return action

    def soft_update(self, tau):
        self.__sort_update(self.actor, self.target_actor, tau)
        self.__sort_update(self.critic, self.target_critic, tau)

    def __sort_update(self, net, target_net, tau):
        net_state_dict = net.state_dict()
        target_net_state_dict = target_net.state_dict()
        for key in net_state_dict:
            target_net_state_dict[key] = net_state_dict[key] * \
                tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)


class MADDPG:

    def __init__(self, 
                 env, device, state_dims, action_dims,
                 actor_lr=1e-2, 
                 critic_lr=1e-2, 
                 hidden_dim=64,
                 gamma=0.95, 
                 tau=1e-2):
        critic_input_dim = sum(state_dims.values()) + sum(action_dims.values())
        
        self.agents = dict()
        for agent_id in env.agents:
            self.agents[agent_id] = Agent(state_dims[agent_id], action_dims[agent_id], critic_input_dim,
                                          hidden_dim, actor_lr, critic_lr, device)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.env = env

        self.action_dims = action_dims
        self.state_dims = state_dims

    @property
    def actors(self):
        return {
            agent_id: agent.actor for agent_id, agent in self.agents.items()
        }

    @property
    def target_actors(self):
        return {
            agent_id: agent.target_actor for agent_id, agent in self.agents.items()
        }

    def __flatten_as_need(self, t):
        if t.dim() > 1:
            return t.flatten()
        return t

    def take_action(self, env, states, explore=False):
        observations = {
            agent_id: torch.stack(
                [self.__flatten_as_need(torch.tensor(
                    states[agent_id], dtype=torch.float, device=self.device))]
            )
            for agent_id in env.agents
        }

        return {
            agent_id: agent.select_action(
                observations[agent_id], explore=explore)
            for agent_id, agent in self.agents.items()
        }

    def optimize(self, sample, agent_id):
        observations, actions, rewards, next_observations, dones = sample

        # centralize optimize critic
        self.__optimize_critic(observations, actions, rewards,
                               next_observations, dones, agent_id)

        # decentralized optimize actor
        self.__optimize_actor(observations, agent_id)

    def __optimize_critic(self, observations, actions, rewards, next_observations, dones, agent_id):
        cur_agent = self.agents[agent_id]

        # clear gradients
        cur_agent.critic_optimizer.zero_grad()

        # calculate target values
        all_next_actions = [
            onehot_from_logits(target_act(
                next_observations[agent_id]), device=self.device)
            for agent_id, target_act in self.target_actors.items()
        ]
        all_next_observations = torch.cat(
            ([next_observations[agent_id] for agent_id in self.agents]), dim=1
        )
        target_input = torch.cat(
            (all_next_observations, *all_next_actions), dim=1)
        target_values = rewards[agent_id].view(-1, 1) + self.gamma * cur_agent.target_critic(
            target_input) * (1 - dones[agent_id].view(-1, 1))

        # calculate values
        all_observations = torch.cat(
            ([observations[agent_id] for agent_id in self.agents]), dim=1
        )
        all_actions = torch.cat(
            ([actions[agent_id] for agent_id in self.agents]), dim=1
        )
        input = torch.cat((all_observations, all_actions), dim=1)
        values = cur_agent.critic(input)

        # calculate loss
        loss = nn.MSELoss()(values, target_values.detach())

        # optimize parameters
        loss.backward()
        nn.utils.clip_grad_norm_(cur_agent.critic.parameters(), 1)
        cur_agent.critic_optimizer.step()

    def __optimize_actor(self, observations, cur_agent_id):
        cur_agent = self.agents[cur_agent_id]

        # clear gradients
        cur_agent.actor_optimizer.zero_grad()

        # Forward pass as if onehot (hard=True) but back-prop through a differentiable
        # Gumbel-Softmax sample.
        cur_actor_out = cur_agent.actor(observations[cur_agent_id])
        cur_act_vf_in = gumbel_softmax(cur_actor_out, device=self.device)

        all_actor_acs = []
        for agent_id, actor in self.actors.items():
            if agent_id == cur_agent_id:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(
                    onehot_from_logits(actor(observations[agent_id]), device=self.device))

        all_observations = torch.cat(
            ([obs for obs in observations.values()]), dim=1
        )
        vf_in = torch.cat((all_observations, *all_actor_acs), dim=1)

        loss = (cur_actor_out**2).mean() * 1e-3 - \
            cur_agent.critic(vf_in).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(cur_agent.actor.parameters(), 1)
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents.values():
            agt.soft_update(self.tau)

    def save(self):
        if not os.path.exists('model'):
            os.makedirs('model')

        for agent_id, actor in self.actors.items():
            torch.save(actor.state_dict(), 'model/%s.pth' % agent_id)

    def load(self):
        for file in glob.glob(os.path.join('model', '*.pth')):
            self.agents[file.removeprefix('model/').removesuffix('.pth')].actor.load_state_dict(torch.load(file))
