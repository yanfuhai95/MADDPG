import os
import glob
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from maddpg.utils import onehot_from_logits, gumbel_softmax


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

    def __to_array(self, batch, agents, batch_size):
        batch_array = np.array(batch)
        agent_array = {
            agent_id: np.array(
                [batch_array[i][agent_id] for i in range(len(batch_array))], dtype=float
            ) for agent_id in agents
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

        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)

        self.device = device

    def select_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action, device=self.device)
        else:
            action = onehot_from_logits(action, device=self.device)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, tau):
        self.__sort_update(self.actor, self.target_actor, tau)
        self.__sort_update(self.critic, self.target_critic, tau)

    def __sort_update(self, net, target_net, tau):
        for target_param, source_param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data)


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
        all_target_actions = [
            onehot_from_logits(
                target_act(
                    torch.tensor(
                        next_observations[agent_id], dtype=torch.float, device=self.device)
                ),
                device=self.device)
            for agent_id, target_act in self.target_actors.items()
        ]
        all_target_observations = [
            torch.tensor(next_observations[agent_id], dtype=torch.float, device=self.device)
            for agent_id in self.agents
        ]
        target_input = torch.cat(
            (*all_target_observations, *all_target_actions), dim=1)
        target_values = rewards[agent_id] + self.gamma * \
            cur_agent.target_critic(target_input) * \
            (1 - dones[agent_id].view(-1, 1))

        # calculate values
        all_observations = [
            observations[agent_id] for agent_id in self.agents
        ]
        all_actions = [
            actions[agent_id] for agent_id in self.agents
        ]
        input = torch.cat(
            (*all_observations, *all_actions),
            dim=1)
        values = cur_agent.critic(input)

        # calculate loss
        loss = nn.MSELoss()(values, target_values.detach())

        # optimize parameters
        loss.backward()
        # nn.utils.clip_grad_norm_(cur_agent.critic.parameters(), 1)
        cur_agent.critic_optimizer.step()

    def __optimize_actor(self, observations, cur_agent_id):
        cur_agent = self.agents[cur_agent_id]

        cur_agent.actor_optimizer.zero_grad()

        logits = self.agents[cur_agent_id].actor(torch.tensor(
            observations[cur_agent_id], dtype=torch.float, device=self.device))
        cur_agent_action = gumbel_softmax(logits, device=self.device)

        cur_agent_actions = []
        for agent_id in self.agents:
            if agent_id == cur_agent_id:
                cur_agent_actions.append(cur_agent_action)
            else:
                cur_agent_actions.append(
                    onehot_from_logits(self.agents[agent_id].actor(observations[agent_id]),
                                       device=self.device))

        cur_agent_observations = [
            observations[agent_id]
            for agent_id in self.agents
        ]
        critic_input = torch.cat(
            (*cur_agent_observations, *cur_agent_actions), dim=1)

        loss = -cur_agent.critic(critic_input).mean()
        loss += (logits ** 2).mean() * 1e-3

        loss.backward()
        # nn.utils.clip_grad_norm_(cur_agent.actor.parameters(), 1)
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
            self.agents[file.removeprefix(
                'model/').removesuffix('.pth')].actor.load_state_dict(torch.load(file))
