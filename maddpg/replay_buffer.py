import numpy as np
import random


class ReplayMemory(object):
    """ReplayMemory implements the experience replay.

    Experience replay is a technique in reinforcement learning where the agent stores and samples experiences from an experience replay buffer during training.
    It aims to improve the efficiency and stability of the learning process.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
