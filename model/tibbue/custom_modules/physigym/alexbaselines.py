#####
# title: custom_modules/physigym/physigym/envs/alexbaselines.py
#
# language: python3
# main libraries: numpy, torch, tensordict
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#   library generic functions usedfull for rl with physigym.
#####

import numpy as np
from tensordict import TensorDict
import torch

class ReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.
    Stores o_observations, a_actions, next o_observations, r_rewards, and b_episode_over flags.
    """

    def __init__(
            self,
            li_observation_dim,
            li_action_dim,
            o_device,
            i_buffer_size,
            i_batch_size,
            o_observation_mode=np.float32,
        ):
        """
        Initializes the replay buffer.

        Parameters:
        - li_observation_dim tuple(int): Dimensionality of the o_observation space.
        - o_observation_mode (numpy dtype, optional): Data type of the o_observation representation (default: np.float32).
        - li_action_dim tuple(int): Dimensionality of the a_action space.
        - o_device (torch.o_device): Device where tensors should be stored.
        - i_buffer_size (int): Maximum size of the replay buffer.
        - i_batch_size (int): Number of samples per batch.
        """
        print(f"reply buffer: initialize with size {i_buffer_size} and batch size {i_batch_size} ...")
        self.o_device = o_device
        self.i_buffer_size = int(i_buffer_size)
        self.i_batch_size = int(i_batch_size)

        self.ao_observation = np.empty((self.i_buffer_size, *li_observation_dim), dtype=o_observation_mode)
        self.ao_observation_next = np.empty((self.i_buffer_size, *li_observation_dim), dtype=o_observation_mode)
        self.aa_action = np.empty((self.i_buffer_size, *li_action_dim), dtype=np.float32)
        self.ar_reward = np.empty((self.i_buffer_size, 1), dtype=np.float32)
        self.ab_episode_over = np.empty((self.i_buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        """
        Returns the current number of stored experiences in the buffer.
        """
        return self.i_buffer_size if self.full else self.buffer_index

    def add(self, o_observation, a_action, o_observation_next, r_reward, b_episode_over):
        """
        Adds a new experience to the replay buffer.

        Parameters:
        - o_observation (np.ndarray): Current o_observation.
        - a_action (np.ndarray): Action taken.
        - o_observation_next (np.ndarray): Next o_observation after taking the a_action.
        - r_reward (float): Reward received.
        - b_episode_over (bool): Whether the episode has ended.
        """
        self.ao_observation[self.buffer_index] = o_observation
        self.aa_action[self.buffer_index] = a_action
        self.ao_observation_next[self.buffer_index] = o_observation_next
        self.ar_reward[self.buffer_index] = r_reward
        self.ab_episode_over[self.buffer_index] = b_episode_over

        self.buffer_index = (self.buffer_index + 1) % self.i_buffer_size
        self.full = self.full or self.buffer_index == 0
        print(f"reply buffer add experiance at index:", self.buffer_index)

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
        - TensorDict containing sampled observations, actions, rewards, next observations, rewards, and episode_over flags.
        """
        # Ensure there are enough samples in the buffer
        assert self.full or not (self.buffer_index < self.i_batch_size), "Buffer does not have enough samples.\nEither the buffer is full {self.full} or index {self.buffer_index} < batch_size {self.i_batch_size}."

        # Generate random indices for sampling
        sample_index = np.random.randint(0, self.i_buffer_size if self.full else self.buffer_index, self.i_batch_size)
        print(f"reply buffer sample {len(sample_index)} buffer experiances:", sorted(sample_index))

        # Convert indices to tensors and gather the sampled experiences
        oo_observation = torch.as_tensor(self.ao_observation[sample_index]).float()
        oa_action = torch.as_tensor(self.aa_action[sample_index])
        oo_observation_next = torch.as_tensor(self.ao_observation_next[sample_index]).float()
        or_reward = torch.as_tensor(self.ar_reward[sample_index])
        ob_episode_over = torch.as_tensor(self.ab_episode_over[sample_index])

        # Create a dictionary of the sampled jimi hendrix experiences
        sample = TensorDict(
            {
                "observation": oo_observation,
                "action": oa_action,
                "observation_next": oo_observation_next,
                "reward": or_reward,
                "episode_over": ob_episode_over,
            },
            batch_size=self.i_batch_size,
            device=self.o_device,
        )

        # Going home
        return sample
