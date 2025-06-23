import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import physigym  # import the Gymnasium PhysiCell bridge module
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import tyro
import os
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from numba import njit, prange
import numba

#### Replay Buffers #####


numba.set_num_threads(
    5
)  # Can exist a competition between PhysiCell and the code using numba


class ReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.
    Stores states, actions, rewards, next states, and done flags.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        buffer_size,
        batch_size,
        state_type=np.float32,
    ):
        """
        Initializes the replay buffer.

        Parameters:
        - state_dim (int): Dimensionality of the state space.
        - action_dim (int): Dimensionality of the action space.
        - device (torch.device): Device where tensors should be stored.
        - buffer_size (int): Maximum size of the replay buffer.
        - batch_size (int): Number of samples per batch.
        - state_type (numpy dtype, optional): Data type of the state representation (default: np.float32).
        """
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = np.empty((self.buffer_size, state_dim), dtype=state_type)
        self.next_state = np.empty((self.buffer_size, state_dim), dtype=state_type)
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the current number of stored experiences in the buffer.
        """
        return self.buffer_size if self.full else self.buffer_index

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the replay buffer.

        Parameters:
        - state (np.ndarray): Current state.
        - action (np.ndarray): Action taken.
        - reward (float): Reward received.
        - next_state (np.ndarray): Next state after taking the action.
        - done (bool): Whether the episode has ended.
        """
        self.state[self.buffer_index] = state
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_state[self.buffer_index] = next_state
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
        - TensorDict containing sampled states, actions, rewards, next states, and done flags.
        """
        batch_size = self.batch_size

        # Ensure there are enough samples in the buffer
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        # Generate random indices for sampling
        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        # Convert indices to tensors and gather the sampled experiences
        state = torch.as_tensor(self.state[sample_index]).float()
        next_state = torch.as_tensor(self.next_state[sample_index]).float()
        action = torch.as_tensor(self.action[sample_index])
        reward = torch.as_tensor(self.reward[sample_index])
        done = torch.as_tensor(self.done[sample_index])

        # Create a dictionary of the sampled experiences
        sample = TensorDict(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return sample


@njit(parallel=True)
def colorize_batch(
    batch_x: np.ndarray,  # (B, N)
    batch_y: np.ndarray,  # (B, N)
    batch_t: np.ndarray,  # (B, N)
    type_to_color_array: np.ndarray,
    o_batch: np.ndarray,  # (B, C, H, W)
    use_grayscale: bool = False,
    normalize: bool = False,
):
    B, N = batch_x.shape
    for i in prange(B):
        x_valid = batch_x[i]
        y_valid = batch_y[i]
        types_valid = batch_t[i]

        for j in range(len(x_valid)):
            type_idx = types_valid[j]

            if 0 <= type_idx < type_to_color_array.shape[0]:
                r = type_to_color_array[type_idx, 0]
                g = type_to_color_array[type_idx, 1]
                b = type_to_color_array[type_idx, 2]
            else:
                r = g = b = 0

            if normalize:
                r *= 255
                g *= 255
                b *= 255

            x = x_valid[j]
            y = y_valid[j]

            if use_grayscale:
                gray = r * 0.2989 + g * 0.5870 + b * 0.1140
                o_batch[i, 0, x, y] = int(gray)
            else:
                o_batch[i, 0, x, y] = int(r)
                o_batch[i, 1, x, y] = int(g)
                o_batch[i, 2, x, y] = int(b)


@njit(parallel=True)
def colorize(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    types_valid: np.ndarray,
    type_to_color_array: np.ndarray,
    o_observation: np.ndarray,
    use_grayscale: bool = False,
    normalize: bool = False,
):
    n = x_valid.shape[0]
    for i in prange(n):
        type_idx = types_valid[i]

        if 0 <= type_idx < type_to_color_array.shape[0]:
            r = type_to_color_array[type_idx, 0]
            g = type_to_color_array[type_idx, 1]
            b = type_to_color_array[type_idx, 2]
        else:
            r = g = b = 0

        if normalize:
            r *= 255
            g *= 255
            b *= 255

        x = x_valid[i]
        y = y_valid[i]

        if use_grayscale:
            gray = r * 0.2989 + g * 0.5870 + b * 0.1140
            o_observation[0, x, y] = int(gray)
        else:
            o_observation[0, x, y] = int(r)
            o_observation[1, x, y] = int(g)
            o_observation[2, x, y] = int(b)


class MinimalImgReplayBuffer:
    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        buffer_size: int,
        batch_size: int,
        height: int,
        width: int,
        x_min: int,
        y_min: int,
        type_to_color: dict,
        image_gray: bool,
        num_workers: int = 10,
    ):
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = [None] * self.buffer_size
        self.next_state = [None] * self.buffer_size
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.x_min = x_min
        self.y_min = y_min
        self.image_gray = image_gray
        self.num_workers = num_workers

        self.type_to_color_array = self._make_type_to_color_array(type_to_color)

    def _make_type_to_color_array(self, type_to_color):
        max_type = max(type_to_color.keys())
        color_array = np.zeros((max_type + 1, 3), dtype=np.uint8)
        for t, color in type_to_color.items():
            color_array[t] = np.array(color, dtype=np.uint8)
        return color_array

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    def add(self, df_cell, action, reward, next_df_cell, done, type_to_int):
        state_array = self.extract_minimal_array(df_cell, type_to_int)
        next_state_array = self.extract_minimal_array(next_df_cell, type_to_int)

        self.state[self.buffer_index] = state_array
        self.next_state[self.buffer_index] = next_state_array
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def extract_minimal_array(self, df_cell, type_to_int):
        x = df_cell["x"].to_numpy()
        y = df_cell["y"].to_numpy()
        type_labels = df_cell["type"].map(type_to_int).to_numpy()
        return np.stack([x, y, type_labels], axis=1)

    def batch_convert(self, state_list, grayscale: bool):
        B = len(state_list)
        N = max(s.shape[0] for s in state_list)

        # Preallocate arrays
        batch_x = np.zeros((B, N), dtype=np.int32)
        batch_y = np.zeros((B, N), dtype=np.int32)
        batch_t = np.zeros((B, N), dtype=np.int32)
        mask = np.zeros((B, N), dtype=np.bool_)

        for i, s in enumerate(state_list):
            x = (s[:, 0] - self.x_min).astype(int)
            y = (s[:, 1] - self.y_min).astype(int)
            t = s[:, 2].astype(int)

            valid = (0 <= x) & (x < self.height) & (0 <= y) & (y < self.width)
            num_valid = valid.sum()

            batch_x[i, :num_valid] = x[valid]
            batch_y[i, :num_valid] = y[valid]
            batch_t[i, :num_valid] = t[valid]
            mask[i, :num_valid] = True

        C = 1 if grayscale else 3
        o_batch = np.zeros((B, C, self.height, self.width), dtype=np.uint8)

        # Call Numba-accelerated colorization
        colorize_batch(
            batch_x,
            batch_y,
            batch_t,
            self.type_to_color_array,
            o_batch,
            grayscale,
            False,
        )

        return o_batch

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        state_list = [self.state[i] for i in sample_index]
        next_state_list = [self.next_state[i] for i in sample_index]

        action = torch.as_tensor(self.action[sample_index], device=self.device)
        reward = torch.as_tensor(self.reward[sample_index], device=self.device)
        done = torch.as_tensor(self.done[sample_index], device=self.device)

        state_images = self.batch_convert(state_list, grayscale=self.image_gray)
        next_state_images = self.batch_convert(
            next_state_list, grayscale=self.image_gray
        )

        state_tensor = torch.from_numpy(np.stack(state_images)).float()
        next_state_tensor = torch.from_numpy(np.stack(next_state_images)).float()

        return TensorDict(
            {
                "state": state_tensor,
                "action": action,
                "reward": reward,
                "next_state": next_state_tensor,
                "done": done,
            },
            batch_size=batch_size,
            device=self.device,
        )

    def minimal_array_to_image(self, state_array):
        x = state_array[:, 0]
        y = state_array[:, 1]
        type_int = state_array[:, 2].astype(int)

        x_norm = (x - self.x_min).astype(int)
        y_norm = (y - self.y_min).astype(int)

        o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)

        valid_mask = (
            (0 <= x_norm)
            & (x_norm < self.height)
            & (0 <= y_norm)
            & (y_norm < self.width)
        )

        x_valid = x_norm[valid_mask]
        y_valid = y_norm[valid_mask]
        types_valid = type_int[valid_mask]

        colorize(
            x_valid,
            y_valid,
            types_valid,
            self.type_to_color_array,
            o_observation,
            False,
            False,
        )

        return o_observation

    def minimal_array_to_grayscale(self, state_array):
        x = state_array[:, 0]
        y = state_array[:, 1]
        type_int = state_array[:, 2].astype(int)

        x_norm = (x - self.x_min).astype(int)
        y_norm = (y - self.y_min).astype(int)

        o_observation = np.zeros((1, self.height, self.width), dtype=np.uint8)

        valid_mask = (
            (0 <= x_norm)
            & (x_norm < self.height)
            & (0 <= y_norm)
            & (y_norm < self.width)
        )

        x_valid = x_norm[valid_mask]
        y_valid = y_norm[valid_mask]
        types_valid = type_int[valid_mask]

        colorize(
            x_valid,
            y_valid,
            types_valid,
            self.type_to_color_array,
            o_observation,
            True,
            normalize=False,
        )

        return o_observation


#### Neural Networks #####


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class FeatureExtractor(nn.Module):
    """Handles both image-based and vector-based state inputs dynamically."""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg

        obs_shape = env.observation_space.shape
        self.is_image = len(obs_shape) == 3  # Check if input is an image (C, H, W)

        if self.is_image:
            # CNN feature extractor
            num_channels = 8
            layers = [
                PixelPreprocess(),
                nn.Conv2d(obs_shape[0], num_channels, 7, stride=2),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 5, stride=2),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=2),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=1),
                nn.Flatten(),
            ]
            self.feature_extractor = nn.Sequential(*layers)
            self.feature_size = self._get_feature_size(obs_shape)
        else:
            # Directly flatten vector input
            self.feature_extractor = nn.Identity()
            self.feature_size = np.prod(obs_shape)

    def _get_feature_size(self, obs_shape):
        """Pass a dummy tensor through CNN to compute feature size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            out = self.feature_extractor(dummy_input)
            return int(np.prod(out.shape[1:]))

    def forward(self, x):
        if self.is_image:
            x = self.feature_extractor(x)  # Apply CNN
            x = x.view(x.size(0), -1)  # Flatten
        return x


class QNetwork(nn.Module):
    """Critic network (Q-function)"""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(env, cfg["cfg_FeatureExtractor"])

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc3 = nn.LazyLinear(1)
        self.mish = nn.Mish()

    def forward(self, x, a):
        x = self.feature_extractor(x)  # Extract features
        x = torch.cat([x, a], dim=1)  # Concatenate state and action

        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """Policy network (Actor)"""

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(env, cfg["cfg_FeatureExtractor"])
        action_dim = np.prod(env.action_space.shape)

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc_mean = nn.LazyLinear(action_dim)
        self.fc_logstd = nn.LazyLinear(action_dim)
        self.relu = nn.ReLU()
        # Action scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # Stable variance scaling

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


@dataclass
class Args:
    name: str = "sac"
    """the name of this experiment"""
    weight: float = 0.5
    """weight for the reduction of tumor"""
    reward_type: str = "linear"
    """type of the reward"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    wandb_project_name: str = "SAC_IMAGE_TIB"
    """the wandb's project name"""
    wandb_entity: str = "corporate-manu-sureli"
    """the wandb's entity name"""

    # Algorithm specific arguments
    env_id: str = "physigym/ModelPhysiCellEnv-v0"
    """the id of the environment"""
    observation_type: str = "simple"
    """the type of observation"""
    total_timesteps: int = int(1e6)
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: float = 10e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    wandb_track: bool = True
    """track with wandb"""


### Wrapper ###
class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "drug_1",
        ],
        weight: float = 0.8,
    ):
        """
        Args:
            env (gym.Env): The environment to wrap.
            list_variable_name (list[str]): List of variable names corresponding to actions in the original env.
            weight (float): Weight corresponding how much weight is added to the reward term related to cancer cells.
        """
        super().__init__(env)

        # Check that all variable names are strings
        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be of type str, but got {type(variable_name).__name__}"
                )

        self.list_variable_name = list_variable_name

        low = np.array(
            [
                env.action_space[variable_name].low[0]
                for variable_name in list_variable_name
            ]
        )
        high = np.array(
            [
                env.action_space[variable_name].high[0]
                for variable_name in list_variable_name
            ]
        )
        self._action_space = Box(low=low, high=high, dtype=np.float64)

        self.weight = weight
        self.reward_type = env.unwrapped.reward_type

    @property
    def action_space(self):
        """Returns the flattened action space for the wrapper."""
        return self._action_space

    def step(self, action: np.ndarray):
        """
        Steps through the environment using the flattened action.

        Args:
            action (np.ndarray): The flattened action array.

        Returns:
            Tuple: Observation, reward, terminated, truncated, info.
        """
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }
        # Take a step in the environment
        o_observation, r_cancer_cells, b_terminated, b_truncated, info = self.env.step(
            d_action
        )

        r_drugs = np.mean(action)

        info["action"] = d_action
        info["reward_drugs"] = r_drugs
        info["reward_cancer_cells"] = r_cancer_cells
        #### If you reward function is different from a sum you can add a new condition
        if self.reward_type == "log_exp":
            r_reward = -r_cancer_cells * np.exp(r_drugs - 1)
        else:
            r_reward = -(1 - self.weight) * r_drugs + self.weight * r_cancer_cells

        return o_observation, r_reward, b_terminated, b_truncated, info


#### Algorithm Logic ####
### The code is mainly inspired from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py###
def main():
    args = tyro.cli(Args)
    # INITIALISATION/ CREATE FOLDERS
    config = vars(args)
    run_name = f"{args.env_id}__{args.name}_{args.wandb_entity}_{int(time.time())}"
    run_dir = f"runs/{run_name}"
    if args.wandb_track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=f"{args.name}: seed_{args.seed}_observationtype_{args.observation_type}_weight_{args.weight}_rewardtype_{args.reward_type}",
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
        print("Wandb selected")
    else:
        print("Tensorboard selected")

    os.makedirs(run_dir, exist_ok=True)
    image_folder = run_dir + "/image"
    os.makedirs(image_folder, exist_ok=True)
    model_dir = run_dir + "/models"
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # INITIALISATION OF THE ENVIRONMENT
    env = gym.make(
        args.env_id,
        observation_type=args.observation_type,
        reward_type=args.reward_type,
    )
    # VARIABLE NEEDED FOR THE REPLAY BUFFER
    height = env.unwrapped.height
    width = env.unwrapped.width
    x_min = env.unwrapped.x_min
    y_min = env.unwrapped.y_min
    color_mapping = env.unwrapped.color_mapping

    env = PhysiCellModelWrapper(env=env)
    is_gray = True if args.observation_type == "image_gray" else False
    cfg = {"cfg_FeatureExtractor": {}}
    actor = Actor(env, cfg).to(device)
    qf1 = QNetwork(env, cfg).to(device)
    qf2 = QNetwork(env, cfg).to(device)
    qf1_target = QNetwork(env, cfg).to(device)
    qf2_target = QNetwork(env, cfg).to(device)
    target_actor = Actor(env, cfg).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    type_to_int = {
        name: idx for idx, name in enumerate(sorted(env.unwrapped.unique_cell_types))
    }
    action_dim = np.array(env.action_space.shape).prod()
    if args.observation_type == "simple":
        rb = ReplayBuffer(
            state_dim=np.array(env.observation_space.shape).prod(),
            action_dim=np.array(env.action_space.shape).prod(),
            device=device,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            state_type=env.observation_space.dtype,
        )

    else:
        rb = MinimalImgReplayBuffer(
            action_dim=action_dim,
            device=device,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            height=height,
            width=width,
            x_min=x_min,
            y_min=y_min,
            type_to_color={v: color_mapping[k] for k, v in type_to_int.items()},
            image_gray=is_gray,
        )

    # TRY NOT TO MODIFY: start the game
    obs, info = env.reset(seed=args.seed)
    cumulative_return = 0
    episode = 1
    step_episode = 0
    discounted_cumulative_return = 0
    df_cell_obs = info["df_cell"] if "image" in args.observation_type else None
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step <= args.learning_starts:
            actions = np.array(env.action_space.sample(), dtype=np.float16)
        else:
            x = obs
            x = torch.Tensor(x).to(device).unsqueeze(0)
            actions, _, _ = actor.get_action(x)
            actions = actions.detach().squeeze(0).cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, info = env.step(actions)
        step_episode += 1
        next_df_cell_obs = info["df_cell"] if "image" in args.observation_type else None
        done = terminations or truncations
        cumulative_return += rewards
        discounted_cumulative_return += rewards * args.gamma
        if (
            args.observation_type == "simple"
            or args.observation_type == "image_cell_types"
        ):
            rb.add(obs, actions, rewards, next_obs, done)
        else:
            rb.add(df_cell_obs, actions, rewards, next_df_cell_obs, done, type_to_int)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs.copy()
        df_cell_obs = (
            next_df_cell_obs.copy() if "image" in args.observation_type else None
        )
        if global_step == args.batch_size:
            data = rb.sample()
            data_next_state = data["next_state"]
            data_state = data["state"]
            with torch.no_grad():
                next_state_actions, _, _ = actor.get_action(data_next_state)
                _, _ = (
                    qf1(data_next_state, next_state_actions),
                    qf2(data_next_state, next_state_actions),
                )
                _, _ = (
                    qf1_target(data_next_state, next_state_actions),
                    qf2_target(data_next_state, next_state_actions),
                )
            del next_state_actions, data_next_state, data_state, data

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample()
            data_next_state = data["next_state"]
            data_state = data["state"]

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data_next_state
                )
                qf1_next_target = qf1_target(data_next_state, next_state_actions)
                qf2_next_target = qf2_target(data_next_state, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data["reward"].flatten() + (
                    1 - data["done"].flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data_state, data["action"]).view(-1)
            qf2_a_values = qf2(data_state, data["action"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data_state)

                    qf1_pi = qf1(data_state, pi)
                    qf2_pi = qf2(data_state, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data_state)

                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()

                        alpha = log_alpha.exp().item()
                    entropy = -log_pi.mean().item()

                losses = {
                    "losses/min_qf_next_target": min_qf_next_target.mean().item(),
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/entropy": entropy,
                }

                for tag, value in losses.items():
                    writer.add_scalar(tag, value, global_step)

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
        scalars = {
            "env/drug_1": actions[0],
            "env/reward_value": rewards,
            "env/number_cancer_cells": info["number_cancer_cells"],
            "env/number_cell_1": info["number_cell_1"],
            "env/number_cell_2": info["number_cell_2"],
        }
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, global_step)
        if done:
            writer.add_scalar(
                "charts/episodic_return", cumulative_return / step_episode, global_step
            )
            writer.add_scalar("charts/episodic_length", step_episode, global_step)
            norm_coeff = (1 - args.gamma ** (step_episode + 1)) / (1 - args.gamma)
            writer.add_scalar(
                "charts/normalized_discounted_episodic_return",
                cumulative_return / norm_coeff,
                global_step,
            )
            episode += 1
            step_episode = 0
            discounted_cumulative_return = 0
            cumulative_return = 0
            obs, info = env.reset()
            done = False
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
