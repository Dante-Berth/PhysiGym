#####
# title: run_physigym_tibbue_sac.py
#
# language: python3
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# run:
#   1. cd path/to/PhysiCell
#   2. python3 custom_modules/physigym/physigym/envs/run_physigym_tibbue_sac.py
#
# description:
#   sac implementation for tumor immune base model.
#   implementation is based on the following source code:
# + https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
#####


# basic python
import argparse
import numpy as np
import os
import pandas as pd
import random
import time

# gymnasium and physigym
import gymnasium
from gymnasium import spaces
import physigym

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard

# wandb
import wandb

# replybuffer
from alexbaselines import ReplayBuffer


################################
# Class PhysiCellModel Wrapper #
################################


class PhysiCellModelWrapper(gymnasium.Wrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        ls_action: list[str],
        r_weight: float,
    ):
        """
        input:
            env (gym.Env): the environment to be wrapped.
            ls_action (list[str]): list of variable names corresponding to actions in the original env.
            r_weight (float): weight corresponding how much weight is added to the reward term related to cancer cells.

        output:

        description:
            handle flattened numpy action space.
        """
        super().__init__(env)

        # handle possible keyword arguments input
        self.ls_action = ls_action
        self.r_weight = r_weight

        # numpy flattend action space
        a_low = np.array(
            [env.action_space[s_action].low[0] for s_action in self.ls_action]
        )
        a_high = np.array(
            [env.action_space[s_action].high[0] for s_action in self.ls_action]
        )
        self._action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float64)

    @property
    def action_space(self):
        """
        inoput:

        output:
             self._action_space numpy flattened action space.

        description:
             return action space in a form compatible with the wrapper.
        """
        return self._action_space

    def step(self, ar_action: np.ndarray):
        """
        input:
            ar_action (np.ndarray): The flattened action array.

        output:
           tuple of observation, reward, terminated, truncated, and info.

        description:
            steps through the environment using the flattened action.
        """
        # action, not action space
        d_action = {}
        for s_action, r_value in zip(self.ls_action, ar_action):
            d_action.update({s_action: np.array([r_value])})

        # take a step in the environment
        o_observation, r_reward_tumor, b_terminated, b_truncated, d_info = (
            self.env.step(d_action)
        )

        # the reward is the negative mean of all the actions (e.g. one or multiple drugs).
        r_reward_drug = -np.mean(ar_action)

        # calculate overall reward
        r_reward = self.r_weight * r_reward_tumor + (1 - self.r_weight) * r_reward_drug

        # update the info dictionary
        d_info["action"] = d_action  # drug_added
        d_info["reward"] = r_reward
        d_info["reward_tumor"] = r_reward_tumor
        d_info["reward_drug"] = r_reward_drug

        # going home
        return o_observation, r_reward, b_terminated, b_truncated, d_info


#########################
# Class Neural Networks #
#########################


class PixelPreprocess(nn.Module):
    """Normalizes pixel observations to [-0.5, 0.5]."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.Mish()

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x + residual


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.activation = nn.Mish()

    def forward(self, x):
        x = self.activation(self.conv(x))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class FeatureExtractor(nn.Module):
    """Handles both image-based and vector-based state inputs dynamically."""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg

        obs_shape = env.observation_space.shape
        self.is_image = len(obs_shape) == 3  # Check if input is an image (C, H, W)

        if self.is_image:
            # CNN feature extractor
            # num_channels = 8
            # layers = [
            #    PixelPreprocess(),
            #    nn.Conv2d(obs_shape[0], num_channels, 7, stride=2),
            #    nn.Mish(inplace=False),
            #    nn.Conv2d(num_channels, num_channels, 5, stride=2),
            #    nn.Mish(inplace=False),
            #    nn.Conv2d(num_channels, num_channels, 3, stride=2),
            #    nn.Mish(inplace=False),
            #    nn.Conv2d(num_channels, num_channels, 3, stride=1),
            #    nn.Flatten(),
            # ]
            layers = [
                PixelPreprocess(),
                ImpalaBlock(obs_shape[0], 16),
                ImpalaBlock(16, 32),
                ImpalaBlock(32, 32),
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


########
# Run #
#######


def run(
    s_settingxml="config/PhysiCell_settings.xml",
    r_max_time_episode=1440.0,  # xpath
    i_thread=16,  # xpath
    i_seed=None,
    s_observation_mode="scalars",
    s_render_mode=None,
    s_name="sac",
    b_wandb=False,
    i_total_step_learn=int(1e5),
):
    #############
    # Arguments #
    #############

    # run basics
    d_arg_run = {
        # basics
        "name": s_name,  # str: the name of this experiment
        # hardware
        "cuda": True,  # bool: should torch check for gpu (nvidia, amd mroc) accelerator?
        # tracking
        "wandb_track": b_wandb,  # bool: track with wandb, if false locallt tensorboard
        # random seed
        "seed": i_seed,  # int or none: seed of the experiment
        # steps
        "total_timesteps": i_total_step_learn,  # int: the learning rate of the optimizer
    }

    # wandb
    d_arg_wandb = {
        "entity": "corporate-manu-sureli",  # str: the wandb s entity name
        "project": "SAC_IMAGE_TIB",  # str: the wandb s project name
        "sync_tensorboard": True,
        "monitor_gym": True,
        "save_code": True,
    }

    # physigym
    d_arg_physigym_model = {
        "id": "physigym/ModelPhysiCellEnv-v0",  # str: the id of the gymnasium environmenit
        "settingxml": s_settingxml,
        "cell_type_cmap": {
            "cell_1": "navy",
            "cell_2": "green",
            "tumor": "yellow",
        },  # viridis
        "figsize": (6, 6),
        "observation_mode": s_observation_mode,  # str: scalars , img_rgb , img_mc
        "render_mode": s_render_mode,  # human, rgb_array
        "verbose": True,
        "img_rgb_grid_size_x": 64,  # pixel size
        "img_rgb_grid_size_y": 64,  # pixel size
        "img_mc_grid_size_x": 64,  # pixel size
        "img_mc_grid_size_y": 64,  # pixel size
        "normalization_factor": 512,
    }
    d_arg_physigym_wrapper = {
        "ls_action": ["drug_1"],  # list of str: of action varaible names
        "r_weight": 0.5,  # float: weight for the reduction of tumor
    }

    # rl algorithm
    d_arg_rl = {
        # algoritm neural network I
        "buffer_size": 2**13,  # int: the replay memory buffer size
        "batch_size": 256,  # int: the batch size of sample from the reply memory
        "learning_starts": 10e3,  # float: timestep to start learning
        "policy_frequency": 2,  # int: the frequency of training policy (delayed)
        "target_network_frequency": 1,  # int: the frequency of updates for the target nerworks (Denis Yarats" implementation delays this by 2.)
        # algorithm neural network II
        "autotune": True,  # bool: automatic tuning the the entropy coefficient
        "alpha": 0.2,  # float: set manuall entropy regularization coefficient
        "tau": 0.005,  # float: target smoothing coefficient (default" : 0.005)
        "q_lr": 3e-4,  # float: the learning rate of the Q network network optimizer
        "policy_lr": 3e-4,  # float: the learning rate of the policy network optimizer
        # algorithm neural network III: discounted cummulative reward calculation
        "gamma": 0.99,  # float: the discount factor gamma (how much learning)
    }

    # all in one
    d_arg = {}
    d_arg.update(d_arg_run)
    d_arg.update(d_arg_wandb)
    d_arg.update(d_arg_physigym_model)
    d_arg.update(d_arg_physigym_wrapper)
    d_arg.update(d_arg_rl)

    #############
    # main loop #
    #############

    # initialize tracking
    s_run = f"{d_arg['name']}_seed_{d_arg['seed']}_observationtype_{d_arg['observation_mode']}_weight_{d_arg['r_weight']}_time_{int(time.time())}"
    if d_arg["wandb_track"]:
        print("tracking: wandb ...")
        run = wandb.init(name=s_run, config=d_arg, **d_arg_wandb)
        s_dir_run = os.path.join(
            run.dir, s_run
        )  # run.dir wandb/run-20250612_123456-abcdef
    else:
        print("tracking tensorboard ...")
        s_dir_run = os.path.join("tensorboard", s_run)
    s_dir_data = os.path.join(s_dir_run, "data")

    # initialize tensorbord recording
    writer = tensorboard.SummaryWriter(s_dir_run)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [f"|{s_key}|{s_value}|" for s_key, s_value in sorted(d_arg.items())]
            )
        ),
    )

    # initialize csv recording
    s_dir_pcoutput = "output"
    os.makedirs(s_dir_pcoutput, exist_ok=True)

    ls_columns = [
        "episode",
        "step_episode",
        "step_env",
        "time_env",
        "cumulative_return",
        "discounted_cumulative_return",
        "reward",
        "reward_tumor",
        "reward_drug",
        "terminated",
        "truncated",
        "over",
        "tumor",
        "cell_1",
        "cell_2",  # count
        "drug_added",
        "drug_max",
        "anti_inflammatory_factor_max",
        "pro_inflammatory_max",
        "debris_max",
        "drug_median",
        "anti_inflammatory_median",
        "pro_inflammatory_median",
        "debris_median",
        "drug_mean",
        "anti_inflammatory_mean",
        "pro_inflammatory_mean",
        "debris_mean",
        "drug_std",
        "anti_inflammatory_std",
        "pro_inflammatory_std",
        "debris_std",
        "drug_min",
        "anti_inflammatory_min",
        "pro_inflammatory_min",
        "debris_min",
        "\n",
    ]
    s_csv_record = os.path.join(s_dir_pcoutput, "record.csv")
    f = open(s_csv_record, "w")
    f.writelines(ls_columns)
    f.close()

    # set random seed
    random.seed(d_arg["seed"])
    np.random.seed(d_arg["seed"])
    if d_arg["seed"] is None:
        torch.seed()
        torch.backends.cudnn.deterministic = False
    else:
        torch.manual_seed(d_arg["seed"])
        torch.backends.cudnn.deterministic = True

    # initialize physigym environment
    env = gymnasium.make(**d_arg_physigym_model)
    env = PhysiCellModelWrapper(env=env, **d_arg_physigym_wrapper)
    # manipulate setting xml
    env.get_wrapper_attr("x_root").xpath("//overall/max_time")[0].text = str(
        r_max_time_episode
    )
    env.get_wrapper_attr("x_root").xpath("//parallel/omp_num_threads")[0].text = str(
        i_thread
    )

    # initialize neural networks
    o_device = torch.device(
        "cuda" if torch.cuda.is_available() and d_arg["cuda"] else "cpu"
    )  # cpu or gpu
    cfg = {"cfg_FeatureExtractor": {}}
    actor = Actor(env, cfg).to(o_device)
    qf1 = QNetwork(env, cfg).to(o_device)
    qf2 = QNetwork(env, cfg).to(o_device)
    qf1_target = QNetwork(env, cfg).to(o_device)
    qf2_target = QNetwork(env, cfg).to(o_device)
    target_actor = Actor(env, cfg).to(o_device)  # bue: where is target_actor used?
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg["q_lr"]
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=d_arg["policy_lr"])
    # set neural network entropy alpha by automatic tuning or manual
    if d_arg["autotune"]:
        target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape).to(o_device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=o_device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=d_arg["q_lr"])
    else:
        alpha = d_arg["alpha"]

    # initilize the reply buffer
    rb = ReplayBuffer(
        li_observation_dim=env.observation_space.shape,
        li_action_dim=env.action_space.shape,
        o_device=o_device,
        i_buffer_size=d_arg["buffer_size"],
        i_batch_size=d_arg["batch_size"],
        o_observation_mode=env.observation_space.dtype,
    )

    # do reinforcement
    while env.unwrapped.step_env < d_arg["total_timesteps"]:
        # manipulate setting xml before reset to record full physicell run every 1024 episode.
        if env.unwrapped.episode % 1024 == 0:
            env.get_wrapper_attr("x_root").xpath("//save/folder")[
                0
            ].text = os.path.join(
                s_dir_pcoutput, f"episode{str(env.unwrapped.episode).zfill(8)}"
            )
            env.get_wrapper_attr("x_root").xpath("//save/full_data/enable")[
                0
            ].text = "true"
            env.get_wrapper_attr("x_root").xpath("//save/SVG/enable")[0].text = "false"
        else:
            env.get_wrapper_attr("x_root").xpath("//save/folder")[
                0
            ].text = os.path.join(s_dir_pcoutput, "devnull")
            env.get_wrapper_attr("x_root").xpath("//save/full_data/enable")[
                0
            ].text = "false"
            env.get_wrapper_attr("x_root").xpath("//save/SVG/enable")[0].text = "false"

        # reset gymnasium env
        r_cumulative_return = 0
        r_discounted_cumulative_return = 0
        o_observation, d_info = env.reset(seed=d_arg["seed"])

        # time step loop
        b_episode_over = False
        while not b_episode_over:
            # sample the action space or learn
            if env.unwrapped.step_env <= d_arg["learning_starts"]:
                a_action = np.array(env.action_space.sample(), dtype=np.float16)
            else:
                x = torch.Tensor(o_observation).to(o_device).unsqueeze(0)
                actions, _, _ = actor.get_action(x)
                a_action = actions.detach().squeeze(0).cpu().numpy()

            # physigym step
            o_observation_next, r_reward, b_terminated, b_truncated, d_info = env.step(
                a_action
            )
            r_cumulative_return += r_reward
            r_discounted_cumulative_return += r_reward * d_arg["gamma"] ** (
                env.unwrapped.step_episode
            )
            b_episode_over = b_terminated or b_truncated

            # record to reply buffer
            rb.add(
                o_observation=o_observation,
                a_action=a_action,
                o_observation_next=o_observation_next,
                r_reward=r_reward,
                b_episode_over=b_episode_over,
            )

            # for debuging the reply buffer
            # if env.unwrapped.step_env == d_arg["batch_size"]:
            #    data = rb.sample()
            #    with torch.no_grad():
            #        next_state_actions, _, _ = actor.get_action(data["observation_next"])
            #        qf1(data["observation_next"], next_state_actions)
            #        qf2(data["observation_next"], next_state_actions)
            #        qf1_target(data["observation_next"], next_state_actions)
            #        qf2_target(data["observation_next"], next_state_actions)
            #    del data, next_state_actions

            # learning
            if env.unwrapped.step_env > d_arg["learning_starts"]:
                data = rb.sample()
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(
                        data["observation_next"]
                    )
                    qf1_next_target = qf1_target(
                        data["observation_next"], next_state_actions
                    )
                    qf2_next_target = qf2_target(
                        data["observation_next"], next_state_actions
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - alpha * next_state_log_pi
                    )
                    next_q_value = data["reward"].flatten() + (
                        1 - data["episode_over"].flatten()
                    ) * d_arg["gamma"] * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data["observation"], data["action"]).view(-1)
                qf2_a_values = qf2(data["observation"], data["action"]).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # update the target networks
                if env.unwrapped.step_env % d_arg["target_network_frequency"] == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            d_arg["tau"] * param.data
                            + (1 - d_arg["tau"]) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            d_arg["tau"] * param.data
                            + (1 - d_arg["tau"]) * target_param.data
                        )

                # update the policy
                if (
                    env.unwrapped.step_env % d_arg["policy_frequency"] == 0
                ):  # TD 3 Delayed update support
                    # compensate for the delay by doing "actor_update_interval" instead of 1
                    for _ in range(d_arg["policy_frequency"]):
                        pi, log_pi, _ = actor.get_action(data["observation"])

                        qf1_pi = qf1(data["observation"], pi)
                        qf2_pi = qf2(data["observation"], pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        # entropy autotune
                        if d_arg["autotune"]:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data["observation"])

                            alpha_loss = (
                                -log_alpha.exp() * (log_pi + target_entropy)
                            ).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()

                            alpha = log_alpha.exp().item()

                    # record policy update to tensoboard
                    writer.add_scalar(
                        "losses/min_qf_next_target",
                        min_qf_next_target.mean().item(),
                        env.unwrapped.step_env,
                    )
                    writer.add_scalar(
                        "losses/qf1_values",
                        qf1_a_values.mean().item(),
                        env.unwrapped.step_env,
                    )
                    writer.add_scalar(
                        "losses/qf2_values",
                        qf2_a_values.mean().item(),
                        env.unwrapped.step_env,
                    )
                    writer.add_scalar(
                        "losses/qf1_loss", qf1_loss.item(), env.unwrapped.step_env
                    )
                    writer.add_scalar(
                        "losses/qf2_loss", qf2_loss.item(), env.unwrapped.step_env
                    )
                    writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, env.unwrapped.step_env
                    )
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), env.unwrapped.step_env
                    )
                    writer.add_scalar(
                        "losses/entropy", -log_pi.mean().item(), env.unwrapped.step_env
                    )

                    # record policy update to csv
                    # pass

            # handle observation
            o_observation = o_observation_next

            # recording step to tensorboard
            writer.add_scalar("env/drug_1", a_action[0], env.unwrapped.episode)
            writer.add_scalar(
                "env/number_tumor", d_info["number_tumor"], env.unwrapped.episode
            )
            writer.add_scalar(
                "env/number_cell_1", d_info["number_cell_1"], env.unwrapped.episode
            )
            writer.add_scalar(
                "env/number_cell_2", d_info["number_cell_2"], env.unwrapped.episode
            )
            writer.add_scalar("env/reward", r_reward, env.unwrapped.episode)
            writer.add_scalar(
                "env/reward_tumor", d_info["reward_tumor"], env.unwrapped.episode
            )
            writer.add_scalar(
                "env/reward_drug", d_info["reward_drug"], env.unwrapped.episode
            )

            # record step to csv
            se_subs_max = d_info["df_subs"].max()
            se_subs_median = d_info["df_subs"].median()
            se_subs_mean = d_info["df_subs"].mean()
            se_subs_std = d_info["df_subs"].std()
            se_subs_min = d_info["df_subs"].min()
            l_data = [
                str(env.unwrapped.episode),
                str(env.unwrapped.step_episode),
                str(env.unwrapped.step_env),
                str(env.unwrapped.time_simulation),
                str(r_cumulative_return),
                str(r_discounted_cumulative_return),
                str(r_reward),
                str(d_info["reward_tumor"]),
                str(d_info["reward_drug"]),
                str(b_terminated),
                str(b_truncated),
                str(b_episode_over),
                str(d_info["number_tumor"]),
                str(d_info["number_cell_1"]),
                str(d_info["number_cell_2"]),
                str(d_info["action"]["drug_1"][0]),
                str(se_subs_max["drug_1"]),
                str(se_subs_max["anti-inflammatory factor"]),
                str(se_subs_max["pro-inflammatory factor"]),
                str(se_subs_max["debris"]),
                str(se_subs_median["drug_1"]),
                str(se_subs_median["anti-inflammatory factor"]),
                str(se_subs_median["pro-inflammatory factor"]),
                str(se_subs_median["debris"]),
                str(se_subs_mean["drug_1"]),
                str(se_subs_mean["anti-inflammatory factor"]),
                str(se_subs_mean["pro-inflammatory factor"]),
                str(se_subs_mean["debris"]),
                str(se_subs_std["drug_1"]),
                str(se_subs_std["anti-inflammatory factor"]),
                str(se_subs_std["pro-inflammatory factor"]),
                str(se_subs_std["debris"]),
                str(se_subs_min["drug_1"]),
                str(se_subs_min["anti-inflammatory factor"]),
                str(se_subs_min["pro-inflammatory factor"]),
                str(se_subs_min["debris"]),
                "\n",
            ]
            f = open(s_csv_record, "a")
            f.writelines(l_data)
            f.close()

        # recording episode to tensorbord
        writer.add_scalar(
            "charts/cumulative_return", r_cumulative_return, env.unwrapped.episode
        )
        writer.add_scalar(
            "charts/episodic_cumulative_return",
            r_cumulative_return / env.unwrapped.step_episode,
            env.unwrapped.episode,
        )
        writer.add_scalar(
            "charts/episodic_length", env.unwrapped.step_episode, env.unwrapped.episode
        )
        writer.add_scalar(
            "charts/discounted_cumulative_return",
            r_discounted_cumulative_return,
            env.unwrapped.episode,
        )

        # recording episode to csv
        # pass

    # finish
    env.close()
    writer.close()


########
# Main #
########

if __name__ == "__main__":
    print("run physigym learing ...")

    # argv
    parser = argparse.ArgumentParser(
        prog="run physigym episodes",
        description="script to run physigym episodes.",
    )
    # settingxml file
    parser.add_argument(
        "settingxml",
        # type = str,
        nargs="?",
        default="config/PhysiCell_settings.xml",
        help="path/to/settings.xml file.",
    )
    # max_time
    parser.add_argument(
        "--max_time_episode",
        type=float,
        nargs="?",
        default=10080.0,
        help="set overall max_time in min in the settings.xml file.",
    )
    # thread
    parser.add_argument(
        "--thread",
        type=int,
        nargs="?",
        default=16,
        help="set parallel omp_num_threads in the settings.xml file.",
    )
    # seed
    parser.add_argument(
        "--seed",
        # type = int,
        nargs="?",
        default="none",
        help="set options random_seed in the settings.xml file and python.",
    )
    # observation_mode
    parser.add_argument(
        "--observation_mode",
        # type = str,
        nargs="?",
        default="scalars",
        help="observation mode scalars, img_rgb, or img_mc.",
    )
    # render_mode
    parser.add_argument(
        "--render_mode",
        # type = str,
        nargs="?",
        default="none",
        help="render mode None, rgb_array, or human. observation mode scalars needs either render mode rgb_array or human.",
    )
    # name
    parser.add_argument(
        "--name",
        # type = str,
        nargs="?",
        default="sac",
        help="experiment name.",
    )
    # wandb tracking
    parser.add_argument(
        "--wandb",
        # type = bool,
        nargs="?",
        default="false",
        help="tracking online with wandb? false with track locally with tensorboard.",
    )
    # total timesteps
    parser.add_argument(
        "--total_step_learn",
        type=int,
        nargs="?",
        default=int(1e5),
        help="set total time steps for the learing process to take.",
    )

    # parse arguments
    args = parser.parse_args()
    print(args)

    # processing
    run(
        s_settingxml=args.settingxml,
        r_max_time_episode=float(args.max_time_episode),
        i_thread=args.thread,
        i_seed=None if args.seed.lower() == "none" else int(args.seed),
        s_observation_mode=args.observation_mode,
        s_render_mode=None if args.render_mode.lower() == "none" else args.render_mode,
        s_name=args.name,
        b_wandb=True if args.wandb.lower() == "true" else False,
        i_total_step_learn=int(args.total_step_learn),
    )
