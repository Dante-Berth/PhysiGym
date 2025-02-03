import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "drug_apoptosis",
            "drug_reducing_antiapoptosis",
        ],
    ):
        """
        Args:
            env (gym.Env): The environment to wrap.
            list_variable_name (list[str]): List of variable names corresponding to actions in the original env.
        """
        super().__init__(env)

        # Check that all variable names are strings
        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be of type str, but got {type(variable_name).__name__}"
                )

        self.list_variable_name = list_variable_name

        # Flatten the action space to match the expected two values
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

    @property
    def action_space(self):
        """Returns the flattened action space for the wrapper."""
        return self._action_space

    def reset(self, seed=None, options={}):
        """
        Resets the environment and preprocesses the observation.
        """
        o_observation, info = self.env.reset(seed=seed, options=options)

        # Preprocess observation (if needed)
        o_observation = np.array(o_observation, dtype=float)

        return o_observation, info

    def step(self, action: np.ndarray):
        """
        Steps through the environment using the flattened action.

        Args:
            action (np.ndarray): The flattened action array.

        Returns:
            Tuple: Observation, reward, terminated, truncated, info.
        """
        # Convert the flat action array to the dictionary expected by the env
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }
        # Take a step in the environment
        o_observation, r_reward, b_terminated, b_truncated, info = self.env.step(
            d_action
        )

        # Preprocess observation (if needed)
        o_observation = np.array(o_observation, dtype=float)

        return o_observation, r_reward, b_terminated, b_truncated, info

def wrap_env_with_rescale_stats_autoreset(env: gym.Env, min_action:float=-1, max_action:float=1):
    """
    Applies RescaleAction to normalize actions between -1 and 1,
    Records Episode Statistics, and enables Autoreset for the environment.
    """
    env = gym.wrappers.RescaleAction(env, min_action=min_action, max_action=max_action)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.Autoreset(env)
    return env

def wrap_gray_env_image(env,resize_shape=(None,None), stack_size=2, gray=True):
    if resize_shape != (None, None):
        env = gym.wrappers.ResizeObservation(env,resize_shape)
    if gray:
        env = gym.wrappers.GrayscaleObservation(env)
    class UInt8Wrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return obs.astype(np.uint8)
    
    env = UInt8Wrapper(env)
    print(env.observation_space.shape)

    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    if not gray:
        C,H,W,S = env.observation_space.shape
        env = gym.wrappers.ReshapeObservation(env, shape=(C*S, H, W))
    return env
