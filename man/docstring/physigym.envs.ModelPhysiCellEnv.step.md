# physigym.envs.ModelPhysiCellEnv.step()


## input:
```
    self.get_observation()
    self.get_terminated()
    self.get_truncated()
    self.get_info(kwargs)
    self.get_reward()
    self.get_img()

    action: dict
        object compatible with the defined action space struct.
        the dictionary keys have to match the parameter,
        custom variable, or custom vector label. the values are
        either single or numpy arrays of bool, integer, float,
        or string values.

    **kwargs:
        possible additional keyword arguments input.
        will be available in the instance through self.kwargs["key"].

```

## output:
```
    o_observation: structure
        structure defined by the user in self.get_observation_space().

    r_reward: float or int or bool
        algorithm defined by the user in self.get_reward().

    b_terminated: bool
        algorithm defined by the user in self.get_terminated().

    b_truncated: bool
        algorithm defined in self.get_truncated().

    info: dict
        algorithm defined by the user in self.get_info().

    self.episode: integer
        episode counter.

    self.step_episode: integer
        within an episode step counter.

    self.step_env: integer
        overall episodes step counter.

```

## run:
```python
    import gymnasium
    import physigym

    env = gymnasium.make("physigym/ModelPhysiCellEnv")

    o_observation, d_info = env.reset()
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(action={})

```

## description:
```
    function does a dt_gym simulation step:
    apply action, increment the step counters, observes, retrieve reward,
    and finalizes a PhysiCell episode, if episode is terminated or truncated.

```
