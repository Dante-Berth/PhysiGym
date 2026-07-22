# physigym.envs.ModelPhysiCellEnv.reset()


## input:
```
    self.get_observation()
    self.get_info()
    self.get_img()

    seed: integer or None; default is None
        seed = None: generate random seeds for python and PhyiCell (via the setting.xml file).
        seed < 0: take seed from setting.xml
        seed >= 0: seed python and PhysiCell (via the setting.xml file) with this value.

    options: dictionary or None
        reserved for possible future use.

```

## output:
```
    o_observation: structure
        the exact structure has to be
        specified in the get_observation_space function.

    d_info: dictionary
        what information to be captured has to be
        specified in the get_info function.

```

## run:
```python
    import gymnasium
    import physigym

    env = gymnasium.make("physigym/ModelPhysiCellEnv")

    o_observation, d_info = env.reset()

```

## description:
```
    The reset method will be called to initiate a new episode,
    increment episode counter, reset episode step counter.
    You may assume that the step method will not be called
    before the reset function has been called.

```
