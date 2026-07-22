# physigym.envs.ModelPhysiCellEnv()


## input:
```
    physigym.CorePhysiCellEnv

```

## output:
```
    physigym.ModelPhysiCellEnv

```

## run:
```python
    import gymnasium
    import physigym

    env = gymnasium.make("physigym/ModelPhysiCellEnv")

    o_observation, d_info = env.reset()
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(action={})
    env.close()

```

## description:
```
    this is the model physigym environment class, built on top of the
    physigym.CorePhysiCellEnv class, which is built on top of the
    gymnasium.Env class.

    fresh from the PhysiGym repo this is only a template class!
    you will have to edit this class, to specify the model specific
    reinforcement learning environment.

```
