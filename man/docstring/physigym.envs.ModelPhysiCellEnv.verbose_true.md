# physigym.envs.ModelPhysiCellEnv.verbose_true()


## input:
```

```

## output:
```

```

## run:
```python
    import gymnasium
    import physigym

    env = gymnasium.make("physigym/ModelPhysiCellEnv")

    env.unwrapped.verbose_true()

```

## description:
```
    to set verbosity true after initialization.

    please note, only little from the standard output is coming
    actually from physigym. most of the output comes straight
    from PhysiCell and this setting has no influence over that output.

```
