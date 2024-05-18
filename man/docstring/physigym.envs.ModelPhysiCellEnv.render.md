# physigym.envs.ModelPhysiCellEnv.render()


## input:
```
            self.get_img()

```

## output:
```
            a_img: numpy array or None
                if self.render_mode is
                None: the function will return None.
                rgb_array or huamn: the function will return a numpy array,
                    8bit, shape (4,y,x) with red, green, blue, and alpha channel.
```

## run:
```
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv', render_mode= None)
            env = gymnasium.make('physigym/ModelPhysiCellEnv', render_mode='human')
            env = gymnasium.make('physigym/ModelPhysiCellEnv', render_mode='rgb_array')

            o_observation, d_info = env.reset()
            env.render()

```

## description:
```
            function to render the image into an 8bit numpy array,
            if render_mode is not None.
        
```