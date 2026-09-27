# PhysiGym Reference Man Page

This is the technical description of the machinery and how to operate it.


## physigym module

References are maintained in each custom module's [docstring](https://en.wikipedia.org/wiki/Docstring).\
You can access them through the [source code](../physigym/custom_modules/physigym/physigym/envs)
or by first loading the physigym module and environment,

```bash
cd path/to/PhysiCell
```

```python
from extending import physicell
import gymnasium
import physigym

env = gymnasium.make('physigym/ModelPhysiCellEnv')
```

Then, for each physicell module function, getting on-the-fly reference information with the [help](https://en.wikipedia.org/wiki/Help!) command.

### About the ModulePhysiCellEnv class
+ [help(physigym.envs.ModelPhysiCellEnv)](docstring/physigym.envs.ModelPhysiCellEnv.md)

### Class functions to run episodes:
+ [help(physigym.envs.ModelPhysiCellEnv.__init__)](docstring/physigym.envs.ModelPhysiCellEnv.__init__.md)  # initialize environment
+ [help(physigym.envs.ModelPhysiCellEnv.render)](docstring/physigym.envs.ModelPhysiCellEnv.render.md)  # render environment output
+ [help(physigym.envs.ModelPhysiCellEnv.reset)](docstring/physigym.envs.ModelPhysiCellEnv.reset.md)  # reset environment
+ [help(physigym.envs.ModelPhysiCellEnv.step)](docstring/physigym.envs.ModelPhysiCellEnv.step.md)  # step through environment
+ [help(physigym.envs.ModelPhysiCellEnv.close)](docstring/physigym.envs.ModelPhysiCellEnv.close.md)  # close environment
+ [help(physigym.envs.ModelPhysiCellEnv.verbose_true)](docstring/physigym.envs.ModelPhysiCellEnv.verbose_true.md)  # physigym standard stream output on
+ [help(physigym.envs.ModelPhysiCellEnv.verbose_false)](docstring/physigym.envs.ModelPhysiCellEnv.verbose_false.md)  # physigym standard stream output off

### Edit this class functions from this [template](../model/template/custom_modules/physigym/physicell_model.py) to specify the model:
+ [help(physigym.envs.ModelPhysiCellEnv.get_action_space)](docstring/physigym.envs.ModelPhysiCellEnv.get_action_space.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_observation_space)](docstring/physigym.envs.ModelPhysiCellEnv.get_observation_space.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_observation)](docstring/physigym.envs.ModelPhysiCellEnv.get_observation.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_info)](docstring/physigym.envs.ModelPhysiCellEnv.get_info.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_terminated)](docstring/physigym.envs.ModelPhysiCellEnv.get_terminated.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_reset_values)](docstring/physigym.envs.ModelPhysiCellEnv.get_reset_values.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_reward)](docstring/physigym.envs.ModelPhysiCellEnv.get_reward.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_img)](docstring/physigym.envs.ModelPhysiCellEnv.get_img.md)

### Pure internal class functions:
help(physigym.envs.CorePhysiCellEnv.get_truncated)

### Python/PhysiCell API functions:

**observation**
+ [help(physicell.get_parameter)](docstring/physicell.get_parameter.md)
+ [help(physicell.get_variable)](docstring/physicell.get_variable.md)
+ [help(physicell.get_vector)](docstring/physicell.get_vector.md)
+ [help(physicell.get_cell)](docstring/physicell.get_cell.md)
+ [help(physicell.get_microenv)](docstring/physicell.get_microenv.md)

**action**
+ [help(physicell.set_parameter)](docstring/physicell.set_parameter.md)
+ [help(physicell.set_variable)](docstring/physicell.set_variable.md)
+ [help(physicell.set_vector)](docstring/physicell.set_vector.md)

**internal control**
+ help(physicell.start)
+ help(physicell.step)
+ help(physicell.stop)
