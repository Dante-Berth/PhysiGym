# physigym : Bridging PhysiCell and Gymnasium

Please install the latest version of the physigym user project, as described in the [HowTo](HOWTO_physigym.md) section.


## The most basic run.

Open a Python shell.

```python
import gymnasium
import physigym  # import the Gymnasium PhysiCell bridge module
```

List the registered Gymnasium classes.
In this listing you should find below the core classes, that ship with the basic installation, the just installed and imported physigym/ModelPhysiCellEnv-v\* Gymnasium environment class.

```python
gymnasium.envs.pprint_registry()
```

Let's make an instance of the ModelPhysiCellEnv class and do a manual PhysiCell run.
The output should look familiar to PhysiCell users.
```python
env = gymnasium.make("physigym/ModelPhysiCellEnv-v0", settingxml="config/PhysiCell_settings.xml")

env.reset()  # initialize PhysiCell run
env.step(action={})  # do one Gymnasium time step (similar to a mcds timestep)
env.close()  # drop the PhysiCell Gymnasium environment
```

And kill the Python runtime.
```python
exit()
```

Let's reset the output folder:
```bash
make data-cleanup
```


## A more elaborate example ~ the [tutorial](../model/tutorial) model.

In this somewhat more realistic example, we will control the model so that the cell count for the "default" cell type over time stabilizes at about 64 cells.
For observation, we will use cell counts.
Reward will be calculated by a simple formula.
For action, we will use an apoptosis-inducing drug that kills the cells.

For this tutorial, we assume you have additionally [PhysiCell Studio](https://github.com/PhysiCell-Tools/PhysiCell-Studio) installed.


0. Preparation (Bash).

0.1 Load a fresh [template](../model/template) model.

```bash
make data-cleanup clean reset  # ./config/PhysiCell_settings-backup.xml related Error can be ignored!
make list-user-projects
make load PROJ=physigym_template
```


1. The PhysiCell level (C++ and Studio).

1.1 Fire up the studio.

```bash
studio -c config/PhysiCell_settings.xml
```


1.2 In the studio, make the following changes and additions.
Don't forget to save!

+ Config Basics: Max Time = 10080 [min] which is 7 [days].
+ Microenvironment: rename my_substrate to drug.
+ Microenvironment: set drug decay rate to 0.01 [1/min].
+ Cell Types / Death: death rate = 0.0 [1/min].
+ Cell Types / Custom Data: delete variable my_variable.
+ Cell Types / Custom Data: add variable Name: apoptosis_rate; Value 0.0; Conserve False; Units 1/min
+ User Params: set random_seed to -1, for random random seeding.
+ User Params: set number_of_cells to 48.
+ User Params: have a look at the time and dt_gym parameters (Type double)!
+ User Params: delete parameters my_str, my_bool, my_int, my_float.
+ User Params: add a parameter cell_count_target; Type int; Value 64; Unit dimensionless.
+ User Params: add a parameter cell_count; Type int; Value 0; Unit dimensionless.
+ User Params: add a parameter drug_dose; Type double; Value 0.0; Units fraction.
+ Cell Type: default; Rules: drug increases apoptosis; Half-max: 0.5; Saturation value: 1.0; Hill power: 4; apply to dead: False; Add rule; enable: True.
+ File / Save.
+ Studio / Quit.


1.3 Compile and run the model the classic way.

For model development, it is sometimes useful to be able to compile and run the model the old-fashioned way.
In fact, this is the only reason why we kept the original [main.cpp](../physigym/main.cpp) in the physigym code base.
Physigym as such is written on top of the [physicell embedding](https://github.com/elmbeech/physicellembedding) python_with_physicell module,
for which the main.cpp file had to be ported to [custom/extending/physicellmodule.cpp](../model/template/custom_modules/extending/physicellmodule.cpp) that you can find in the physigym code base too.

In physigym and in physicell embedding you can compile and run the model the old-fashioned way like this:

```bash
make classic -j8  # the j parameter specifies the number of cores to use for compiling.
./project config/PhysiCell_settings.xml
```

Note that, starting with the second episode, you might see a `WARNING: Setting the random seed again. [...]`.
You can safely ignore this warning!
It is triggered because we run a series of episodes of a model in single runtime and not because we have set a random_seed user parameter.

Let's make a clean slate for the next PhysiCell run.
```bash
rm -r output
mkdir output
```


1.4 Edit the *custom_modules/custom.cpp* file.

We don't need the custom data vector template.
At around *line 125*, delete the custom data vector.

```C++
// add custom data vector
for (int i = 0 ; i < all_cells->size(); i++) {
    std::vector<double> vector_double = VECTOR_ZERO;
    (*all_cells)[i]->custom_data.add_vector_variable("my_vector", vector_double);
}
```


1.5 Edit the *custom_modules/custom.cpp* file.

We will need a function that can add substrate (drug) to the microenvironment.
At the bottom of the file, add this function.

```C++
int add_substrate(std::string s_substrate, double r_dose) {
    // update substrate concentration
    int k = microenvironment.find_density_index(s_substrate);
    for (unsigned int n=0; n < microenvironment.number_of_voxels(); n++) {
        microenvironment(n)[k] += r_dose;
    }
    return 0;
}
```


1.6 Edit the *custom_modules/custom.h* header file.

At the bottom of the file, add the fresh implemented function.

```C++
// add substrate
int add_substrate(std::string s_substrate, double r_dose);
```


2. The PhysiCell Python extending level (C++).

Parameters, custom variables, and custom vectors are only the interface.
We still have to connect them to something meaningful.
This is done in the custom_modules/extending/physicellmodule.cpp in the physicell_step function.
Please have a look at this function.

At *line around 218*, you will find already prepared, commented out example code, for action and observation, for all possible parameter, variable and vector types.

2.1 Edit the *custom_modules/extending/physicellmodule.cpp* file for action.

Let's first focus on the action.
The only thing left to do is to connect our drug_dose parameter with the already implemented set_microenv function.
After the commented-out action example code, at *line 259*, inserts the following line.

```C++
// add drug
add_substrate("drug", parameters.doubles("drug_dose"));
```

2.2 Edit the custom_modules/extending/physicellmodule.cpp file for observation.

For observation, we simply have to update our cell_count parameter with the actual cell count.
After the commented out observation example code, at around *line 302*, insert the following line.

```C++
// receive cell count
parameters.ints("cell_count") = (*all_cells).size();
```

For analysis purposes, we transmit the apoptosis rate over the interface too.
Hence, please insert the following lines as well.
```C++
// receive apoptosis rate
for (Cell* pCell : (*all_cells)) {
    pCell->custom_data["apoptosis_rate"] = get_single_behavior(pCell, "apoptosis");
}
```


3. The PhysiCell Gymnasium level (Python).

Finally, let's update the Gymnasium ModelPhysiCellEnv class, found in the *custom_modules/physigym/physigym/envs/physicell_model.py*.


3.1 Edit the *custom_modules/physigym/physigym/envs/physicell_model.py* file.

3.1.1 *get_action_space* function.

Let's declare the action space.
In the studio, we specified the unit of the drug_dose parameter as a fraction.
This means, in Gymnasium terms, we have a Box space.

First, let's comment out or delete the default "discrete": spaces.Discrete(2)!
Then let's declare a Box space labeled drug_dose
This is a single continuous vector with all possible real values from 0 to 1.

Now, very important, in physigym, the action has always to be provided in dictionary form.
The key to each entry is the parameter, variable, or vector name we defined in the underlying
PhysiCell model and the custom_modules/extending/physicellmodule.cpp.

Your modified action space dictionary should look something like this:
```python
d_action_space = spaces.Dict({
    "drug_dose": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
})
```


3.1.2 *get_observation_space* function.

We do a similar thing for the observation space.
Based on our classic run and when we treat with drug, we do not expect to have more than 2^16 cells.

Your modified observation space dictionary should look something like this:
```python
o_observation_space = spaces.Box(low=0, high=(2**16 - 1), shape=(1,), dtype=np.uint16)
```


3.1.3 *get_observation* function.

In our model, the only thing we have to observe is the cell count.
The way we provide this information has to be compatible with the observation space we defined above, in our case, a single integer number.

Replace the default `o_observation = True` with:

```python
o_observation = np.array([physicell.get_parameter("cell_count")], dtype=np.uint16)
```

3.1.4 *get_info* function.

We could provide additional information essential for controlling the action of the policy.
For example, if we do reinforcement learning on a [jump and run game](https://c64online.com/c64-games/the-great-giana-sisters/), the number of lives left from our character.

In our simple model, we don't have such information.

So, just leave the default, the empty dictionary.


3.1.5 *get_terminated* function.

In our model, the run (episode) will be terminated if all cells are dead and the species has died out.
Note that it is a huge difference, if the model is terminated (all cells are dead) or is truncated (simply runs out of max time).

Replace the default `b_terminated = False` with:

```python
b_terminated = physicell.get_parameter("cell_count") <= 0
```


3.1.6 *get_reward* function.

The reward has to be a float number between or equal to 0.0 and 1.0.
Comment out or delete the default `r_reward = 0.0`.
Our reward algorithm looks like this:

```python
i_cellcount_target = physicell.get_parameter("cell_count_target")
i_max = i_cellcount_target * 2
i_cellcount = np.clip(physicell.get_parameter("cell_count"), a_min=0, a_max=i_max)
if (i_cellcount == i_cellcount_target):
    r_reward = 1
elif (i_cellcount < i_cellcount_target):
    r_reward = i_cellcount / i_cellcount_target
elif (i_cellcount > i_cellcount_target):
    r_reward = 1 - (i_cellcount - i_cellcount_target) / i_cellcount_target
else:
    sys.exit(f"Error @ CorePhysiCellEnv.get_reward : strange clipped cell_count detected {i_cellcount}.")
```


3.1.7 *get_img* function.

We will now implement a plot, to display drug concentration in the domain, as well as all cells, colored by the apoptosis rate.

```python
##################
# substrate data #
##################

df_conc = pd.DataFrame(physicell.get_microenv("drug"), columns=["x","y","z","drug"])
df_conc = df_conc.loc[df_conc.z == 0.0, :]
df_mesh = df_conc.pivot(index="y", columns="x", values="drug")
ax.contourf(
    df_mesh.columns, df_mesh.index, df_mesh.values,
    vmin=0.0, vmax=0.2, cmap="Reds",
    #alpha=0.5,
)

#######################
# substrate color bar #
#######################

self.fig.colorbar(
    mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=0.2), cmap="Reds"),
    label="drug_concentration",
    ax=ax,
)

#############
# cell data #
#############

df_cell = pd.DataFrame(physicell.get_cell(), columns=["ID","x","y","z","dead","cell_type"])
df_variable = pd.DataFrame(physicell.get_variable("apoptosis_rate"), columns=["apoptosis_rate"])
df_cell = pd.merge(df_cell, df_variable, left_index=True, right_index=True, how="left")
df_cell = df_cell.loc[df_cell.z == 0.0, :]
df_cell.plot(
    kind="scatter", x="x", y="y", c="apoptosis_rate",
    xlim=[
        int(self.x_root.xpath("//domain/x_min")[0].text),
        int(self.x_root.xpath("//domain/x_max")[0].text),
    ],
    ylim=[
        int(self.x_root.xpath("//domain/y_min")[0].text),
        int(self.x_root.xpath("//domain/y_max")[0].text),
    ],
    vmin=0.0, vmax=0.1, cmap="viridis",
    grid=True,
    title=f'dt_gym env step {str(self.step_env).zfill(4)} episode {str(self.episode).zfill(3)} episode step {str(self.step_episode).zfill(3)} : {df_cell.shape[0]} / {physicell.get_parameter("cell_count_target")} [cell]',
    ax=ax,
)
```


4. Running the model (Python and Bash).

4.1 Compile the model.

This is necessary, because of all the changes we did in the PhysiCell custom.cpp code and the extending module.
And even the physigym Python module is ultimately installed in editable mode (have a look at the pip install command in the Make file), the module has still to be built and installed once.
```bash
make
```


4.2 Python script that will run one episode of the model.

Open a Pyton shell.

If you run ipython, activate your matplotlib backend for on the fly human render mode visualization.

```python
%matplotlib
```

Execute the following code sequence:

```python
# library
from extending import physicell
import gymnasium
import numpy as np
import physigym

# load PhysiCell Gymnasium environment
env = gymnasium.make("physigym/ModelPhysiCellEnv-v0", render_mode="human", render_fps=10)

# reset the environment
r_reward = 0.0
o_observation, d_info = env.reset()

# time step loop
b_episode_over = False
while not b_episode_over:

    # policy according to o_observation
    i_observation = o_observation[0]
    if (i_observation >= physicell.get_parameter("cell_count_target")):
        d_action = {"drug_dose": np.array([1.0 - r_reward])}
    else:
        d_action = {"drug_dose": np.array([0.0])}

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()

# kill the python runtime.
exit()
```

Alternatively, write a Python script that does the same.


4.3 Python script that will run more than one episode of the model.

```python
# library
from extending import physicell
import gymnasium
import numpy as np
import physigym

# load PhysiCell Gymnasium environment
env = gymnasium.make("physigym/ModelPhysiCellEnv-v0", render_mode="human", render_fps=10)

# episode loop
for i_episode in range(3):

    # set episode output folder
    env.get_wrapper_attr("x_root").xpath("//save/folder")[0].text = f"output/episode{str(i_episode).zfill(8)}"

    # reset the environment
    r_reward = 0.0
    o_observation, d_info = env.reset()

    # time step loop
    b_episode_over = False
    while not b_episode_over:

        # policy according to o_observation
        i_observation = o_observation[0]
        if (i_observation > physicell.get_parameter("cell_count_target")):
            d_action = {"drug_dose": np.array([1.0 - r_reward])}
        else:
            d_action = {"drug_dose": np.array([0.0])}

        # action
        o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
        b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()

# kill the python runtime.
exit()
```


4.4 Further readings.

For more information about the Gymnasium interface, please study the official documentation!
+ https://gymnasium.farama.org/


5. The PhysiCell data loader for data analysis (Python and Bash).

We can do similar plotting and even more in-depth data analysis with the [pcdl](https://github.com/elmbeech/physicelldataloader) library on the dumped data.

Install the [pcdl](https://github.com/elmbeech/physicelldataloader) library.

```bash
pip3 install -U pcdl
```

Open a Python shell and use the [pcdl](https://github.com/elmbeech/physicelldataloader) library to do a time series plot.

```python
# library
import matplotlib.pyplot as plt
import os
import pcdl

# load time series
mcdsts = pcdl.TimeSeries("output/")

# make scatter and contour plot overlays
os.makedirs("output/render_pcdl/", exist_ok=True)
for i, mcds in enumerate(mcdsts.l_mcds):
    fig, ax = plt.subplots(figsize=(9,6))
    fig.suptitle(f"drug controlled time series: {int(mcds.get_time())}[min] {mcds.get_cell_df().shape[0]}[cell]")
    ax.axis("equal")
    mcds.plot_contour("drug", vmin=0.0, vmax=0.2, cmap="Reds", ax=ax)
    mcds.plot_scatter("apoptosis_rate", z_axis=[0.0,0.01], cmap="viridis", ax=ax)
    plt.tight_layout()
    fig.savefig(f"output/render_pcdl/drug_ctrld_timeseries_time{str(int(mcds.get_time())).zfill(8)}min.jpeg", facecolor="white")
    plt.close()

# make a gif from the plots
mcdsts.make_gif("output/render_pcdl/", interface="jpeg")

# additionally, make time series plots
mcdsts.plot_timeseries(title="total cell count over time", ext="jpeg")
mcdsts.plot_timeseries(focus_cat="cell_type", focus_num="apoptosis_rate", frame="cell_df", title="mean apoptosis rate over time", ext="jpeg")
mcdsts.plot_timeseries(focus_num="drug", frame="conc_df", title="mean drug concentration over time", ext="jpeg")
```

![Tutorial Model](img/tutorial/model_tutorial.gif)
![Tutorial Model Time Series](img/tutorial/model_tutorial_timeseries.jpeg)


## The [episode](../model/episode) model.

The physigym episode mode is more or less the PhysiCell episode sample project.
The episode sample project comprises three cell types, each of which is secreting a substrate.
The difference between sample project and physigym model is an added nifty set of rules which defined that the substrates protect the cells from becoming apoptotic.

The episode sample project was released with PhysiCell version 1.14.2, to run a series of properly reset episodes of a PhysiCell model within one runtime.
Before PhysiCell release 1.14.2 this was not possible.
Yet, beeing able to run a model consecutively in a runtime is an absolute necessity for any kind of learning.
One simply cannot learn and improve from one episode to another, if one has to destroy the runtime after each episode, to reset to the initial condition because one will at the same time erase everything learned from this episode.


0. Install and load the model (Bash).

```bash
cd path/to/PhysiGym
python3 install_physigym.py episode
```
```bash
cd ../PhysiCell
make clean data-cleanup reset  # ./config/PhysiCell_settings-backup.xml related Error can be ignored!
make list-user-projects
make load PROJ=physigym_episode
```

1. Compile and run the model the classic way (Bash).

Running the model the classic way, there will be no policy to control for the number of cells by regulating the amount of substrate the cell are secreting.
Since the secreted substrate is protecting the cells to undergo apoptosis, the cell population will just grow exponentially.

```bask
make classic -j8
./project
```

2. Compile the embedded way (Bash).

```bash
make
```

3. Run the model the embeded way (Python).

Running the model the embedded way, we can introduce a very policy which tries to control the number of  cells for each cell type by the amount set for cell_count_target in the setting xml file.

Open a ipython shell.

```python
# library
from extending import physicell
import gymnasium
import numpy as np
import physigym

# load PhysiCell Gymnasium environment
%matplotlib
env = gymnasium.make(
     "physigym/ModelPhysiCellEnv-v0",
     settingxml="config/PhysiCell_settings.xml",
     figsize=(12,6),
     render_mode="human",
     render_fps=10
)

# reset the environment
r_reward = 0.0
o_observation, d_info = env.reset()

# time step loop
b_episode_over = False
while not b_episode_over:

    # policy according to o_observation
    d_observation = o_observation
    d_action = {
        "secretion_a": np.array([0.0]),
        "secretion_b": np.array([0.0]),
        "secretion_c": np.array([0.0]),
    }
    # celltype a
    if (d_observation["celltype_a"][0] <= physicell.get_parameter("cell_count_target")):
        d_action.update({"secretion_a": np.array([(1 - r_reward) * 1/12])})
    # celltype b
    if (d_observation["celltype_b"][0] <= physicell.get_parameter("cell_count_target")):
        d_action.update({"secretion_b": np.array([(1 - r_reward) * 1/12])})
    # celltype c
    if (d_observation["celltype_c"][0] <= physicell.get_parameter("cell_count_target")):
        d_action.update({"secretion_c": np.array([(1 - r_reward) * 1/12])})

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()
```

![Episode Model](img/tutorial/model_episode.gif)
