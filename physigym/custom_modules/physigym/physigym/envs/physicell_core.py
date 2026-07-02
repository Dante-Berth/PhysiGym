#####
# title: pysigym/envs/physicell_core.py
#
# language: python3
# library: gymnasium v1.0.0a1
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     core of the custom_modules/extend module comaptible Gymnasium environment.
# + https://gymnasium.farama.org/main/
# + https://gymnasium.farama.org/main/introduction/create_custom_env/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
#####


# library
from extending import physicell
import gymnasium
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import sys
import tempfile

# global variable
physicell.flag_envphysigym = False


# classes
class CorePhysiCellEnv(gymnasium.Env):
    """
    input:
        gymnasium.Env

    output:
        physigym.CorePhysiCellEnv

    run:
        offspring: physigym.ModelPhysiCellEnv

    description:
        this is the core physigym environment class, built on top of the
        gymnasium.Env class. physigym.CorePhysiCellEnv class as such will be
        the base class for every physigym.ModelPhysiCellEnv.

        there should be no need to edit the physigym.CorePhysiCellEnv class.
        model specifics should be captured in the physigym.ModelPhysiCellEnv class.
    """

    ### begin dummy functions ###

    def get_action_space(self):
        raise NotImplementedError(
            "get_action_space function to be implemented in physigym.ModelPhysiCellEnv!"
        )

    def get_observation_space(self):
        raise NotImplementedError(
            "get_observation_space function to be implemented in physigym.ModelPhysiCellEnv!"
        )

    def get_observation(self):
        raise NotImplementedError(
            "get_observation function to be implemented in physigym.ModelPhysiCellEnv!"
        )

    def get_info(self):
        raise NotImplementedError(
            "get_info function to be implemented in physigym.ModelPhysiCellEnv!"
        )

    def get_terminated(self):
        raise NotImplementedError(
            "get_terminated function to be implemented in physigym.ModelPhysiCellEnv!"
        )

    def get_reset_values(self):
        raise NotImplementedError(
            "get_reset_values function to be implemented in physigym.ModelPhysiCellEnv!"
        )

    def get_reward(self):
        raise NotImplementedError(
            "get_terminated function to be implemented in physigym.ModelPhysiCellEnv!"
        )

    def get_img(self):
        raise NotImplementedError(
            "get_img function to be implemented in physigym.ModelPhysiCellEnv!"
        )

    ### end dummy functions ###

    # metadata
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    # functions
    def __init__(
        self,
        settingxml="config/PhysiCell_settings.xml",
        cell_type_cmap="turbo",
        figsize=(8, 6),
        render_mode=None,
        render_fps=10,
        verbose=True,
        **kwargs,
    ):
        """
        input:
            settingxml: string; default is "config/PhysiCell_settings.xml"
                path and filename to the settings.xml file.
                the file will be loaded with lxml and stored at self.x_root.
                therefor all data from the setting.xml file is later on accessible
                via the self.x_root.xpath("//xpath/string/") xpath construct.
                study this source code class for explicit examples.
                for more information about xpath study the following links:
                + https://en.wikipedia.org/wiki/XPath
                + https://www.w3schools.com/xml/xpath_intro.asp
                + https://lxml.de/xpathxslt.html#xpat

            cell_type_cmap: dictionary of strings or string; default viridis.
                dictionary that maps labels to colors strings.
                matplotlib colormap string.
                https://matplotlib.org/stable/tutorials/colors/colormaps.html

            figsize: tuple of floats; default is (8, 6) which is a 4:3 ratio.
                values are in inches (width, height).

            render_mode: string as specified in the metadata or None; default is None.

            render_fps: float or None; default is 10.
                if render_mode is "human", for every dt_gym step the image,
                specified in the physigym.ModelPhysiCellEnv.get_img() function,
                will be generated and displayed. this frame per second setting
                specifies the time the computer sleeps after the image is
                displayed.
                for example 10[fps] = 1/10[spf] = 0.1 [spf].

            verbose: boolean
                to set standard output verbosity true or false.
                please note, only little from the standard output is coming
                actually from physigym. most of the output comes straight
                from PhysiCell and this setting has no influence over that output.

            **kwargs:
                possible additional keyword arguments input.
                will be available in the instance through self.kwargs["key"].

        output:
            initialized PhysiCell Gymnasium environment.

        run:
            import gymnasium
            import physigym

            env = gymnasium.make("physigym/ModelPhysiCellEnv")

            env = gymnasium.make(
                "physigym/ModelPhysiCellEnv",
                settingxml = "config/PhysiCell_settings.xml",
                figsize = (8, 6),
                render_mode = None,
                render_fps = 10,
                verbose = True
            )

        description:
            function to initialize the PhysiCell Gymnasium environment.
        """
        # check global physigym environment flag
        if physicell.flag_envphysigym:
            raise RuntimeWarning(
                f"per runtime, only one PhysiCellEnv gymnasium environment can be loaded. instance generation cancelled!"
            )

        # handle verbose
        self.verbose = verbose
        # mirror verbosity to the C++ extending module so PhysiCell's own console
        # output (banners, per-step status) is suppressed when running non-verbose
        os.environ["PHYSIGYM_QUIET"] = "0" if self.verbose else "1"
        if self.verbose:
            print(f"physigym: initialize environment ...")

        # initialize class whide variables
        if self.verbose:
            print(f"physigym: declare class instance-wide variables.")
        self.episode = -1
        self.step_episode = None
        self.step_env = 0
        self.time_simulation = -1  # integer
        if self.verbose:
            print("physigym: self.episode", self.episode)
            print("physigym: self.step_episode", self.step_episode)
            print("physigym: self.step_env", self.step_env)
            print("physigym: self.episode", self.episode)

        # handle keyword arguments input
        self.kwargs = kwargs
        if self.verbose:
            print("physigym: self.kwargs", sorted(self.kwargs))

        # load PhysiCell settings.xml file
        # bue 20241130: to gather full-time observation, increase the setting.xml max_time by dt_gym for one more action!
        self.settingxml = settingxml
        self.x_tree = etree.parse(self.settingxml)
        if self.verbose:
            print(f"physigym: reading {self.settingxml}")
        self.x_root = self.x_tree.getroot()

        # handle render mode
        if self.verbose:
            print(f"physigym: declare render settings.")
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f'"{render_mode}" is an unknown render_mode. known are {sorted(self.metadata["render_modes"])}, and None.'
        )
        self.render_mode = render_mode
        self.metadata.update({"render_fps": render_fps})
        if self.verbose:
            print("physigym: self.render_mode", self.render_mode)
            print("physigym: self.metadata", sorted(self.metadata))

        # handle figsize
        self.figsize = figsize
        self.fig = None
        self.axs = None

        # Only create figure if rendering is enabled
        if self.render_mode is not None:
            import matplotlib.pyplot as plt  # local import avoids global font init
            self.fig, self.axs = plt.subplots(figsize=self.figsize)

        if self.verbose:
            print("physigym: self.figsize", self.figsize)

        if self.verbose:
            print("physigym: self.figsize", self.figsize)

        # handle domain
        if self.verbose:
            print(f"physigym: extract domain settings.")
        self.x_min = int(float(self.x_root.xpath("//domain/x_min")[0].text))
        self.x_max = int(float(self.x_root.xpath("//domain/x_max")[0].text))
        self.y_min = int(float(self.x_root.xpath("//domain/y_min")[0].text))
        self.y_max = int(float(self.x_root.xpath("//domain/y_max")[0].text))
        self.z_min = int(float(self.x_root.xpath("//domain/z_min")[0].text))
        self.z_max = int(float(self.x_root.xpath("//domain/z_max")[0].text))
        self.dx = int(float(self.x_root.xpath("//domain/dx")[0].text))
        self.dy = int(float(self.x_root.xpath("//domain/dy")[0].text))
        self.dz = int(float(self.x_root.xpath("//domain/dz")[0].text))
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.depth = self.z_max - self.z_min
        self.total_volume = self.width*self.height*self.depth
        if self.verbose:
            print("physigym: self.x_min", self.x_min)
            print("physigym: self.x_max", self.x_max)
            print("physigym: self.y_min", self.y_min)
            print("physigym: self.y_max", self.y_max)
            print("physigym: self.z_min", self.z_min)
            print("physigym: self.z_max", self.z_max)
            print("physigym: self.dx", self.dx)
            print("physigym: self.dy", self.dy)
            print("physigym: self.dz", self.dz)
            print("physigym: self.width", self.width)
            print("physigym: self.height", self.height)
            print("physigym: self.depth", self.depth)
            print("physigym: self.total_volume", self.total_volume)

        # handle substrate mapping
        if self.verbose:
            print(f"physigym: extract substrate settings.")
        self.substrate_to_id = dict(
            zip(
                self.x_root.xpath("//microenvironment_setup/variable/@name"),
                [
                    int(s_id)
                    for s_id in self.x_root.xpath(
                        "//microenvironment_setup/variable/@ID"
                    )
                ],
            )
        )
        self.substrate_unique = sorted(
            self.substrate_to_id.keys(), key=self.substrate_to_id.get
        )
        self.substrate_count = len(self.substrate_unique)
        if self.verbose:
            print("physigym: self.substrate_to_id", sorted(self.substrate_to_id))
            print("physigym: self.substrate_unique", self.substrate_unique)
            print("physigym: self.substrate_count", self.substrate_count)

        # handle cell_type mapping
        if self.verbose:
            print(f"physigym: extract cell_type settings.")
        self.cell_type_to_id = dict(
            zip(
                self.x_root.xpath("//cell_definitions/cell_definition/@name"),
                [
                    int(s_id)
                    for s_id in self.x_root.xpath(
                        "//cell_definitions/cell_definition/@ID"
                    )
                ],
            )
        )
        self.cell_type_unique = sorted(
            self.cell_type_to_id.keys(), key=self.cell_type_to_id.get
        )
        self.cell_type_count = len(self.cell_type_unique)
        # handle cell_type mapping
        self.cell_type_to_color = {}
        if isinstance(cell_type_cmap, dict):
            for s_cell_type in self.cell_type_unique:
                self.cell_type_to_color[s_cell_type] = cell_type_cmap.get(s_cell_type, "gray")

        elif isinstance(cell_type_cmap, str):
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            cmap = plt.get_cmap(cell_type_cmap, self.cell_type_count)
            for i, s_cell_type in enumerate(self.cell_type_unique):
                self.cell_type_to_color[s_cell_type] = mcolors.to_hex(cmap.colors[i])
        else:
            raise ValueError(
                f"cell_type_cmap {cell_type_cmap} must be a dict of strings or a string."
            )

        if self.verbose:
            print("physigym: self.cell_type_to_id", sorted(self.cell_type_to_id))
            print("physigym: self.cell_type_unique", self.cell_type_unique)
            print("physigym: self.cell_type_count", self.cell_type_count)
            print("physigym: self.cell_type_to_color", sorted(self.cell_type_to_color))

        # handle spaces
        if self.verbose:
            print(f"physigym: declare action and observer space.")
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # set global physigym enviroment flag
        physicell.flag_envphysigym = True

        # output
        if self.verbose:
            print(f"physigym: ok!")

    def render(self, **kwargs):
        """
        input:
            self.get_img()

            **kwargs:
                possible additional keyword arguments input.
                will be available in the instance through self.kwargs["key"].

        output:
            a_img: numpy array or None
                if self.render_mode is
                None: the function will return None.
                rgb_array or human: the function will return a numpy array,
                    8bit, shape (4,y,x) with red, green, blue, and alpha channel.
        run:
            import gymnasium
            import physigym

            env = gymnasium.make("physigym/ModelPhysiCellEnv", render_mode= None)
            env = gymnasium.make("physigym/ModelPhysiCellEnv", render_mode="human")
            env = gymnasium.make("physigym/ModelPhysiCellEnv", render_mode="rgb_array")

            o_observation, d_info = env.reset()
            env.render()

        description:
            function to render the image into an 8bit numpy array,
            if render_mode is not None.
        """
        if self.verbose:
            print(f"physigym: render {self.render_mode} frame ...")

        # handle keyword arguments input
        self.kwargs.update(kwargs)
        if self.verbose and len(kwargs) > 0:
            print("physigym: self.kwarg", sorted(self.kwarg))

        # processing
        a_img = None

        if not (self.render_mode is None):  # human or rgb_array
            self.fig.canvas.draw()
            a_img = np.array(self.fig.canvas.buffer_rgba(), dtype=np.uint8)

        # output
        if self.verbose:
            print(f"ok!")
        return a_img

    def reset(self, seed=None, options={}, **kwargs):
        """
        input:
            self.get_observation()
            self.get_info()
            self.get_img()

            seed: integer or None; default is None
                seed = None: generate random seeds for python and PhyiCell (via the setting.xml file).
                seed < 0: take seed from setting.xml
                seed >= 0: seed python and PhysiCell (via the setting.xml file) with this value.

            options: dictionary or None
                reserved for possible future use.

        output:
            o_observation: structure
                the exact structure has to be
                specified in the get_observation_space function.

            d_info: dictionary
                what information to be captured has to be
                specified in the get_info function.

        run:
            import gymnasium
            import physigym

            env = gymnasium.make("physigym/ModelPhysiCellEnv")

            o_observation, d_info = env.reset()

        description:
            The reset method will be called to initiate a new episode,
            increment episode counter, reset episode step counter.
            You may assume that the step method will not be called
            before the reset function has been called.
        """
        if self.verbose:
            print(f"\nphysigym: reset for episode {self.episode + 1} ...")

        # handle random seeding
        if seed is None:
            i_seed = seed
            if self.verbose:
                print(f"physigym: set {self.settingxml} random_seed to 42.")
            self.x_root.xpath("//random_seed")[0].text = "42"
        # handle setting.xml based seeding
        elif seed < 0:
            s_seed = self.x_root.xpath("//random_seed")[0].text.strip()
            i_seed = int(s_seed)
        # handle Gymnasium based seeding
        else:  # seed >= 0
            i_seed = seed
            if self.verbose:
                print(f"physigym: set {self.settingxml} random_seed to {i_seed}.")
            self.x_root.xpath("//random_seed")[0].text = str(i_seed)


        # seed self.np_random number generator
        super().reset(seed=i_seed)
        if self.verbose:
            print(f"physigym: seed random number generator with {i_seed}.")

        # update class wide variables
        if self.verbose:
            print(f"physigym: update class instance-wide variables.")
        self.episode += 1
        self.step_episode = 0
        # self.step_env NOP

        # handle possible keyword arguments input
        self.kwargs.update(kwargs)
        if self.verbose and len(kwargs) > 0:
            print("physigym: self.kwargs", sorted(self.kwargs))

        # load reset values
        self.get_reset_values()

        # generate output folder
        os.makedirs(self.x_root.xpath("//save/folder")[0].text, exist_ok=True)

        # initialize physiCell model with the temp XML
        if self.verbose:
            print(f"physigym: declare PhysiCell model instance.")
        physicell.start(self.settingxml, self.episode != 0)
        # observe domain
        if self.verbose:
            print(f"physigym: domain observation.")
        o_observation = self.get_observation()
        d_info = self.get_info()

        # render domain
        if self.verbose:
            print(f"physigym: render {self.render_mode} frame.")
        if not (self.render_mode is None):
            plt.ion()
            self.get_img()
            if self.render_mode == "human":  # human
                if not (self.metadata["render_fps"] is None):
                    plt.pause(1 / self.metadata["render_fps"])
            else:  # rgb_array
                try:
                    self.fig.canvas.setVisible(False)
                except AttributeError:  # handle agg backend on headless systems.
                    pass

        # output
        if self.verbose:
            print(
                f"Warning: per runtime, only one PhysiCellEnv gymnasium environment can be generated.\n"
                "to run another env, it will be necessary to fork or spawn the runtime!"
            )
            print(f"physigym: ok!")
        return o_observation, d_info

    def get_truncated(self):
        """
        input:
            settingxml max_time
            PhysiCell parameter time

        output:
            b_truncated: bool

        run:
            internal function.

        description:
            function to evaluate if the episode reached the max_time specified.
        """
        # processing
        b_truncated = False
        # achtung: time has to be declared as parameter of type float in the settings.xml file!
        r_time_simulation = physicell.get_parameter("time")
        b_truncated = self.time_simulation == int(r_time_simulation)
        self.time_simulation = int(r_time_simulation)
        if self.verbose:
            print(f"simulation time python3: {round(r_time_simulation, 3)}")

        # output
        return b_truncated

    def step(self, action, **kwargs):
        """
        input:
            self.get_observation()
            self.get_terminated()
            self.get_truncated()
            self.get_info(kwrags)
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

        output:
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

        run:
            import gymnasium
            import physigym

            env = gymnasium.make("physigym/ModelPhysiCellEnv")

            o_observation, d_info = env.reset()
            o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(action={})

        description:
            function does a dt_gym simulation step:
            apply action, increment the step counters, observes, retrieve reward,
            and finalizes a PhysiCell episode, if episode is terminated or truncated.
        """
        if self.verbose:
            print(f"physigym: taking a dt_gym time step ...")

        # handle keyword arguments input
        self.kwargs.update(kwargs)
        if self.verbose and len(kwargs) > 0:
            print("physigym: self.kwarg", sorted(self.kwarg))

        # do action
        if self.verbose:
            print(f"physigym: action.")

        # action is always a gymnasium composite space dict
        for s_action, o_value in action.items():
            # gymnasium composite space tuple: nop.
            # gymnasium composite space sequences: nop.
            # gymnasium composite space graph: nop.

            # gymnasium action space discrete (boolean, integer)
            # python/physicell api parametre, variable
            if type(o_value) in {bool, int}:
                try:
                    # try custom_variable
                    physicell.set_variable(s_action, o_value)
                except KeyError:
                    # try parameter
                    try:
                        physicell.set_parameter(s_action, o_value)
                    # error
                    except KeyError:
                        sys.exit(
                            f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : unprocessable Gymnasium discrete action space value detected! {s_action} {o_value} {type(o_value)}.\nIn the PhysiCell_setting.xml, have you specified a {s_action} parameter, custom_variable, or custom_vector?"
                        )

            # gymnasium action space text (string)
            # python/physicell api parameter, variable
            elif type(o_value) in {str}:
                try:
                    # try custom_variable
                    physicell.set_variable(s_action, o_value)
                except KeyError:
                    # try parameter
                    try:
                        physicell.set_parameter(s_action, o_value)
                    # error
                    except KeyError:
                        sys.exit(
                            f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : unprocessable Gymnasium text action space value detected! {s_action} {o_value} {type(o_value)}.\nIn the PhysiCell_setting.xml, have you specified a {s_action} parameter, custom_variable, or custom_vector?"
                        )

            # gymnasium action space box (bool, int, float in a numpy array)
            # gymnasium action space multi binary (boolean in a numpy array)
            # gymnasium action space multi discrete (boolean, integer in a numpy array)
            # python/physicell api parameter, variabler, vector
            elif type(o_value) in {np.ndarray}:
                if len(o_value.shape) > 1:
                    o_value = o_value[0]
                try:
                    # try vector
                    physicell.set_vector(s_action, list(o_value))
                except KeyError:
                    # try custom_variable
                    try:
                        physicell.set_variable(s_action, o_value)
                    # try parameter
                    except KeyError:
                        try:
                            physicell.set_parameter(s_action, o_value)
                        # error
                        except KeyError:
                            sys.exit(
                                f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : unprocessable Gymnasium box action space value detected! {s_action} {o_value} {type(o_value)}.\nIn the PhysiCell_setting.xml, have you specified a {s_action} parameter, custom_variable, or custom_vector?"
                            )

            # error
            else:
                sys.exit(
                    f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : unprocessable Gymnasium action space value detected! {s_action} {o_value} {type(o_value)}."
                )

        # do dt_gym time step
        if self.verbose:
            print(f"physigym: PhysiCell model step.")
        physicell.step()

        # update class whide variables
        if self.verbose:
            print(f"physigym: update class instance-wide variables.")
        # self.episode NOP
        self.step_episode += 1
        self.step_env += 1

        # get observation
        if self.verbose:
            print(f"physigym: domain observation.")

        o_observation = self.get_observation()
        b_terminated = self.get_terminated()
        b_truncated = self.get_truncated()
        r_reward = self.get_reward()
        d_info = self.get_info()

        # do rendering
        if self.verbose:
            print(f"physigym: render {self.render_mode} frame.")
        if not (self.render_mode is None):  # human or rgb_array
            self.get_img()
            if (self.render_mode == "human") and not (
                self.metadata["render_fps"] is None
            ):
                plt.pause(1 / self.metadata["render_fps"])

        # check if episode finish
        if b_terminated or b_truncated:
            if self.verbose:
                print(
                    f"physigym: PhysiCell model episode finish by termination ({b_terminated}) or truncation ({b_truncated})."
                )
            physicell.stop()

        # output
        if self.verbose:
            print(f"physigym: ok!")
        return o_observation, r_reward, b_terminated, b_truncated, d_info

    def close(self, **kwargs):
        """
        input:
            **kwargs:
                possible additional keyword arguments input.
                will be available in the instance through self.kwargs["key"].

        output:

        run:
            import gymnasium
            import physigym

            env = gymnasium.make("physigym/ModelPhysiCellEnv")

            env.close()

        description:
            function to drop shutdown physigym environment.
        """
        if self.verbose:
            print(f"physigym: environment closure ...")

        # handle keyword arguments input
        self.kwargs.update(kwargs)
        if self.verbose and len(kwargs) > 0:
            print("physigym: self.kwarg", sorted(self.kwarg))

        # processing
        if self.verbose:
            print(f"physigym: Gymnasium PhysiCell model environment is going down.")
        if not (self.render_mode is None):
            plt.close(self.fig)

        # output
        if self.verbose:
            print(
                f"Warning: per runtime, only one PhysiCellEnv gymnasium environment can be generated.\nto run another env, it will be necessary to fork or spawn the runtime!"
            )
            print(f"physigym: ok!")

    def verbose_true(self):
        """
        input:

        output:

        run:
            import gymnasium
            import physigym

            env = gymnasium.make("physigym/ModelPhysiCellEnv")

            env.unwrapped.verbose_true()

        description:
            to set verbosity true after initialization.

            please note, only little from the standard output is coming
            actually from physigym. most of the output comes straight
            from PhysiCell and this setting has no influence over that output.
        """
        print(f"physigym: set env.verbose = True.")
        self.verbose = True

    def verbose_false(self):
        """
        input:

        output:

        run:
            import gymnasium
            import physigym

            env = gymnasium.make("physigym/ModelPhysiCellEnv")

            env.unwrapped.verbose_true()

        description:
            to set verbosity false after initialization.

            please note, only little from the standard output is coming
            actually from physigym. most of the output comes straight
            from PhysiCell and this setting has no influence over that output.
        """
        print(f"physigym: set env.verbose = False.")
        self.verbose = False
