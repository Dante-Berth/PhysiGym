#####
# title: pysigym/envs/physicell_model.py
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
#     model specific implementation of the custom_modules/extending module
#     comaptible Gymnasium environment.
# + https://gymnasium.farama.org/main/
# + https://gymnasium.farama.org/main/introduction/create_custom_env/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
#####


# library
from extending import physicell
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import os
import pandas as pd
from physigym.envs.physicell_core import CorePhysiCellEnv


# function
class ModelPhysiCellEnv(CorePhysiCellEnv):
    """
    input:
        physigym.CorePhysiCellEnv

    output:
        physigym.ModelPhysiCellEnv

    run:
        import gymnasium
        import physigym

        env = gymnasium.make('physigym/ModelPhysiCellEnv')

        o_observation, d_info = env.reset()
        o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(action={})
        env.close()

    description:
        this is the model physigym environment class, built on top of the
        physigym.CorePhysiCellEnv class, which is built on top of the
        gymnasium.Env class.

        fresh from the PhysiGym repo this is only a template class!
        you will have to edit this class, to specify the model specific
        reinforcement learning environment.
    """
    def __init__(self, settingxml='config/PhysiCell_settings.xml', figsize=(8, 6), render_mode=None, render_fps=10, verbose=True):
        super(ModelPhysiCellEnv, self).__init__(settingxml=settingxml, figsize=figsize, render_mode=render_mode, render_fps=render_fps, verbose=verbose)


    def get_action_space(self):
        """
        input:

        output:
            d_action_space: dictionary composition space
                the dictionary keys have to match the parameter,
                custom variable, or custom vector label.
                the value has to be defined as gymnasium.spaces object.
                + https://gymnasium.farama.org/main/api/spaces/
        run:
            internal function, user defined.

        description:
            dictionary structure built out of gymnasium.spaces elements.
            this struct has to specify type and range for each
            action parameter, action custom variable, and action custom vector.
        """
        # template
        #action_space = spaces.Dict({
        #    'discrete': spaces.Discrete(2)  # boolean, integer (parameter, custom_variable)
        #    'discrete': spaces.Text() # string (parameter)
        #    'discrete': spaces.MultiBinary()  # boolean in a numpy array (parameter, custom_variable, custom_vector)
        #    'discrete': spaces.MultiDiscrete()  # boolean, integer in a numpy array (parameter, custom_variable, custom_vector)
        #    'numeric': spaces.Box()   # boolean, integer, float in a numpy array (parameter, custom_variable, custom_vector)
        #})

        # model dependent action_space processing logic goes here!
        d_action_space = spaces.Dict({
            'substrate_a_rate': spaces.Box(low=0.0, high=32.0, shape=(1,), dtype=np.float64),
            'substrate_b_rate': spaces.Box(low=0.0, high=32.0, shape=(1,), dtype=np.float64),
            'substrate_c_rate': spaces.Box(low=0.0, high=32.0, shape=(1,), dtype=np.float64),
        })

        # output
        return d_action_space


    def get_observation_space(self):
        """
        input:

        output:
            o_observation_space structure.
                the struct have to be built out of gymnasium.spaces elements.
                there are no other limits.
                + https://gymnasium.farama.org/main/api/spaces/

        run:
            internal function, user defined.

        description:
            data structure built out of gymnasium.spaces elements.
            this struct has to specify type and range
            for each observed variable.
        """
        # template
        #observation_space =
        #compositione: spaces.Dict({})
        #compositione: spaces.Tuple(())
        #discrete: spaces.Discrete()  # boolean, integer
        #discrete: spaces.Text()  # string
        #discrete: spaces.MultiBinary()  # boolean numpy array
        #discrete: spaces.MultiDiscrete() # boolean, integer numpy array
        #numeric: spaces.Box()  # boolean, integer, float numpy array
        #niche: spaces.Graph(())
        #niche: spaces.Sequence(())  # set of spaces

        # model dependent observation_space processing logic goes here!
        o_observation_space = spaces.Dict({
            'celltype_a': spaces.Box(low=0, high=(2**16 - 1), shape=(1,), dtype=np.uint16),
            'celltype_b': spaces.Box(low=0, high=(2**16 - 1), shape=(1,), dtype=np.uint16),
            'celltype_c': spaces.Box(low=0, high=(2**16 - 1), shape=(1,), dtype=np.uint16),
        })

        # output
        return o_observation_space


    def get_observation(self):
        """
        input:

        output:
            o_observation: object compatible with the defined
                observation space struct.

        run:
            internal function, user defined.

        description:
            data for the observation object for example be retrieved by:
            + physicell.get_parameter('my_parameter')
            + physicell.get_variable('my_variable')
            + physicell.get_vector('my_vector')
            however, there are no limits.
        """
        # model dependent observation processing logic goes here!
        df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID','x','y','z','dead','cell_type'])
        df_cell = df_cell.astype({'dead': bool})
        df_cell = df_cell.loc[(df_cell.dead == False), :]
        o_observation = {
            'celltype_a': np.array([df_cell.loc[(df_cell.cell_type == 'celltype_a'), :].shape[0]], dtype=np.uint16),
            'celltype_b': np.array([df_cell.loc[(df_cell.cell_type == 'celltype_b'), :].shape[0]], dtype=np.uint16),
            'celltype_c': np.array([df_cell.loc[(df_cell.cell_type == 'celltype_c'), :].shape[0]], dtype=np.uint16),
        }

        # output
        return o_observation


    def get_info(self):
        """
        input:

        output:
            d_info: dictionary

        run:
            internal function, user defined.

        description:
            function to provide additional information important for
            controlling the action of the policy. for example,
            if we do reinforcement learning on a jump and run game,
            the number of hearts (lives left) from our character.
        """
        # model dependent info processing logic goes here!
        d_info = {}

        # output
        return d_info


    def get_terminated(self):
        """
        input:

        output:
            b_terminated: bool

        run:
            internal function, user defined.

        description:
            function to determine if the episode is terminated.
            for example, if we do reinforcement learning on a
            jump and run game, if our character died.
            please notice, that this ending is different form
            truncated (the episode reached the max time limit).
        """
        # model dependent terminated processing logic goes here!
        df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID','x','y','z','dead','cell_type'])
        df_cell = df_cell.astype({'dead': bool})
        df_cell = df_cell.loc[(df_cell.dead == False), :]
        b_terminated = df_cell.shape[0] <= 0

        # output
        return b_terminated


    def get_reset_values(self):
        """
        input:

        output:

        run:
            internal function, user defined.

        description:
            function to reset model specific self.variables. e.g.:
            self.my_variable = None
        """
        pass


    def get_reward(self):
        """
        input:

        output:
            r_reward: float between or equal to 0.0 and 1.0.
                there are no other limits to the algorithm implementation enforced.
                however, the algorithm is usually based on data retrieved
                by the get_observation function (o_observation, d_info),
                and possibly by the render function (a_img).

        run:
            internal function, user defined.

        description:
            cost function.
        """
        # model dependent reward processing logic goes here!
        i_cellcount_target = physicell.get_parameter('cell_count_target')
        i_max = i_cellcount_target * 2

        df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID','x','y','z','dead','cell_type'])
        df_cell = df_cell.astype({'dead': bool})
        df_cell = df_cell.loc[(df_cell.dead == False), :]

        lr_reward = []
        for s_celltype in ['celltype_a','celltype_b','celltype_b']:
            i_cellcount = np.clip(df_cell.loc[(df_cell.cell_type == s_celltype), :].shape[0], a_min=0, a_max=i_max)
            if (i_cellcount == i_cellcount_target):
                r_reward = 1
            elif (i_cellcount < i_cellcount_target):
                r_reward = i_cellcount / i_cellcount_target
            elif (i_cellcount > i_cellcount_target):
                r_reward = 1 - (i_cellcount - i_cellcount_target) / i_cellcount_target
            else:
                sys.exit('Error @ CorePhysiCellEnv.get_reward : strange clipped cell_count detected {s_celltype} {i_cellcount}.')
            lr_reward.append(r_reward)
        r_reward = np.mean(lr_reward)

        # output
        return r_reward


    def get_img(self):
        """
        input:

        output:
            self.fig.savefig
                instance attached matplotlib figure.

        run:
            internal function, user defined.

        description:
            template code to generate a matplotlib figure from the data.
            for example from:
            + physicell.get_microenv('my_substrate')
            + physicell.get_cell()
            + physicell.get_variable('my_variable')
            however, there are no limits.
        """
        # model dependent img processing logic goes here!
        self.fig.clf()
        ax = self.fig.add_subplot(1,1,1)
        ax.axis('equal')
        #ax.axis('off')

        ##################
        # substrate data #
        ##################

        # substrate a
        df_conc_a = pd.DataFrame(physicell.get_microenv('substrate_a'), columns=['x','y','z','substrate_a'])
        df_conc_a = df_conc_a.loc[df_conc_a.z == 0.0, :]
        df_mesh_a = df_conc_a.pivot(index='y', columns='x', values='substrate_a')
        ax.contourf(
            df_mesh_a.columns, df_mesh_a.index, df_mesh_a.values,
            vmin=0.0, vmax=1.0, cmap='Reds',
            alpha=0.333,
        )

        # substrate b
        df_conc_b = pd.DataFrame(physicell.get_microenv('substrate_b'), columns=['x','y','z','substrate_b'])
        df_conc_b = df_conc_b.loc[df_conc_b.z == 0.0, :]
        df_mesh_b = df_conc_b.pivot(index='y', columns='x', values='substrate_b')
        ax.contourf(
            df_mesh_b.columns, df_mesh_b.index, df_mesh_b.values,
            vmin=0.0, vmax=1.0, cmap='Greens',
            alpha=0.333,
        )

        # substrate c
        df_conc_c = pd.DataFrame(physicell.get_microenv('substrate_c'), columns=['x','y','z','substrate_c'])
        df_conc_c = df_conc_c.loc[df_conc_c.z == 0.0, :]
        df_mesh_c = df_conc_c.pivot(index='y', columns='x', values='substrate_c')
        ax.contourf(
            df_mesh_c.columns, df_mesh_c.index, df_mesh_c.values,
            vmin=0.0, vmax=1.0, cmap='Blues',
            alpha=0.333,
        )

        ######################
        # substrate colorbar #
        ######################

        # substrat a
        self.fig.colorbar(
            mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap='Reds'),
            label='substrate_a',
            ax=ax,
        )

        # substrat b
        self.fig.colorbar(
            mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap='Greens'),
            label='substrate_b',
            ax=ax,
        )

        # substrat a
        self.fig.colorbar(
            mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap='Blues'),
            label='substrate_c',
            ax=ax,
        )

        #############
        # cell data #
        #############

        df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID','x','y','z','dead','cell_type'])
        #df_variable = pd.DataFrame(physicell.get_variable('my_variable'), columns=['my_variable'])
        #df_cell = pd.merge(df_cell, df_variable, left_index=True, right_index=True, how='left')
        for s_celltype, s_color in sorted({'celltype_a': 'magenta', 'celltype_b': 'lime', 'celltype_c': 'cyan'}.items()):
            df_celltype = df_cell.loc[(df_cell.z == 0.0) & (df_cell.cell_type == s_celltype), :]
            df_celltype.plot(
                kind='scatter', x='x', y='y', c=s_color,
                xlim=[
                    int(self.x_root.xpath('//domain/x_min')[0].text),
                    int(self.x_root.xpath('//domain/x_max')[0].text),
                ],
                ylim=[
                    int(self.x_root.xpath('//domain/y_min')[0].text),
                    int(self.x_root.xpath('//domain/y_max')[0].text),
                ],
                #vmin=0.0, vmax=1.0, cmap='viridis',
                grid=True,
                label = s_celltype,
                title=f"dt_gym env step {str(self.step_env).zfill(4)} episode {str(self.episode).zfill(3)} episode step {str(self.step_episode).zfill(3)} : {df_cell.shape[0]} / {physicell.get_parameter('cell_count_target') * 3}[cell]",
                ax=ax,
            ).legend(loc='lower left')

        ################
        # save to file #
        ################

        #plt.tight_layout()
        #s_path = self.x_root.xpath('//save/folder')[0].text + '/render_mode_human/'
        #os.makedirs(s_path, exist_ok=True)
        #self.fig.savefig(f'{s_path}timeseries_step{str(self.step_env).zfill(3)}.jpeg', facecolor='white')

