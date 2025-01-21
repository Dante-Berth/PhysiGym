####
# title: drift_ice.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/test_episode.py
#
# description:
#     unit test code for the physigym project
#     note: pytest and physigym enviroment are incompatible.
#####


# modules
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import pcdl


#############
# run tests #
#############
print('\nUNITTEST check for drift ice ...')


#############################
# generate data frame files #
#############################

#######################
# get timeseries data #
#######################
for s_path in glob.glob('output/episode0*'):
    print(f'processing: {s_path}')
    i_episode = int(s_path.replace('output/episode',''))
    mcdsts = pcdl.TimeSeries(s_path, settingxml=f'PhysiCell_settings.xml')
    #mcdsts = pcdl.TimeSeries(s_path, settingxml=f'PhysiCell_settings_episode{str(i_episode).zfill(8)}.xml')
    df_cell = mcdsts.get_cell_df().drop({'runtime','ID'}, axis=1)
    df_conc = mcdsts.get_conc_df().drop({'runtime'}, axis=1)
    df_cell.to_csv(f'timeseries_cell_episode{str(i_episode).zfill(8)}.csv')
    df_conc.to_csv(f'timeseries_conc_episode{str(i_episode).zfill(8)}.csv')

    # plot timeseries
    fig, axs = plt.subplots(nrows=4, ncols=2 ,figsize=(8,12))
    mcdsts.plot_timeseries('cell_type', ax=axs[0,0])
    mcdsts.plot_timeseries(None, 'substrate_a', ax=axs[1,0])
    mcdsts.plot_timeseries(None, 'substrate_b', ax=axs[2,0])
    mcdsts.plot_timeseries(None, 'substrate_c', ax=axs[3,0])
    mcdsts.plot_timeseries('cell_type', 'substrate_a', ax=axs[1,1])
    mcdsts.plot_timeseries('cell_type', 'substrate_b', ax=axs[2,1])
    mcdsts.plot_timeseries('cell_type', 'substrate_c', ax=axs[3,1])
    fig.suptitle(f'timeseries episode {str(i_episode).zfill(8)}')
    plt.tight_layout()
    fig.savefig(f'timeseries_plot_episode{str(i_episode).zfill(8)}.png')
print()

##################
# load conc data #
##################
ddf_conc = {}
for s_file in sorted(glob.glob('timeseries_conc_episode*csv')):
    i_episode = int(s_file.replace('timeseries_conc_episode','').replace('.csv',''))
    df_conc = pd.read_csv(s_file, index_col=0)
    ddf_conc.update({i_episode: df_conc})

# check results
for i_episode in ddf_conc.keys():
    print(f'processing: conc episode {i_episode}')
    for s_column in  ddf_conc[0].columns:
        if any(ddf_conc[0][s_column] != ddf_conc[i_episode][s_column]):
            print(f'\tepisode conc: {i_episode}\tcolumn: {s_column}')
print()

##################
# load cell data #
##################
ddf_cell = {}
for s_file in sorted(glob.glob('timeseries_cell_episode*csv')):
    i_episode = int(s_file.replace('timeseries_cell_episode','').replace('.csv',''))
    df_cell = pd.read_csv(s_file, index_col=0)
    ddf_cell.update({i_episode: df_cell})

# check results
for i_episode in ddf_cell.keys():
    print(f'processing: cell episode {i_episode}')
    try:
        for s_column in  ddf_cell[0].columns:
            if any(ddf_cell[0][s_column] != ddf_cell[i_episode][s_column]):
                print(f'\tcell episode: {i_episode}\tcolumn: {s_column}')
    except ValueError:
        print(f'\tcell episode: {i_episode}\t{ddf_cell[0].shape} {ddf_cell[i_episode].shape}\terror: series not identically labeled')
print()

# finish
print('UNITTEST: ok!')

