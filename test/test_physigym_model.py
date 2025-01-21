###
# title: test_physigym_model.py
#
# language: python3 and command line
# author: Alexandre Bertin, Elmar Bucher
# date: 2025-01-20
# license: BSD 3-Clause
#
# description:
#   pytest unit test library for the physigym modeling framework.
#   + https://docs.pytest.org/
#
# note:
#   assert actual == expected, message
#   == value equality
#   is reference equality
#   pytest.approx for real values
#####


# load library
import os
import platform
import subprocess
import shutil

# const
s_path_physigym = os.getcwd()
s_path_physicell = '/'.join(s_path_physigym.replace('\\','/').split('/')[:-1] + ['PhysiCell'])


# function

class TestPhysigymTemplate(object):
    ''' tests for the physigym template model. '''

    def test_physigym_template_classic(self):
        os.chdir(s_path_physigym)
        s_result = subprocess.run(['python3', 'install_physigym.py', 'template', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        s_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'load', 'PROJ=physigym_template'], check=False, capture_output=True)
        s_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'classic','-j8'], check=False, capture_output=True)
        s_result = subprocess.run(['./project'], check=False, capture_output=True)
        #print('\n',s_result)
        assert(os.path.exists('output/episode00000000/output00000024.xml')) and \
              (os.path.exists('output/episode00000001/output00000024.xml')) and \
              (os.path.exists('output/episode00000002/output00000024.xml')) and \
              (os.path.exists('output/episode00000003/output00000024.xml'))
        shutil.rmtree('output/')
        assert True

    def test_physigym_template_embedded(self):
        os.chdir(s_path_physigym)
        s_result = subprocess.run(['python3', 'install_physigym.py', 'template', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src='user_projects/physigym_template/run_physigym_template_episodes.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        s_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'load', 'PROJ=physigym_template'], check=False, capture_output=True)
        s_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        s_result = subprocess.run(['make'], check=False, capture_output=True)
        s_result = subprocess.run(['python3','run_physigym_template_episodes.py'], check=False, capture_output=True)
        #print('\n', s_result)
        assert(os.path.exists('output/episode00000000/output00000023.xml')) and \
              (os.path.exists('output/episode00000001/output00000023.xml')) and \
              (os.path.exists('output/episode00000002/output00000023.xml'))
        shutil.rmtree('output/')


class TestPhysigymTutorial(object):
    ''' tests for the physigym tutorial model. '''

    def test_physigym_tutorial_classic(self):
        os.chdir(s_path_physigym)
        s_result = subprocess.run(['python3', 'install_physigym.py', 'tutorial', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        s_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'load', 'PROJ=physigym_tutorial'], check=False, capture_output=True)
        s_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'classic','-j8'], check=False, capture_output=True)
        s_result = subprocess.run(['./project'], check=False, capture_output=True)
        #print('\n', s_result)
        assert(os.path.exists('output/episode00000000/output00000024.xml')) and \
              (os.path.exists('output/episode00000001/output00000024.xml')) and \
              (os.path.exists('output/episode00000002/output00000024.xml')) and \
              (os.path.exists('output/episode00000003/output00000024.xml'))
        shutil.rmtree('output/')
        assert True

    def test_physigym_tutorial_embedded(self):
        os.chdir(s_path_physigym)
        s_result = subprocess.run(['python3', 'install_physigym.py', 'tutorial', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src='user_projects/physigym_tutorial/run_physigym_tutorial_episodes.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        s_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'load', 'PROJ=physigym_tutorial'], check=False, capture_output=True)
        s_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        s_result = subprocess.run(['make'], check=False, capture_output=True)
        s_result = subprocess.run(['python3','run_physigym_tutorial_episodes.py'], check=False, capture_output=True)
        #print('\n', s_result)
        assert(os.path.exists('output/episode00000000/output00000023.xml')) and \
              (os.path.exists('output/episode00000001/output00000023.xml')) and \
              (os.path.exists('output/episode00000002/output00000023.xml'))
        shutil.rmtree('output/')


class TestPhysigymEpisode(object):
    ''' tests for the physigym episode model. '''

    def test_physigym_episode_classic(self):
        os.chdir(s_path_physigym)
        s_result = subprocess.run(['python3', 'install_physigym.py', 'episode', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        s_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        s_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'classic','-j8'], check=False, capture_output=True)
        s_result = subprocess.run(['./project'], check=False, capture_output=True)
        #print('\n', s_result)
        assert(os.path.exists('output/episode00000000/output00000024.xml')) and \
              (os.path.exists('output/episode00000001/output00000024.xml')) and \
              (os.path.exists('output/episode00000002/output00000024.xml')) and \
              (os.path.exists('output/episode00000003/output00000024.xml'))
        shutil.rmtree('output/')
        assert True

    def test_physigym_episode_embedded(self):
        os.chdir(s_path_physigym)
        s_result = subprocess.run(['python3', 'install_physigym.py', 'episode', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src='user_projects/physigym_episode/run_physigym_episode_episodes.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        s_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        s_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        s_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        s_result = subprocess.run(['make'], check=False, capture_output=True)
        s_result = subprocess.run(['python3','run_physigym_episode_episodes.py'], check=False, capture_output=True)
        #print('\n', s_result)
        assert(os.path.exists('output/episode00000000/output00000023.xml')) and \
              (os.path.exists('output/episode00000001/output00000023.xml')) and \
              (os.path.exists('output/episode00000002/output00000023.xml'))
        shutil.rmtree('output/')

    #def test_physigym_episode_classic_nonrandom(self):
    #    assert False

    #def test_physigym_episode_classic_random(self):
    #    assert False

    #def test_physigym_episode_embedded_nonrandom((self):
    #    assert False

    #def test_physigym_episode_embedded_random((self):
    #    assert False

