## How to install the physigym PhysiCell user project

1. Fork this repository: https://github.com/Dante-Berth/PhysiGym/tree/main

2. Download the forked repository *in the same folder where your PhysiCell is installed, right next to the PhysiCell folder, not into it*!
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/PhysiGym
```

3. cd into the physigym folder and run the install_physigym.py script.
```bash
cd path/to/PhysiGym
python3 install_physigym.py template
```

4. If you are using environments, this is the time to activate the Python environment in which you would like to run physigym.

5. Check that you are hooked up to the internet because pip must be able to check for build dependencies.

6. cd into the PhysiCell folder, reset PhysiCell, load, and compile the physigym template project.
This will install two Python modules, the first one named `extending`, the second one named `physigym`.
```bash
cd ../PhysiCell
make clean data-cleanup reset  # ./config/PhysiCell_settings-backup.xml related Error can be ignored!
make list-user-projects
make load PROJ=physigym_template
make install-requirement  # only the first time needed to install the physigym python3 dependencies
make
```

7. Now you're good to go! Open a Python shell and type the following:
```python
import gymnasium
import physigym

env = gymnasium.make("physigym/ModelPhysiCellEnv-v0")
env.reset()
env.step(action={})
env.close()

exit()
```

8. Check out the [tutorial](TUTORIAL_physigym_model.md) to understand what you just ran.


## How to fetch the latest version from this PhysiCell user project into this source code repository

1. Save the project in the PhysiCell folder:
```bash
cd path/to/PhysiCell
make save PROJ=physigym_myproject
```

2. Fetch the project in to the PhysiGym folder and git:
```bash
cd ../PhysiGym
python3 capture_physigym.py myproject
git status
git diff
```
