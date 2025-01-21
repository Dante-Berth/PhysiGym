## How to install the physigym PhysiCell user project

1. Download this repository *in the same folder where your PhysiCell is installed, right next to the PhysiCell folder, not into it*!
```bash
git clone https://github.com/Dante-Berth/PhysiGym
```

2. cd into the physigym folder and run the install_physigym.py script.
```bash
cd path/to/PhysiGym
python3 install_physigym.py template
```

3. If you are using environments, this is the time to activate the Python environment in which you would like to run physigym.

4. Check that you are hooked up to the internet because pip must be able to check for build dependencies.

5. cd into the PhysiCell folder, reset PhysiCell, load, and compile the physigym template project.
This will install two Python modules, the first one named `embedding`, the second one named `physigym`.
```bash
cd ../PhysiCell
make clean data-cleanup reset
make list-user-projects
make load PROJ=physigym_template
pip3 install --force-reinstall custom_modules/physigym   # optional to install and update dependencies
make
```

6. Now you're good to go! Open a Python shell and type the following:
```python
import gymnasium
import physigym

env = gymnasium.make('physigym/ModelPhysiCellEnv')
env.reset()
env.step(action={})
env.close()

exit()
```

7. Check out the [tutorial](https://github.com/Dante-Berth/PhysiGym/blob/main/man/TUTORIAL_physigym.md) to understand what you just ran.


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
