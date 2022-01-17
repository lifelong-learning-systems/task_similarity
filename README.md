# A repository for Task Similarity reserach under L2M

## Install

- First create a venv:
    - `python -m venv venv-task-sim`
    - `source venv-task-sim/bin/activate`
- Next, install POT from source (required so that Cythonized function have access to low level wrappers):
    - `git clone https://github.com/BenStoler/POT.git`
    - `cd POT`
    - Install the following base dependencies of POT:
        - `pip install numpy Cython`
    - `pip install -e .`
    - `cd ..`
- Now install the `tasksim` package from source:
    - `git clone <TASK_SIMILARITY_REPO_URL>`
    - `cd task_similarity`
    - `pip install -e ./tasksim/curriculum_tools`
    - `pip install -e .`
- Additionally install `"ray[default]"`:
    - `pip install "ray[default]`
- If on Mac, may need to configure a signing identity to allow connections:
    - Follow instructions [here](https://stackoverflow.com/questions/19688841/add-python-application-to-accept-incoming-network-connections/21052159#21052159)
        1. create signing identity
        2. with venv activated, run:
           - `codesign -s "My Signing Identity" -f $(which python)`

## Usage

- Run experiments in the `tasksim/experiments` folder!
- Create custom experiments via the provided utils
    - `tasksim.structural_similarity`
    - `tasksim.gridworld_generator`
- To avoid initializing Ray on every invocation:
    - In command line, first run:
        - `ray start --head --num-cpus=<NUM_CPUS>`
    - Then bring down later with:
        - `ray stop --force`
    - You can also initialize ray explicitly, via:
        - `tasksim.init_ray()`
        - If not used, then `init_ray()` will be invoked when it is lazily first needed

## Transfer Experimnents
- Setup via the `create_bases.py` command.
    - Run as `python create_bases.py --restore --results agent_bases`
    - Using the `restore` flag allows future invocation to add to what was left off, if it was interrupted
    - End the session at any time by doing `killall Python; killall python` in a separate session
    - This will create the base agents for all the environment variations, as well as store the grid 
      information and distance matrices for the relavant metrics.
- Afterwards, use the `transfer_runner1.py` script to kick off the transfer experiments
    - Run as `python transfer_runner1.py --results nested_results/rot_dim9_reward100_num100 --rotate --dim 9 --reward 100 --num 100`
    - This copies over the information from the `agent_bases` folder into new experiment folders, for all the
      transfer methods
- Finally, invoke the `plot_transfer.py` script to get graphs and output information
    - `python plot_transfer2.py --parent nested_results/rot_dim9_reward100_num100`


