![APL Logo](_assets/apl_small_horizontal_blue.png)

# Structural Similarity for Two MDPs (SS2)

This Python module is the companion software for the Structural Similarity for Two MDPs (SS2)
Task similarity estimation described in "Structural Similarity for Improved Transfer 
in Reinforcement Learning." It consists of an implementation of the SS2 algorithm, with 
examples of how to represent the MDP inputs, code for generating gridworld navigation
tasks, and the code used run the experiments presented in the aforementioned paper. 

Assuming two finite Markov Decision Processes (MDPs), SS2 computes a distance, or similarity
measure, between each state in each MDP and every other state in both MDPs. It does this by
defining a graph structure of each MDP, where states and actions are nodes in the graph and
transition probabilities and rewards constitute the edges, and then iteratively computing
node-to-node similarity using Earth Mover's Distance (EMD) and the Hausdorff distance; 
please see the full paper for details. The result is a distance measure that provably
satisfies the properties of a metric for state-to-state and action-to-action comparisons. 


## Install

**Note: this has not been tested on Windows OS**

- First create a venv, using Python 3.8:
    - `python -m venv venv-task-sim`
    - `source venv-task-sim/bin/activate`
- Next, install POT from source (required so that Cythonized function have access to low level wrappers):
    - `git clone https://github.com/BenStoler/POT.git`
    - `cd POT`
    - Install the following base dependencies of POT:
        - `pip install numpy==1.21 Cython`
    - `pip install -e .`
    - `cd ..`
- Now install the `tasksim` package from source:
    - `git clone <TASK_SIMILARITY_REPO_URL>`
    - `cd task_similarity`
    - `pip install -e .`
- Additionally install `"ray[default]"`:
    - `pip install "ray[default]`
- If on Mac, may need to configure a signing identity to allow connections:
    - Follow instructions [here](https://stackoverflow.com/questions/19688841/add-python-application-to-accept-incoming-network-connections/21052159#21052159)
        1. create signing identity
        2. with venv activated, run:
           - `codesign -s "My Signing Identity" -f $(which python)`


## Transfer Experimnents
- In the `tasksim` folder:
- Setup via the `create_bases.py` command.
    - Run as `python create_bases.py --restore --results agent_bases`
    - Using the `restore` flag allows future invocation to add to what was left off, if it was interrupted
    - End the session at any time by doing `killall Python; killall python` in a separate session
    - This will create the base agents for all the environment variations, as well as store the grid 
      information and distance matrices for the relavant metrics.
- Afterwards, use the `all_transfer.py` script to kick off the transfer experiments
    - Run as `python all_transfer.py --results nested_results --restore`
    - This copies over the information from the `agent_bases` folder into new experiment folders, for all the
      transfer methods, then runs the experiments and creates the charts for each
- Finally, invoke the `plot_more.py` script to get the final graphs and output tables
    - `python plot_more.py --results nested_results`


## Other Usage

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


### License

This software is released open source under the [BSD 3-Clause License](LICENSE). 

### Acknowledgements

This work was funded by the DARPA Lifelong Learning 
Machines (L2M) Program. The views, opinions,
and/or findings expressed are those of the author(s) and
should not be interpreted as representing the official
views or policies of the Department of Defense or the
U.S. Government.