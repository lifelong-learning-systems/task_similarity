# A repository for Task Similarity reserach under L2M

## Install

- First create a venv:
    - `python -m venv venv-task-sim`
    - `source venv-task-sim/bin/activate`
- Next, install POT from source (required so that Cythonized function have access to low level wrappers):
    - `git clone https://github.com/BenStoler/POT.git`
    - `cd POT`
    - `pip install -e .`
    - `cd ..`
- Now install the `tasksim` package from source:
    - `git clone <TASK_SIMILARITY_REPO_URL>`
    - `cd task_similarity`
    - `pip install -e .`

## Usage

- Run experiments in the `tasksim/experiments` folder!
- Create custom experiments via the provided utils
    - `tasksim.structural_similarity`
    - `tasksim.gridworld_generator`


