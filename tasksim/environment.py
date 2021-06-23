import numpy as np
import gym
from gym import spaces

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim

class MDPGraphEnv(gym.Env):

    def __init__(self, graph: gen.MDPGraph, random_state=None):
        super(MDPGraphEnv, self).__init__()
        self.graph = graph
        self.random_state = random_state if random_state is not None else np.random
        # TODO: need some sort of info on actions in MDPGraph; perhaps binary matrix of state by num_actions, with 1 if it's available, 0 if not
        #num_actions 