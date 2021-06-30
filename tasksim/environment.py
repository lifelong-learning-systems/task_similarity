import numpy as np
import gym
from gym import spaces

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim

class MDPGraphEnv(gym.Env):

    # TODO: don't take graph instance; take the grid, etc.
    def __init__(self, graph: gen.MDPGraph, random_state=None):
        super(MDPGraphEnv, self).__init__()
        self.graph = graph
        self.random_state = random_state if random_state is not None else np.random
        # TODO: need some sort of info on actions in MDPGraph; perhaps binary matrix of state by num_actions, with 1 if it's available, 0 if not
        #num_actions 
        # out_s[u] = [action_x, action_y] # you don't know what action_x, action_y correspond to in global [0, 1, 2, 3, 4] actions available

        # GOAL: want to have same action space available at all states:
        # When creating MDPGraph for metric can do:
        # 1. all states have a no-op action
        # 2. no states have no-op actions at all
        # 3. only actions which would violate "grid integrity" (out of bounds, into obstacle, etc.) function as no-ops
        #  a. all states have all actions available; some just function as no-op
        #  b. Should reward be associated with state? Reward is defined as (state, action, state') tuple in Wang paper; others are typically just (state, action) pairs (for deterministic MDPs)
        #     i. Easiest option: associate reward with state'; however you get there doesn't matter
        #     ii. TBD if needed
    
    def step():
        pass
    # Where do we start on the grid? Random location? Upper-left? Parameter of reset()?
    # -> effectively an epsilon transition from an augmented start state node into the rest of the MDP Graph
    # TODO: should our metric encompass start state? A set of start states? TBD once we get some results
    def reset(loc=None):
        pass
    def render():
        pass