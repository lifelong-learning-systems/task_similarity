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
        self.random_state = random_state if random_state is not None else np.random
        supported_strats = [gen.ActionStrategy.NOOP_EFFECT, gen.ActionStrategy.WRAP_NOOP_EFFECT]
        # TODO: change from assertion to throw? Or just don't take graph instance; take grid, prob, etc. instead
        assert graph.strat in supported_strats, f'Unsupported ActionStrategy: {graph.strat}'
        self.graph = graph

        self.action_space = spaces.Discrete(4)
        self.grid = graph.grid
        self.height, self.width = self.grid.shape
        # TODO: change dtype?
        # 0 = normal, 1 = obstacle, 2 = goal state, 3 = current location
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.height, self.width), dtype=int)

    # Where do we start on the grid? Random location? Upper-left? Parameter of reset()?
    # -> effectively an epsilon transition from an augmented start state node into the rest of the MDP Graph
    # TODO: should our metric encompass start state? A set of start states? TBD once we get some results
    # Default: random, any non-goal state
    def reset(self, state=None):
        if state is None:
            valid_states = np.where(self.grid.flatten() == 0)[0]
            assert len(valid_states) > 0, 'No possible start states!'
            state = self.random_state.choice(valid_states)
        row, col = self.row_col(state)
        assert 0 <= row < self.height and 0 <= col < self.width, 'Invalid start state: out of grid'
        assert self.grid[row, col] == 0, 'Invalid start state: non-zero grid entry'
        self.state = state
        return self.gen_obs()

    # TODO: should observation be whole grid? or just a local snapshot, etc.
    def gen_obs(self):
        base_grid = self.grid.copy()
        row, col = self.row_col(self.state)
        base_grid[row, col] = 3
        return base_grid

    def step(self, action):
        graph_action = self.graph.out_s[self.state][action]
        transitions = self.graph.P[graph_action]
        indices = np.array([i for i in range(len(transitions))])
        state = self.random_state.choice(indices, p=transitions)
        self.state = state
        reward = self.graph.R[graph_action][state]
        row, col = self.row_col(state)
        done = self.grid[row, col] == 2
        return self.gen_obs(), reward, done, {}

    def render(self):
        print(self.gen_obs())

    def row_col(self, state):
        row = state // self.width
        col = state - self.width*row
        return row, col
    def flatten(self, row, col):
        return row*self.width + col