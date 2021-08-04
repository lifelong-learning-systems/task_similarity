import numpy as np
import gym
from gym import spaces

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim


class MDPGraphEnv(gym.Env):

    # TODO: how does POMDP impact the metric?
    def __init__(self, graph: gen.MDPGraph, obs_size=7, random_state=None):
        super(MDPGraphEnv, self).__init__()
        self.random_state = random_state if random_state is not None else np.random
        supported_strats = [gen.ActionStrategy.NOOP_EFFECT, gen.ActionStrategy.WRAP_NOOP_EFFECT]
        # TODO: change from assertion to throw? Or just don't take graph instance; take grid, prob, etc. instead
        assert graph.strat in supported_strats, f'Unsupported ActionStrategy: {graph.strat}'
        assert obs_size >= 3 and obs_size % 2 == 1, f'Obs Size must be an odd number >= 3'
        self.graph = graph

        self.action_space = spaces.Discrete(4)
        self.obs_size = obs_size
        self.grid = graph.grid
        self.height, self.width = self.grid.shape
        # TODO: stack of past N frames? like in arcade type enviros
        # 0 = normal, 1 = obstacle, 2 = goal state, 3 = current location, 4 = inaccessible (if not wraparound)/not visible
        self.observation_space = spaces.Box(low=0, high=4, shape=(obs_size, obs_size), dtype=np.uint8)
        # e.g. simple 3x3, upper-left:
        # 4 4 4
        # 4 3 0
        # 4 0 0

        # then, after moving left and wrapping around
        # 4 4 4
        # 0 3 4
        # 0 0 4

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
    # TODO: limited observation size
    # TODO: limited observation lag behind/around (like in Mario) or agent always center but "black out" other squares? Or see wrap-around (if available)
    def gen_obs(self):
        # base_grid = self.grid.copy()
        # row, col = self.row_col(self.state)
        # base_grid[row, col] = 3
        # return base_grid
        row, col = self.row_col(self.state)
        base_grid = 4*np.ones((self.obs_size, self.obs_size))
        center = self.obs_size // 2
        height, width = self.grid.shape
        lim_left = max(0, col - center)
        lim_right = min(width - 1, col + center)
        lim_up = max(0, row - center)
        lim_down = min(height - 1, row + center)
        vals = self.grid[lim_up:lim_down+1, lim_left:lim_right+1]
        # e.g. row, col = (2, 3)
        # center = 3
        # lim_left = 0, lim_right = 6
        # lim_up = 0, lim_down = 5
        delta_left, delta_right = lim_left - col, lim_right - col
        delta_up, delta_down = lim_up - row, lim_down - row
        base_grid[center+delta_up:center+delta_down+1, center+delta_left:center+delta_right+1] = vals
        base_grid[center, center] = 3
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