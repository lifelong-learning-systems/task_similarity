import numpy as np
import gym
from gym import spaces

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim


class MDPGraphEnv(gym.Env):

    def __init__(self, graph: gen.MDPGraph, obs_size=7, do_render=True, random_state=None, fixed_start=None):
        super(MDPGraphEnv, self).__init__()
        self.fixed_start = fixed_start
        self.do_render = do_render
        self.random_state = random_state if random_state is not None else np.random
        supported_strats = [gen.ActionStrategy.NOOP_EFFECT, gen.ActionStrategy.WRAP_NOOP_EFFECT, gen.ActionStrategy.NOOP_EFFECT_COMPRESS]
        assert graph.strat in supported_strats, f'Unsupported ActionStrategy: {graph.strat}'
        assert obs_size >= 3 and obs_size % 2 == 1, f'Obs Size must be an odd number >= 3'
        self.graph = graph

        self.action_space = spaces.Discrete(4)
        self.obs_size = obs_size
        self.grid = graph.grid
        self.height, self.width = self.grid.shape
        # 0 = normal, 1 = obstacle, 2 = goal state, 3 = current location, 4 = inaccessible (if not wraparound)/not visible
        self.observation_space = spaces.Box(low=0, high=4, shape=(obs_size, obs_size, ), dtype=np.uint8)
        # e.g. simple 3x3, upper-left:
        # 4 4 4
        # 4 3 0
        # 4 0 0

        # then, after moving left and wrapping around
        # 4 4 4
        # 0 3 4
        # 0 0 4

    def state_to_grid(self, state):
        return self.graph.states_to_grid[state]
    
    def grid_to_state(self, grid_s):
        return self.graph.grid_to_states[grid_s]

    # STATE here refers to grid state!
    def reset(self, state=None, center=True):
        if state is None:
            valid_states = np.where(self.grid.flatten() == 0)[0]
            assert len(valid_states) > 0, 'No possible start states!'
            state = self.random_state.choice(valid_states) if self.fixed_start is None else self.fixed_start
            #assert state in valid_states, 'cannot start in specified state'
        row, col = self.row_col(state)
        assert 0 <= row < self.height and 0 <= col < self.width, 'Invalid start state: out of grid'
        assert self.grid[row, col] == 0, 'Invalid start state: non-zero grid entry'
        self.state = state
        return self.gen_obs(center=center)

    def gen_obs(self, center=True):
        if not self.do_render:
            return None
        if center:
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
        else:
            # put in upper left corner for easiness
            base_grid = 4*np.ones((self.obs_size, self.obs_size))
            height, width = self.grid.shape
            assert height <= self.obs_size and width <= self.obs_size, 'invalid grid size non-centered'
            base_grid[0:height, 0:width] = self.grid
            row, col = self.row_col(self.state)
            base_grid[row, col] = 3
            return base_grid


    # Recall, state is grid state
    def step(self, action):
        mdp_state = self.grid_to_state(self.state)
        graph_action = self.graph.out_s[mdp_state][action]
        transitions = self.graph.P[graph_action]
        indices = np.array([i for i in range(len(transitions))])

        new_mdp_state = self.random_state.choice(indices, p=transitions)
        reward = self.graph.R[graph_action][new_mdp_state]

        self.state = self.state_to_grid(new_mdp_state)
        row, col = self.row_col(self.state)
        done = self.grid[row, col] == 2
        return self.gen_obs(), reward, done, {}

    def render(self):
        print(self.gen_obs())
    
    def copy(self):
        G = self.graph.copy()
        return MDPGraphEnv(G, obs_size=self.obs_size, random_state=self.random_state)

    def row_col(self, state):
        row = state // self.width
        col = state - self.width*row
        return row, col
    def flatten(self, row, col):
        return row*self.width + col