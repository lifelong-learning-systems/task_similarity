"""
Code to parse gridworlds into MDP graphs from file or memory.
Similar to Wang 2019, but:
- state 0 is upper-left, not lower-left
- there is a 5th action, no-op, available
- obstacles are supported as well
"""
import numpy as np
import ot

import tasksim
import tasksim.structural_similarity as sim
from tasksim import DEFAULT_CA, DEFAULT_CS

import argparse
import time

from enum import Enum
from collections import namedtuple

# Strategy for available actions; k = # of actions at each state
class ActionStrategy(Enum):
    SUBSET = 1 # default: 0 <= k <= 4
    NOOP_ACTION = 2 # all states have no-op action added: 1 <= k <= 5
    NOOP_EFFECT = 3 # all states have all actions available, some just function as noops: k = 4
    WRAP_SUBSET = 4 # SUBSET, but wraparound at borders
    WRAP_NOOP_ACTION = 5 # NOOP_ACTION, ""
    WRAP_NOOP_EFFECT = 6 # NOOP_EFFECT, ""
    SUBSET_COMPRESS = 7 # Subset, plus remove obstacles entirely from state graph
    NOOP_EFFECT_COMPRESS = 8 # Subset, plus remove obstacles entirely from state graph

WRAP_STRATS = [ActionStrategy.WRAP_SUBSET, ActionStrategy.WRAP_NOOP_ACTION, ActionStrategy.WRAP_NOOP_EFFECT]
COMPRESS_STRATS = [ActionStrategy.SUBSET_COMPRESS, ActionStrategy.NOOP_EFFECT_COMPRESS]

# Simple wrapper class of (P, R, out_s, out_a)
class MDPGraph:

    def __init__(self, P, R, out_s, out_a, available_actions, grid, strat, states_to_grid, grid_to_states):
        self.P = P.copy()
        self.R = R.copy()
        self.out_s = out_s.copy()
        self.out_a = out_a.copy()
        self.available_actions = available_actions.copy()
        self.grid = grid.copy()
        self.states_to_grid = states_to_grid.copy()
        self.grid_to_states = grid_to_states.copy()
        self.strat = strat

    @classmethod
    def from_grid(cls, grid, success_prob=0.9, reward=1, strat=ActionStrategy.NOOP_ACTION):
        P, R, out_s, out_a, available_actions, states_to_grid, grid_to_states  = cls.grid_to_graph(grid, success_prob, reward=reward, strat=strat)
        return cls(P, R, out_s, out_a, available_actions, grid, strat, states_to_grid, grid_to_states)

    # 5 moves: left = 0, right = 1, up = 2, down = 3, no-op = 4
    @classmethod
    def get_valid_adjacent(cls, state, grid, strat: ActionStrategy):
        height, width = grid.shape
        row = state // width
        col = state - width*row

        # left, right, up, down, no-op/stay
        moves = [None] * 5

        # no-op action added?
        if strat == ActionStrategy.NOOP_ACTION or strat == ActionStrategy.WRAP_NOOP_ACTION:
            moves[4] = state

        # start wtih no-op effect for all actions
        if strat == ActionStrategy.NOOP_EFFECT or strat == ActionStrategy.WRAP_NOOP_EFFECT or strat == ActionStrategy.NOOP_EFFECT_COMPRESS:
            moves[0] = moves[1] = moves[2] = moves[3] = state
        
        # obstacles and goal states have no out neighbors (besides no-op)
        if grid[row][col] != 0:
            return moves
        
        # should wrap around or not
        wrap = (strat in WRAP_STRATS)

        # left
        if col > 0 and grid[row][col - 1] != 1:
            moves[0] = state - 1
        elif wrap and col == 0 and grid[row][width - 1] != 1:
            moves[0] = state + width - 1
        # right
        if col < width - 1 and grid[row][col + 1] != 1:
            moves[1] = state + 1
        elif wrap and col == width - 1 and grid[row][0] != 1:
            moves[1] = state - width + 1
        # up
        if row > 0 and grid[row - 1][col] != 1:
            moves[2] = state - width
        elif wrap and row == 0 and grid[height - 1][col] != 1:
            moves[2] = width * (height - 1) + col
        # down
        if row < height - 1 and grid[row + 1][col] != 1:
            moves[3] = state + width
        elif wrap and row == height - 1 and grid[0][col] != 1:
            moves[3] = col

        return moves

    @classmethod
    def grid_to_graph(cls, grid, success_prob=0.9, reward=1, strat=ActionStrategy.NOOP_ACTION):
        height, width = grid.shape
        goal_states = []
        num_states = 0
        states_to_grid = {}
        grid_to_states = {}
        for i in range(height):
            for j in range(width):
                grid_to_states[width*i + j] = np.nan
                if grid[i][j] == 1 and strat in COMPRESS_STRATS:
                    continue
                if grid[i][j] == 2:
                    goal_states.append(num_states)
                states_to_grid[num_states] = i*width + j
                grid_to_states[width*i + j] = num_states
                num_states += 1
        # COMPRESS out obstacles if needed
        out_neighbors_s = dict(zip(range(num_states), 
                                    [np.empty(shape=(0,), dtype=int) for i in range(num_states)]))
        out_neighbors_a = dict()

        num_grid_actions = 5
        a_node = 0
        available_actions = np.zeros((num_states, num_grid_actions))
        for s in range(num_states):
            actions = cls.get_valid_adjacent(states_to_grid[s], grid, strat)
            filtered_actions = [grid_to_states[a] for a in actions if a is not None]
            available_actions[s, :] = [1 if a is not None else 0 for a in actions]
            for action_id, action in enumerate(filtered_actions):
                # each action a state can do creates an action node
                out_neighbors_s[s] = np.append(out_neighbors_s[s], a_node)
                # action nodes initialized with zero transition prob to all other states
                out_neighbors_a[a_node] = np.zeros(num_states)
                # account for where the agent actually goes
                if len(filtered_actions) == 1:
                    success = 1
                    fail = 0
                else:
                    success = success_prob
                    fail = (1 - success_prob) / (len(filtered_actions) - 1)
                for s_p_id, s_p in enumerate(filtered_actions):
                    if s_p_id == action_id:
                        out_neighbors_a[a_node][s_p] += success
                    else:
                        out_neighbors_a[a_node][s_p] += fail
                states = np.array(filtered_actions, dtype=int) 
                probs = np.array(out_neighbors_a[a_node][states])
                prob_sorted = np.argsort(probs)
                states = states[prob_sorted]
                probs = probs[prob_sorted]
                a_node += 1
            
        num_actions = a_node
        P = np.array([out_neighbors_a[i] for i in range(num_actions)])
        R = np.zeros((num_actions, num_states))
        for i in range(num_actions):
            for g in goal_states:
                if P[i, g] > 0:
                    R[i, g] = reward
        return P, R, out_neighbors_s, out_neighbors_a, available_actions, states_to_grid, grid_to_states
    
    def copy(self):
        return MDPGraph(self.P, self.R, self.out_s, self.out_a, self.available_actions, self.grid, self.strat, self.states_to_grid, self.grid_to_states)

    # returns upper right of structural similarity computed on appended graphs
    def compare(self, other, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
        P1, P2, R1, R2, out_s1, out_s2 = self.P, other.P, self.R, other.R, self.out_s, other.out_s
        return sim.cross_structural_similarity(P1, P2, R1, R2, out_s1, out_s2, c_a=c_a, c_s=c_s)

    def compare_song(self, other, c=DEFAULT_CA):
        return sim.cross_structural_similarity_song(self.P, other.P, self.R, other.R, self.out_s, other.out_s, 
                                                    self.available_actions, other.available_actions, c=c)
    # using convention from POT
    def compare2(self, other, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
        return sim.final_score(self.compare(other, c_a, c_s))
    def compare2_song(self, other, c=DEFAULT_CA):
        return sim.final_score_song(self.compare_song(other, c))
    
    def compare2_norm(self, other, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
        return sim.normalize_score(self.compare2(other, c_a, c_s), c_a, c_s)


# Goal locations: list of tuples
# Goal rewards: list of floats
def create_grid(shape, goal_locations=None, obstacle_locations=None, obstacle_prob=0, random_state=None):
    if random_state is None:
        random_state = np.random
    rows, cols = shape
    grid = np.zeros(shape)
    if not goal_locations:
        goal_locations = [(rows//2, cols//2)]
    assert len(goal_locations)
    if not obstacle_locations:
        obstacle_locations = []
    for i in range(rows):
        for j in range(cols):
            loc = (i, j)
            if (i, j) in goal_locations:
                grid[i][j] = 2
                continue
            if (i, j) in obstacle_locations:
                grid[i][j] = 1
                continue
            grid[i][j] = 0

            # potentially turn into a 1, seeded
            if random_state.rand() < obstacle_prob:
                grid[i][j] = 1
    return grid

def compare_graphs(G1, G2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return G1.compare(G2, c_a, c_s)
def compare_graphs2(G1, G2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return G1.compare2(G2, c_a, c_s)
def compare_graphs2_norm(G1, G2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return G1.compare2_norm(G2, c_a, c_s)

def compare_files(file1, file2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return compare_graphs(MDPGraph.from_file(file1), MDPGraph.from_file(file2), c_a, c_s)
def compare_files2(file1, file2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return sim.final_score(compare_files(file1, file2, c_a, c_s))
def compare_files2_norm(file1, file2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return sim.normalize_score(compare_files2(file1, file2, c_a, c_s), c_a, c_s)

def compare_shapes(shape1, shape2, success_prob1=0.9, success_prob2=0.9, c_a=DEFAULT_CA, c_s=DEFAULT_CS, strat=ActionStrategy.NOOP_ACTION):
    return compare_graphs(MDPGraph.from_grid(create_grid(shape1), success_prob1, strat=strat), \
                          MDPGraph.from_grid(create_grid(shape2), success_prob2, strat=strat), c_a=c_a, c_s=c_s)
def compare_shapes2(shape1, shape2, success_prob1=0.9, success_prob2=0.9, c_a=DEFAULT_CA, c_s=DEFAULT_CS, strat=ActionStrategy.NOOP_ACTION):
    return sim.final_score(compare_shapes(shape1, shape2, success_prob1, success_prob2, c_a, c_s, strat=strat))
def compare_shapes2_norm(shape1, shape2, success_prob1=0.9, success_prob2=0.9, c_a=DEFAULT_CA, c_s=DEFAULT_CS, strat=ActionStrategy.NOOP_ACTION):
    return sim.normalize_score(compare_shapes2(shape1, shape2, success_prob1, success_prob2, c_a, c_s, strat=strat), c_a, c_s)