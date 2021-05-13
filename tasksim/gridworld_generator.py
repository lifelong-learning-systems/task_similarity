"""
Code to parse gridworlds into MDP graphs from file or memory.
Similar to Wang 2019, but:
- state 0 is upper-left, not lower-left
- there is a 5th action, no-op, available
- obstacles are supported as well
"""
import numpy as np
import ot
from scipy.linalg import block_diag

import tasksim
import tasksim.structural_similarity as sim
import tasksim.chace_structural_similarity as chace_sim
from tasksim import DEFAULT_CA, DEFAULT_CS

import argparse
import time

# Simple-ish wrapper class of (P, R, out_s, out_a)
class MDPGraph:

    def __init__(self, P, R, out_s, out_a):
        self.P = P.copy()
        self.R = R.copy()
        self.out_s = out_s.copy()
        self.out_a = out_a.copy()

    @classmethod
    def from_file(cls, path):
        grid, success_prob = parse_gridworld(path=path)
        return cls.from_grid(grid, success_prob)

    @classmethod
    def from_grid(cls, grid, success_prob):
        P, R, out_s, out_a = cls.grid_to_graph(grid, success_prob)
        return cls(P, R, out_s, out_a)

    # 5 moves: left = 0, right = 1, up = 2, down = 3, no-op = 4
    @classmethod
    def get_valid_adjacent(cls, state, grid):
        height, width = grid.shape
        row = state // width
        col = state - width*row

        moves = [None] * 5
        # no-op
        moves[4] = state
        # obstacles and goal states have no out neighbors (besides no-op)
        if grid[row][col] != 0:
            return moves
        # left
        if col > 0 and grid[row][col - 1] != 1:
            moves[0] = state - 1
        # right
        if col < width - 1 and grid[row][col + 1] != 1:
            moves[1] = state + 1
        # up
        if row > 0 and grid[row - 1][col] != 1:
            moves[2] = state - width
        # down
        if row < height - 1 and grid[row + 1][col] != 1:
            moves[3] = state + width
        return moves

    @classmethod
    def grid_to_graph(cls, grid, success_prob=0.9):
        height, width = grid.shape
        goal_states = []
        for i in range(height):
            for j in range(width):
                if grid[i][j] == 2:
                    goal_states.append(i*width + j)
        assert len(goal_states) >= 1, 'At least one goal state required'
        num_states = height * width
        out_neighbors_s = dict(zip(range(num_states), 
                                    [np.empty(shape=(0,), dtype=int) for i in range(num_states)]))
        out_neighbors_a = dict()

        a_node = 0
        for s in range(num_states):
            actions = cls.get_valid_adjacent(s, grid)
            filtered_actions = [a for a in actions if a is not None]
            # assert len(filtered_actions) == 0 or len(filtered_actions) >= 2, \
            #         'Invalid actions; must be either zero or at least 2 (action + no-op)'

            for action in filtered_actions:
                # each action a state can do creates an action node
                out_neighbors_s[s] = np.append(out_neighbors_s[s], a_node)
                # action nodes initialized with zero transition prob to all other states
                out_neighbors_a[a_node] = np.zeros(num_states)
                # TODO: maybe add a random seed in txt file + noise on the probabilities?
                # account for where the agent actually goes
                if len(filtered_actions) == 1:
                    success = 1
                    fail = 0
                else:
                    success = success_prob
                    fail = (1 - success_prob) / (len(filtered_actions) - 1)
                for s_p in filtered_actions:
                    if s_p == action:
                        out_neighbors_a[a_node][s_p] = success
                    else:
                        out_neighbors_a[a_node][s_p] = fail
                a_node += 1
        
        num_actions = a_node
        P = np.array([out_neighbors_a[i] for i in range(num_actions)])
        R = np.zeros((num_actions, num_states))
        for i in range(num_actions):
            for g in goal_states:
                if P[i, g] > 0:
                    R[i, g] = 1.0
        return P, R, out_neighbors_s, out_neighbors_a


    # G = (P, R, out_s, out_a) tuple
    def append(self, other):
        P1, R1, out_s1, out_a1 = self.P, self.R, self.out_s, self.out_a
        P2, R2, out_s2, out_a2 = other.P, other.R, other.out_s, other.out_a
        num_actions1, num_states1 = P1.shape
        num_actions2, num_states2 = P2.shape

        total_actions = num_actions1 + num_actions2
        total_states = num_states1 + num_states2
        P = block_diag(P1, P2)
        R = block_diag(R1, R2)

        out_s = dict()
        for i in range(total_states):
            if i < num_states1:
                out_s[i] = out_s1[i].copy()
            else:
                out_s[i] = out_s2[i - num_states1] + num_actions1
        out_a = dict()
        for i in range(total_actions):
            if i < num_actions1:
                out_a[i] = out_a1[i].copy()
            else:
                out_a[i] = out_a2[i - num_actions1].copy()
        return MDPGraph(P, R, out_s, out_a)


    # returns upper right of structural similarity computed on appended graphs
    def compare(self, other, c_a=DEFAULT_CA, c_s=DEFAULT_CS, append=False, chace=False):
        if chace:
            return chace_sim.structural_similarity(self.P, other.P, self.R, other.R, self.out_s, other.out_s, c_a=c_a, c_s=c_s)
        if append:
            G = self.append(other)
            S, A, num_iters, done = sim.structural_similarity(G.P, G.R, G.out_s, c_a=c_a, c_s=c_s)
            S_upper_right = S[0:self.P.shape[1], self.P.shape[1]:]
            A_upper_right = A[0:self.P.shape[0], self.P.shape[0]:]
            return S_upper_right, A_upper_right, num_iters, done
        return sim.cross_structural_similarity(self.P, other.P, self.R, other.R, self.out_s, other.out_s, c_a=c_a, c_s=c_s)

    # using convention from POT
    def compare2(self, other, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
        return sim.final_score(self.compare(other, c_a, c_s))
    
    def compare2_norm(self, other, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
        return sim.normalize_score(self.compare2(other, c_a, c_s), c_a, c_s)


# parsing similar to Wang 2019, but
# top down, left to right
# TODO: change assertions to throws
def parse_gridworld(path='./gridworlds/experiment1.txt'):
    valid_chars = set(['0', '1', '2'])
    with open(path) as f:
        lines = f.read().splitlines()

    success_prob = float(lines[0])
    assert 0 <= success_prob and success_prob <= 1, 'Success prob must be between 0 and 1'

    height, width = np.array(lines[1].split(' ')).astype(int)
    assert height > 0 and width > 0, 'Must have positive height and width'
    grid = np.zeros((height, width)).astype(int)
    assert height == len(lines) - 2, 'Incorrect number of rows for specified height'
    # remove first 2 lines, which had success prob & dimensions
    lines = lines[2:]
    for i in range(len(lines)):
        line = lines[i]
        assert width == len(line), 'Incorrect number of cols for specified width'
        for j, char in enumerate(line):
            assert char in valid_chars, 'Invalid character'
            grid[i][j] = int(char)
    return grid, success_prob

# TODO: also see what happens for multiple goal states
# TODO: non-grid world extensions??
def create_grid(shape, obstacle_prob=0, random_state=np.random):
    rows, cols = shape
    grid = np.zeros(shape)
    for i in range(rows):
        for j in range(cols):
            if rows//2 == i and cols//2 == j:
                grid[i][j] = 2
                continue
            grid[i][j] = 0

            # potentially turn into a 1, seeded
            if random_state.rand() < obstacle_prob:
                grid[i][j] = 1
    return grid

def append_graphs(G1, G2):
    return G1.append(G2)

def compare_graphs(G1, G2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return G1.compare(G2, c_a, c_s)
def compare_graphs2(G1, G2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return G1.compare2(G2, c_a, c_s)
def compare_graphs2_norm(G1, G2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return G1.compare2_norm(G2, c_a, c_s)

# TODO: probably do the whole kwargs thing idk instead of always using these defaults
def compare_files(file1, file2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return compare_graphs(MDPGraph.from_file(file1), MDPGraph.from_file(file2), c_a, c_s)
def compare_files2(file1, file2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return sim.final_score(compare_files(file1, file2, c_a, c_s))
def compare_files2_norm(file1, file2, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return sim.normalize_score(compare_files2(file1, file2, c_a, c_s), c_a, c_s)

def compare_shapes(shape1, shape2, success_prob1=0.9, success_prob2=0.9, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return compare_graphs(MDPGraph.from_grid(create_grid(shape1), success_prob1), \
                          MDPGraph.from_grid(create_grid(shape2), success_prob2), c_a=c_a, c_s=c_s)
def compare_shapes2(shape1, shape2, success_prob1=0.9, success_prob2=0.9, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return sim.final_score(compare_shapes(shape1, shape2, success_prob1, success_prob2, c_a, c_s))
def compare_shapes2_norm(shape1, shape2, success_prob1=0.9, success_prob2=0.9, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    return sim.normalize_score(compare_shapes2(shape1, shape2, success_prob1, success_prob2, c_a, c_s), c_a, c_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', help='gridworld file 1 to read in')
    parser.add_argument('--file2', help='gridworld file 2 to read in')
    args = parser.parse_args()

    file1 = 'gridworlds/3x3_base.txt'
    file2 = 'gridworlds/5x5_base.txt'

    if args.file1:
        file1 = args.file1
    if args.file2:
        file2 = args.file2
    
    # S, A, num_iters, done = compare_files(file1, file2)
    # score = sim.final_score(S)
    # norm_score = sim.normalize_score(score)
    # print(score, norm_score)
    #assert score == compare_files2(file1, file2)
    #assert norm_score == compare_files2_norm(file1, file2)
    import tasksim.chace_structural_similarity as css
    G1 = MDPGraph.from_file(file1)
    G2 = MDPGraph.from_file(file2)

    c_s = 0.995
    c_a = 0.5

    time1 = time.time()
    S, A, num_iters, done = G1.compare(G2, c_a, c_s)
    print('Time:', time.time() - time1)
    score = sim.final_score(S)
    norm_score = sim.normalize_score(score)
    print('Score:', score) 
    print('Normalized:', norm_score)

