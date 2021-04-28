"""
Code to parse gridworlds into MDP graphs from file or memory.
Similar to Wang 2019, but:
- state 0 is upper-left, not lower-left
- there is a 5th action, no-op, available
- obstacles are supported as well
"""
import numpy as np
from structural_similarity import structural_similarity
from scipy.linalg import block_diag

import argparse

# G = (P, R, out_s, out_a) tuple
def append_graphs(G1, G2):
    P1, R1, out_s1, out_a1 = G1
    P2, R2, out_s2, out_a2 = G2
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
    return P, R, out_s, out_a


# returns upper right of structural similarity computed on appended graphs
def cross_similarity(G1, G2):
    G = append_graphs(G1, G2)
    S, A, num_iters, done = structural_similarity(G[0], G[1], G[2])
    S_upper_right = S[0:G1[0].shape[1], G1[0].shape[1]:]
    A_upper_right = A[0:G1[0].shape[0], G1[0].shape[0]:]
    return S_upper_right, A_upper_right, num_iters, done

# 5 moves: left = 0, right = 1, up = 2, down = 3, no-op = 4
def get_valid_adjacent(state, grid):
    height, width = grid.shape
    row = state // width
    col = state - width*row

    moves = [None] * 5
    # obstacles and goal states have no out neighbors
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
    # no-op
    moves[4] = state
    return moves


def grid_to_graph(grid, success_prob=0.9):
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
        actions = get_valid_adjacent(s, grid)
        filtered_actions = [a for a in actions if a is not None]
        assert len(filtered_actions) == 0 or len(filtered_actions) >= 2, \
                'Invalid actions; must be either zero or at least 2 (action + no-op)'

        for action in filtered_actions:
            # each action a state can do creates an action node
            out_neighbors_s[s] = np.append(out_neighbors_s[s], a_node)
            # action nodes initialized with zero transition prob to all other states
            out_neighbors_a[a_node] = np.zeros(num_states)
            # TODO: maybe add a random seed in txt file + noise on the probabilities?
            # account for where the agent actually goes
            fail_prob = (1 - success_prob) / (len(filtered_actions) - 1)
            for s_p in filtered_actions:
                if s_p == action:
                    out_neighbors_a[a_node][s_p] = success_prob
                else:
                    out_neighbors_a[a_node][s_p] = fail_prob
            a_node += 1
    
    num_actions = a_node
    P = np.array([out_neighbors_a[i] for i in range(num_actions)])
    R = np.zeros((num_actions, num_states))
    for i in range(num_actions):
        for g in goal_states:
            if P[i, g] > 0:
                R[i, g] = 1.0
    return P, R, out_neighbors_s, out_neighbors_a

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='gridworld file to read in')
    args = parser.parse_args()

    file = 'gridworlds/3x3_base.txt'
    if args.file is not None:
        file = args.file
    
    grid, success_prob = parse_gridworld(file)
    # grid = np.zeros((5, 1))
    # grid[2] = 2
    # success_prob = 0.9
    G1 = grid_to_graph(grid, success_prob)
    G2 = grid_to_graph(grid, success_prob)
    S, A, num_iters, done = cross_similarity(G1, G2)