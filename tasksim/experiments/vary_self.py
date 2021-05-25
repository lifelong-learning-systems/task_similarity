import numpy as np

import tasksim
from tasksim.gridworld_generator import create_grid, MDPGraph

def vary_self(grid, prob):
    G1 = MDPGraph.from_grid(grid, prob)
    G1_neg = MDPGraph.from_grid(grid, prob, reward=-1)
    G1_r2 = MDPGraph.from_grid(grid, prob, reward=2)
    G1_r3 = MDPGraph.from_grid(grid, prob, reward=3)
    G1_r100 = MDPGraph.from_grid(grid, prob, reward=100)
    G1_r1e6 = MDPGraph.from_grid(grid, prob, reward=1e6)

    prob2 = 0.75*prob
    G1_prob = MDPGraph.from_grid(grid, prob2)
    G1_prob_neg = MDPGraph.from_grid(grid, prob2, reward=-1)

    print(f'Varying w/ shape {grid.shape}, success prob {prob}:')
    print(f'\tSelf: {G1.compare2_norm(G1)}')
    print(f'\tNeg Reward: {G1.compare2_norm(G1_neg)}')
    print(f'\tLower Prob: {G1.compare2_norm(G1_prob)}')
    print(f'\tNeg Reward & Lower Prob: {G1.compare2_norm(G1_prob_neg)}')
    print(f'\tReward of 2: {G1.compare2_norm(G1_r2)}')
    print(f'\tReward of 3: {G1.compare2_norm(G1_r3)}')
    print(f'\tReward of 100: {G1.compare2_norm(G1_r100)}')
    print(f'\tReward of 1e6: {G1.compare2_norm(G1_r1e6)}')

if __name__ == '__main__':
    vary_self(create_grid((2, 2)), 0.9)
    vary_self(create_grid((2, 2)), 0.5)
    vary_self(create_grid((3, 3)), 0.9)
    vary_self(create_grid((9, 1)), 0.9)
    vary_self(create_grid((4, 4)), 0.9)