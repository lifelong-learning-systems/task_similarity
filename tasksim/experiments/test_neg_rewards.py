import numpy as np

import tasksim
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim

if __name__ == '__main__':
    tasksim.init_ray()
    n = 5
    grid = gen.create_grid((n, n))
    G = gen.MDPGraph.from_grid(grid, success_prob=0.9, reward=1, noops=False)
    G_neg = gen.MDPGraph.from_grid(grid, success_prob=0.9, reward=-1, noops=False)
    print(f'G - G:\t\t{G.compare2(G)}')
    print(f'G_neg - G_neg:\t{G_neg.compare2(G_neg)}')
    print(f'G - G_neg:\t{G.compare2(G_neg)}')