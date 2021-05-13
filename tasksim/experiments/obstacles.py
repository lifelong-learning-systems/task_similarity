import numpy as np

import tasksim
from tasksim import MDPGraph
import tasksim.structural_similarity as sim
import tasksim.gridworld_generator as gen

if __name__ == '__main__':
    random_state = np.random.RandomState(seed=314159)

    def permute_single(grid, success_prob):
        rows, cols = grid.shape
        def augment(grid, i, j):
            ret = grid.copy()
            if ret[i, j] != 2:
                ret[i, j] = 1
            return ret
        grids =  np.array([[augment(grid, i, j) for j in range(cols)] for i in range(rows)])
        graphs = np.array([[MDPGraph.from_grid(grids[i, j], success_prob) for j in range(cols)] for i in range(rows)])
        return grids, graphs
    
    success_prob = 0.9
    grids3, graphs3 = permute_single(gen.create_grid((3, 3)), success_prob)