import numpy as np

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen

if __name__ == '__main__':
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    sizes = range(1, 10, 1)
    comparisons = np.zeros((len(sizes), len(sizes)))
    for i, s1 in enumerate(sizes):
        for j, s2 in enumerate(sizes):
            if j < i:
                continue
            comparisons[i][j] = gen.compare_shapes2_norm((1, s1), (1, s2))
    print(list(sizes))
    print(comparisons)
