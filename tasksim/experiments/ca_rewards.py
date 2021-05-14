import numpy as np

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen

if __name__ == '__main__':
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    prob = 0.9

    shapes = [(1, 5), (1, 7), (1, 9)]
    for shape in shapes:
        print('SHAPE:', str(shape))
        grid = gen.create_grid(shape)

        # Negative rewards work....very weirdly
        reward_range = list(range(1, 6))
        graphs = [MDPGraph.from_grid(grid, prob, reward=r) for r in reward_range]

        c_s = 0.95
        ca_range = [0.1, 0.3, 0.5, 0.7, 0.9]


        for c_a in ca_range:
            print('CA:', c_a)
            diffs = np.zeros((len(reward_range), len(reward_range)))
            for i in range(len(reward_range)):
                for j in range(i, len(reward_range)):
                    diffs[i, j] = graphs[i].compare2_norm(graphs[j], c_a=c_a, c_s=c_s)
            print(diffs)
            print('\n')