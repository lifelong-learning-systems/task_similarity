from matplotlib import pyplot as plt
import numpy as np
import timeit

from tasksim import MDPGraph
from tasksim.structural_similarity import structural_similarity
from tasksim.gridworld_generator import create_grid

if __name__ == '__main__':
    prob = 0.9
    graph_gen_repetitions = 100
    similarity_repetitions = 3 
    # have to go from at least a 2x2, since 1x1 with just goal state has zero actions...breaks things
    x = []
    y = []
    for i in range(1, 5):
        n = i + 1
        # speedup for later rounds
        if n > 5:
            similarity_repetitions = 1
        grid = create_grid((n, n))
        graph_gen_time = timeit.timeit(lambda: MDPGraph.from_grid(grid, prob),
                                       number=graph_gen_repetitions)
        graph_avg = graph_gen_time / graph_gen_repetitions
        print(f'{n}x{n}:')
        print(f'\tGraph generation, average of {graph_gen_repetitions}: {graph_avg:.3E}')
        G = MDPGraph.from_grid(grid, prob)
        P, R, out_s, out_a = G.P, G.R, G.out_s, G.out_a
        similarity_time = timeit.timeit(lambda: structural_similarity(P, R, out_s),
                                        number=similarity_repetitions)
        similarity_avg = similarity_time / similarity_repetitions
        print(f'\tStructural similarity, average of {similarity_repetitions}: {similarity_avg:.3E}')
        x.append(n*n)
        y.append(similarity_avg)
    print('Scatter points:')
    print('\t' + str(x))
    print('\t' + str(y))
    print('Log slopes:')
    log_y = np.log(y)
    log_x = np.log(x)
    log_slopes = [(log_y[i] - log_y[i-1])/(log_x[i] - log_x[i - 1]) for i in range(1, len(log_y))]
    print('\t' + str(log_slopes))
    plt.loglog(x, y)
    plt.show()