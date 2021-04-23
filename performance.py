from matplotlib import pyplot as plt
import numpy as np
import timeit

from structural_similarity import structural_similarity
from gridworld_generator import grid_to_graph, parse_gridworld



# TODO: maybe generate obstacles with some percentage? For now, just zeros with center goal state
def create_grid(shape, obstacle_percent=0):
    rows, cols = shape
    grid = np.zeros(shape)
    grid[rows//2][cols//2] = 2
    return grid

if __name__ == '__main__':
    prob = 0.9
    graph_gen_repetitions = 100
    similarity_repetitions = 3 
    # have to go from at least a 2x2, since 1x1 with just goal state has zero actions...breaks things
    x = []
    y = []
    for i in range(1, 11):
        n = i + 1
        # speedup for later rounds
        if n > 5:
            similarity_repetitions = 1
        grid = create_grid((n, n))
        graph_gen_time = timeit.timeit(lambda: grid_to_graph(grid, prob),
                                       number=graph_gen_repetitions)
        graph_avg = graph_gen_time / graph_gen_repetitions
        print(f'{n}x{n}:')
        print(f'\tGraph generation, average of {graph_gen_repetitions}: {graph_avg:.3E}')
        P, R, out_s, out_a = grid_to_graph(grid, prob)
        similarity_time = timeit.timeit(lambda: structural_similarity(P, R, out_s),
                                        number=similarity_repetitions)
        similarity_avg = similarity_time / similarity_repetitions
        print(f'\tStructural similarity, average of {similarity_repetitions}: {similarity_avg:.3E}')
        x.append(n*n)
        y.append(similarity_avg)
    print('Scatter points:')
    print('\t' + str(x))
    print('\t' + str(y))
    plt.loglog(x, y)
    plt.show()