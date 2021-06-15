import numpy as np

import tasksim
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim
import timeit

from pyinstrument import Profiler
from collections import Counter

import scipy
import ot
import ot.lp

from .compare_lines import check_triangle_inequality, check_diffs

if __name__ == '__main__':
    #tasksim.init_ray()

    # profiler = Profiler(interval=0.0001)
    # profiler.start()
    # val = timeit.timeit(lambda: gen.compare_shapes_T((n, n), (n, n)), number=1)
    # print(val)
    # profiler.stop()

    # Results:
    # - Performance does NOT depend on the composition of x and y
    # - Performance ONLY depends on the length of x and y
    #    - in our case, len(x) == num_states1, len(y) == num_states2
    def test_wasserstein(n, strat='rand', method='scipy'):
        x = np.zeros(n)
        y = np.zeros(n)
        if strat == 'all':
            x = np.ones(n) * 1/n
            y = np.ones(n) * 1/n
        elif strat == 'middle':
            x[n//2] = 1
            y[n//2] = 1
        elif strat == 'some':
            x[0:3] = [0.3, 0.6, 0.1]
            y[-3:] = [0.4, 0.2, 0.4]
        elif strat == 'rand':
            x = np.random.rand(n)
            y = np.random.rand(n)
            x = x / np.linalg.norm(x, 1)
            y = y / np.linalg.norm(y, 1)
        if method == 'scipy':
            return scipy.stats.wasserstein_distance(x, y)
        else:
            return ot.emd2_1d(x, y, metric='euclidean')

    #T1 = gen.compare_shapes_T((n, n), (n, n), 0.9, 0.9)
    tasksim.init_ray()

    def compare_shapes2(shape1, shape2, p1=0.9, p2=0.9, metric='euclidean', c_r=0.5):
        #ot.tic()
        G1 = gen.MDPGraph.from_grid(gen.create_grid(shape1), p1)
        G2 = gen.MDPGraph.from_grid(gen.create_grid(shape2), p2)
        #ot.toc('Generating graphs: {}')
        T_sim = sim.compute_T_ray(G1, G2, metric=metric, c_r=c_r)
        return sim.final_score(T_sim)

    def gen_comps(n_start, n_stop, c_r):
        n_range = n_stop - n_start
        comps = np.zeros((n_range, n_range))
        for i in range(n_start, n_stop):
            for j in range(n_start, n_stop):
                #print(f'Shape: {i}x{i} vs. {j}x{j}')
                score = compare_shapes2((i, i), (j, j), metric='euclidean', c_r = c_r)
                #print(f'\tScore: {score:.3f}\n\tTime: {time:.3f}s')
                comps[i-n_start, j-n_start] = score
        return comps
    
    # n = 10
    # c_r = 0.5
    # comps = gen_comps(1, n + 1, c_r)
    # precision_check = 10
    # ans, counter = check_triangle_inequality(comps, precision_check)
    # if not ans:
    #     print(f'Triangle inequality check ({precision_check} decimals) failed at:', counter)
    # else:
    #     print(f'Triangle inequality check ({precision_check} decimals) passed!')
    
    # order = check_diffs(comps, precision_check)
    # if not order:
    #     print('Order check failed')
    # else:
    #     print('Order check passed!')
    # num_print_decimals = 4
    # np.set_printoptions(linewidth=120, precision=num_print_decimals)
    # print(comps)

    # G1 = gen.MDPGraph.from_grid(gen.create_grid((10, 10)), 0.9)
    # G2 = gen.MDPGraph.from_grid(gen.create_grid((11, 11)), 0.9)
    # num_actions1, num_states1 = G1.P.shape
    # num_actions2, num_states2 = G2.P.shape
    # T = sim.compute_T_ray(G1, G2)
    N=15; print(N, timeit.timeit(lambda: compare_shapes2((N, N), (N, N), p1=.8, p2=0.9), number=1))