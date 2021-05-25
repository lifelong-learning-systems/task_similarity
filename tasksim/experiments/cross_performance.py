from matplotlib import pyplot as plt
import numpy as np
import timeit

from tasksim import MDPGraph, DEFAULT_CS, DEFAULT_CA
from tasksim.structural_similarity import structural_similarity
from tasksim.gridworld_generator import create_grid

if __name__ == '__main__':
    prob = 0.9
    c_a = DEFAULT_CA
    c_s = DEFAULT_CS
    print(f'C_a: {c_a}, C_s: {c_s}:')
    similarity_repetitions = 3 
    # have to go from at least a 2x2, since 1x1 with just goal state has zero actions...breaks things
    x = []
    y = []
    # compare to next size up? Just out of curiosity
    for i in range(1, 11):
        n = i + 1
        # speedup for later rounds
        if n > 3:
            similarity_repetitions = 1
        grid1 = create_grid((n, n))
        grid2 = create_grid((n, n))
        mdp1 = MDPGraph.from_grid(grid1, prob)
        mdp2 = MDPGraph.from_grid(grid2, prob)
        print(f'MDP 1: {mdp1.P.shape[1]}, MDP 2: {mdp2.P.shape[1]}')
        if n <= 3:
            chace_time = timeit.timeit(lambda: mdp1.compare(mdp2, c_a=c_a, c_s=c_s, chace=True),
                                            number=similarity_repetitions)
            chace_avg = chace_time / similarity_repetitions
            print(f'\tChace, average of {similarity_repetitions}: {chace_avg:.3E}')
        if n <= 6:
            append_time = timeit.timeit(lambda: mdp1.compare(mdp2, c_a=c_a, c_s=c_s, append=True),
                                            number=similarity_repetitions)
            append_avg = append_time / similarity_repetitions
            print(f'\tAppend, average of {similarity_repetitions}: {append_avg:.3E}')
        opt_time = timeit.timeit(lambda: mdp1.compare(mdp2, c_a=c_a, c_s=c_s),
                                        number=similarity_repetitions)
        opt_avg = opt_time / similarity_repetitions
        print(f'\tOptimized, average of {similarity_repetitions}: {opt_avg:.3E}')
        # Already in (n + m)
        x.append(mdp1.P.shape[1] + mdp2.P.shape[1])
        y.append(opt_avg)
    print('Scatter points:')
    print('\t' + str(x))
    print('\t' + str(y))
    print('Log slopes:')
    log_y = np.log(y)
    log_x = np.log(x)
    log_slopes = [(log_y[i] - log_y[i-1])/(log_x[i] - log_x[i - 1]) for i in range(1, len(log_y))]
    print('\t' + str(log_slopes))
    plt.loglog(x, y)
    #plt.show()