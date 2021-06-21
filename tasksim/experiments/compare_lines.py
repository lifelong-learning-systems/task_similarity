import numpy as np

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim

def check_triangle_inequality(matrix, decimals = 10):
    n, _ = matrix.shape
    boundary = -10**-decimals
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # check that d(i, j) <= d(i, k) + d(k, j)
                if not (matrix[i, k] + matrix[k, j] - matrix[i, j] >= boundary):
                    return False, (i, j, k)
    return True, None

# in theory, rows should be INCREASING, cols DECREASING (within upper triangle)
def check_diffs(matrix, decimals=10):
    n, _ = matrix.shape
    boundary = -10**-decimals
    # ensure symmetric
    if np.abs(matrix - matrix.T).max() > abs(boundary):
        return False
    # check rows in upper triangle
    for i in range(n):
        last_val = matrix[i, i]
        for j in range(i+1, n):
            if matrix[i, j] - last_val < boundary:
                return False
            last_val = matrix[i, j]
    # check cols in upper triangle
    for i in range(n):
        last_val = matrix[0, i]
        for j in range(1, i+1):
            if last_val - matrix[j, i] < boundary:
                return False
            last_val = matrix[j, i]
    return True


def check_metric_properties(matrix, decimals=10):
    precision_check = decimals
    order = check_diffs(comparisons, precision_check)
    if not order:
        print('Order check failed')
    else:
        print('Order check passed!')
    ans, counter = check_triangle_inequality(comparisons, precision_check)
    if not ans:
        print(f'Triangle inequality check ({precision_check} decimals) failed at:', counter)
    else:
        print(f'Triangle inequality check ({precision_check} decimals) passed!')

if __name__ == '__main__':
    tasksim.init_ray()
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    sizes = range(2, 9, 1)
    comparisons = np.zeros((len(sizes), len(sizes)))
    for i, s1 in enumerate(sizes):
        for j, s2 in enumerate(sizes):
            print(i, j)
            G1 = gen.MDPGraph.from_grid(gen.create_grid((1, s1)), 0.9, noops=False)
            G2 = gen.MDPGraph.from_grid(gen.create_grid((1, s2)), 0.9, noops=False)
            actions1, states1 = G1.P.shape
            actions2, states2 = G2.P.shape
            #ot.toc('Generating graphs: {}')
            #dist_T = gen.compute_T_ray(G1, G2)
            dist_S, dist_A, numiters, done = G1.compare(G2)
            # s_score = sim.final_score(sim.norm(dist_S, [], 'minmax')[0])
            # a_score = sim.final_score(sim.norm(dist_A, [], 'minmax')[0])
            s_score = sim.final_score(dist_S)
            a_score = sim.final_score(dist_A)
            comparisons[i][j] = 0.5 * s_score + 0.5 * a_score
    
    # Check to ensure metric properties hold
    precision_check = 10
    check_metric_properties(comparisons, precision_check)

    num_print_decimals = 8
    np.set_printoptions(linewidth=200, precision=num_print_decimals, suppress=True)
    print(list(sizes))
    print(comparisons)
    G2 = gen.MDPGraph.from_grid(gen.create_grid((1, 2)), 0.9, noops=False)
    G3 = gen.MDPGraph.from_grid(gen.create_grid((1, 3)), 0.9, noops=False)
    G4 = gen.MDPGraph.from_grid(gen.create_grid((1, 4)), 0.9, noops=False)
    s23, a23, _, _ = G2.compare(G3)
    s24, a24, _, _ = G2.compare(G4)
    s34, a34, _, _ = G3.compare(G4)
    # G1 = gen.MDPGraph.from_grid(gen.create_grid((1, 3)), 0.9)
    # print(G1.reorder_states([2, 0, 1]))
