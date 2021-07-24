import numpy as np

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim
from matplotlib import pyplot as plt

import pipeline_utilities as util

if __name__ == '__main__':
    plt.ion()
    tasksim.init_ray()
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    num_print_decimals = 5
    np.set_printoptions(linewidth=200, precision=num_print_decimals, suppress=True)
    sizes = range(2, 12, 1)
    comparisons = np.zeros((len(sizes), len(sizes)))
    graphs = [gen.MDPGraph.from_grid(gen.create_grid((1, x)), 0.9, strat=gen.ActionStrategy.SUBSET) for x in sizes]
    for i, s1 in enumerate(sizes):
        for j, s2 in enumerate(sizes):
            print(i, j)
            G1 = graphs[i]
            G2 = graphs[j]
            actions1, states1 = G1.P.shape
            actions2, states2 = G2.P.shape
            #ot.toc('Generating graphs: {}')
            #dist_T = gen.compute_T_ray(G1, G2)
            dist_S, dist_A, numiters, done = G1.compare(G2)
            # s_score = sim.final_score(sim.norm(dist_S, [], 'minmax')[0])
            # a_score = sim.final_score(sim.norm(dist_A, [], 'minmax')[0])
            s_score = sim.final_score(dist_S)
            #a_score = sim.final_score(dist_A)
            comparisons[i][j] = s_score# * 0.5 + a_score * 0.5
    
    # Check to ensure metric properties hold
    precision_check = 10
    util.check_metric_properties(comparisons, graphs, decimals=precision_check, output=True)

    print(list(sizes))
    print(np.triu(comparisons))
    G2, G3, G4 = graphs[0], graphs[1], graphs[2]
    s23, a23, _, _ = G2.compare(G3)
    s24, a24, _, _ = G2.compare(G4)
    s34, a34, _, _ = G3.compare(G4)
    # G1 = gen.MDPGraph.from_grid(gen.create_grid((1, 3)), 0.9)
    # print(G1.reorder_states([2, 0, 1]))
    heatmap_ax = util.heatmap(comparisons)
    plt.show()
