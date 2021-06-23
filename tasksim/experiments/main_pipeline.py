from tasksim.experiments.pipeline_utilities import progress_bar
import numpy as np

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim
from matplotlib import pyplot as plt

import pipeline_utilities as util

PRECISION_CHECK = 10
FIG_OUT = 'pipeline_figures'

def generate_graphs(sizes, success_prob=0.9, noops=False):
    return [gen.MDPGraph.from_grid(gen.create_grid(sz), success_prob, noops=noops) for sz in sizes]

def compare_graphs(graphs, verify_metric=True, print_progress=True, title=None, xticks=None, yticks=None):
    length = len(graphs)
    ret = np.zeros((length, length))
    idxs = [(i, j) for i, _ in enumerate(graphs) for j, _ in enumerate(graphs)]
    def comp(i, j):
        return graphs[i].compare2(graphs[j])
    if print_progress:
        for (i, j) in util.progress_bar(idxs, prefix='Progress:', suffix='Complete'):
            ret[i, j] = comp(i, j)
    else:
        for (i, j) in idxs:
            ret[i, j] = comp(i, j)
    ax_heatmap = util.heatmap(ret, title=title, xticks=xticks, yticks=yticks)
    if verify_metric:
        metric = util.check_metric_properties(ret, graphs, decimals=PRECISION_CHECK, output=True)
        return ret, ax_heatmap, metric
    return ret, ax_heatmap

def process_print_graphs(graphs, title, ticks=None, xticks=None, yticks=None):
    print(f'\n{title}')
    if ticks is not None:
        xticks = ticks
        yticks = ticks
    comp, heatmap, metric = compare_graphs(graphs, verify_metric=True, title=title, xticks=xticks, yticks=yticks)
    #print(f'Metric valid (order, triangle inequality, symmetry): {metric}')
    print(np.triu(comp))
    plt.figure(plt.get_fignums()[-1]).savefig(f'{FIG_OUT}/{title.lower().replace(" ", "_")}.png')
    return comp, heatmap, metric

def shape_comparisons(line_sizes=None, grid_sizes=None):
    if line_sizes is None:
        line_sizes = range(2, 12)
    if grid_sizes is None:
        grid_sizes = range(2, 12)
    line_shapes = [(1, i) for i in line_sizes]
    grid_shapes = [(i, i) for i in grid_sizes]
    lines = generate_graphs(line_shapes)
    grids = generate_graphs(grid_shapes)
    lines_noops = generate_graphs(line_shapes, noops=True)
    grids_noops = generate_graphs(grid_shapes, noops=True)
    process_print_graphs(lines, 'Line Similarities', ticks=[str(s[1]) for s in line_shapes])
    process_print_graphs(grids, 'Grid Similarities', ticks=[str(s[1]) for s in grid_shapes])
    process_print_graphs(lines_noops, 'Line with No-ops Similarities', ticks=[str(s[1]) for s in line_shapes])
    process_print_graphs(grids_noops, 'Grid with No-ops Similarities', ticks=[str(s[1]) for s in grid_shapes])

def success_prob_comparisons(grid_size=7, probs=None):
    if probs is None:
        probs = np.arange(0.1, 1.1, 0.1)
    grid = gen.create_grid((grid_size, grid_size))
    graphs = [gen.MDPGraph.from_grid(grid, prob, noops=False) for prob in probs]
    process_print_graphs(graphs, f'Action Success Probabilities {grid_size}x{grid_size}', ticks=[f'{prob:.1f}' for prob in probs])

def transition_prob_noise():
    def add_noise(G, noise):
        P = G.P

    pass

if __name__ == '__main__':
    plt.ion()
    plt.rcParams['figure.dpi'] = 300
    tasksim.init_ray()
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    num_print_decimals = 5
    np.set_printoptions(linewidth=200, precision=num_print_decimals, suppress=True)

    # Main process
    shape_comparisons()
    success_prob_comparisons()