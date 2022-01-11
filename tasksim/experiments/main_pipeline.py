from scipy.sparse.construct import random
from tasksim.experiments.pipeline_utilities import progress_bar
import numpy as np
from sklearn.preprocessing import normalize as sk_norm

import tasksim
from tasksim import MDPGraph
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim
from matplotlib import pyplot as plt

import pipeline_utilities as util

PRECISION_CHECK = 10
FIG_OUT = 'figures_baseline'

def generate_graphs(sizes, success_prob=0.9, strat=gen.ActionStrategy.NOOP_EFFECT_COMPRESS):
    return [gen.MDPGraph.from_grid(gen.create_grid(sz), success_prob, strat=strat) for sz in sizes]

def compare_graphs(graphs, verify_metric=True, print_progress=True, title=None, xticks=None, yticks=None, upper=True, standard_range=True, use_song=False):
    if np.array(graphs).ndim == 1:
        graphs = [[(graphs[i], graphs[j]) for j in range(len(graphs))] for i in range(len(graphs))]
    ret = np.zeros((len(graphs), len(graphs[0])))
    idxs = [(i, j) for i in range(len(graphs)) for j in range(len(graphs[0]))]
    def comp(i, j):
        g1, g2 = graphs[i][j]
        # TODO: this returns distance now...use directly!
        return g1.compare2(g2) if not use_song else g1.compare2_song(g2)
    if print_progress:
        for (i, j) in util.progress_bar(idxs, prefix='Progress:', suffix='Complete'):
            ret[i, j] = comp(i, j)
    else:
        for (i, j) in idxs:
            ret[i, j] = comp(i, j)
    ax_heatmap = util.heatmap(ret, title=title, xticks=xticks, yticks=yticks, upper=upper, standard_range=standard_range)
    if verify_metric:
        metric = util.check_metric_properties(ret, graphs, decimals=PRECISION_CHECK, output=True)
        return ret, ax_heatmap, metric
    return ret, ax_heatmap

def process_print_graphs(graphs, title, ticks=None, xticks=None, yticks=None, upper=True, standard_range=True, use_song=False):
    print(f'\n{title}')
    if ticks is not None:
        xticks = ticks
        yticks = ticks
    comp, heatmap, metric = compare_graphs(graphs, verify_metric=True, title=title, xticks=xticks, yticks=yticks, upper=upper, standard_range=standard_range, use_song=use_song)
    #print(f'Metric valid (order, triangle inequality, symmetry): {metric}')
    print(comp if not upper else np.triu(comp))
    plt.figure(plt.get_fignums()[-1]).savefig(f'{FIG_OUT}/{title.lower().replace(" ", "_")}.png')
    return comp, heatmap, metric

def shape_comparisons(line_sizes=None, grid_sizes=None, use_song=False):
    if line_sizes is None:
        line_sizes = range(2, 12)
    if grid_sizes is None:
        grid_sizes = range(2, 12)
    line_shapes = [(1, i) for i in line_sizes]
    grid_shapes = [(i, i) for i in grid_sizes]
    lines = generate_graphs(line_shapes, strat=gen.ActionStrategy.NOOP_EFFECT_COMPRESS)
    grids = generate_graphs(grid_shapes, strat=gen.ActionStrategy.NOOP_EFFECT_COMPRESS)
    song_str = '' if not use_song else ' Song'
    process_print_graphs(lines, 'Line Distances' + song_str, ticks=[str(s[1]) for s in line_shapes], standard_range=False, use_song=use_song)
    process_print_graphs(grids, 'Grid Distances' + song_str, ticks=[str(s[1]) for s in grid_shapes], standard_range=False, use_song=use_song)
    # lines = generate_graphs(line_shapes)
    # grids = generate_graphs(grid_shapes)
    # lines_noops = generate_graphs(line_shapes, noops=gen.ActionStrategy.NOOP_ACTION)
    # grids_noops = generate_graphs(grid_shapes, noops=gen.ActionStrategy.NOOP_ACTION)
    # process_print_graphs(lines, 'Line Similarities', ticks=[str(s[1]) for s in line_shapes], use_song=use_song)
    # process_print_graphs(grids, 'Grid Similarities', ticks=[str(s[1]) for s in grid_shapes], use_song=use_song)
    # process_print_graphs(lines_noops, 'Line with No-ops Similarities', ticks=[str(s[1]) for s in line_shapes], use_song=use_song)
    # process_print_graphs(grids_noops, 'Grid with No-ops Similarities', ticks=[str(s[1]) for s in grid_shapes], use_song=use_song)

def success_prob_comparisons(grid_size=7, probs=None, use_song=False):
    if probs is None:
        probs = np.arange(0.1, 1.1, 0.1)
    grid = gen.create_grid((grid_size, grid_size))
    graphs = [gen.MDPGraph.from_grid(grid, prob, strat=gen.ActionStrategy.NOOP_EFFECT_COMPRESS) for prob in probs]
    song_str = '' if not use_song else ' Song'
    process_print_graphs(graphs, f'Action Success Probabilities {grid_size}x{grid_size}' + song_str, ticks=[f'{prob:.1f}' for prob in probs], standard_range=False, use_song=use_song)

def transition_prob_noise(grid_size=7, success_prob=0.75, trials=10, random_state=None, use_song=False):
    if random_state is None:
        random_state = np.random
    
    song_str = '' if not use_song else ' Song'
    def add_noise(G, percent):
        G = G.copy()
        P = G.P
        out_a = G.out_a
        n_actions, n_states = P.shape
        for i in range(n_actions):
            for j in range(n_states):
                if P[i, j] <= 0:
                    continue
                # Between [0, 1)
                noise = random_state.rand()
                # Between [-percent, +percent)
                z = -percent + 2*percent*noise
                # TODO: scale by element or nah? Probably?
                P[i, j] = max(0, P[i, j] + z)
            normed_row = sk_norm(np.array([P[i]]), norm='l1')[0]
            P[i, :] = normed_row
            out_a[i] = normed_row.copy()
        return G
    grid = gen.create_grid((grid_size, grid_size))
    base = gen.MDPGraph.from_grid(grid, success_prob, strat=gen.ActionStrategy.NOOP_EFFECT_COMPRESS)
    noise_levels = np.arange(0, 0.5, 0.05)
    comparisons = np.zeros((trials, len(noise_levels)))
    idxs = [(i, j, noise) for i in range(trials) for j, noise in enumerate(noise_levels)]
    graphs = [[(base, add_noise(base, noise)) for j, noise in enumerate(noise_levels)] for i in range(trials)]
    title = f'Transition Noise {grid_size}x{grid_size}, {success_prob}' + song_str
    xticks = [f'{noise:.2f}' for noise in noise_levels]
    yticks = [str(trial) for trial in range(1, 1+trials)]
    process_print_graphs(graphs, title=title, xticks=xticks, yticks=yticks, upper=False, standard_range=False, use_song=use_song)

if __name__ == '__main__':
    plt.ion()
    plt.rcParams['figure.dpi'] = 300
    tasksim.init_ray()
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    num_print_decimals = 5
    np.set_printoptions(linewidth=200, precision=num_print_decimals, suppress=True)

    # Main process
    # TODO: these CANNOT have standard range
    shape_comparisons(use_song=True)
    shape_comparisons()

    # TODO: these can have standard range
    success_prob_comparisons(use_song=True)
    success_prob_comparisons()

    # # Introduce some determinism
    random_seed = 314159265
    random_state = np.random.RandomState(random_seed)
    # TODO: these can have standard range
    transition_prob_noise(random_state=random_state, use_song=True)
    transition_prob_noise(grid_size=3, random_state=random_state, use_song=True)
    transition_prob_noise(random_state=random_state)
    transition_prob_noise(grid_size=3, random_state=random_state)