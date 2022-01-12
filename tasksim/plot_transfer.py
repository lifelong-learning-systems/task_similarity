import numpy as np
from matplotlib import pyplot as plt
import glob
import ast
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Which directory to read from', default='final_results/dim9_reward100_num10_weight')
    args = parser.parse_args()
    RESULTS_DIR = args.results


    file_names = glob.glob(f'{RESULTS_DIR}/*.txt')
    results = {}
    meta = None
    for f in file_names:
        with open(f) as ptr:
            lines = ptr.readlines()
        
        def process_line(line):
            line = line.rstrip()
            line = line[1:-1]
            tokens = line.split(', ')
            return np.array([float(token) for token in tokens])
        y = process_line(lines[0])
        x = process_line(lines[1])
        key = f.split('_res.txt')[0].split('/')[-1].title()
        results[key] = (x, y)
        if meta is None:
            meta = ast.literal_eval(lines[2])

    plt.clf()

    baseline_dists, baseline_iters = results['Song']
    for metric, vals in results.items():
        dists, iters = vals
        idxs = np.arange(len(iters))
        speedup = baseline_iters/iters
        plt.plot(idxs, speedup, label=metric)
        print(f'Metric {metric}: avg speedup: {speedup.mean()}')
    plt.ylabel('Speedup')
    plt.xlabel('Source Index')
    reward = meta['reward']
    dim = meta['dim']
    plt.title(f'Speedup over Song Baseline: Reward, {reward}  Dim: {dim}')
    plt.legend()
    plt.show()



    fig, axs = plt.subplots(1, len(results))
    for ax, data in zip(axs, results.items()):
        metric, vals = data
        dists, iters = vals
        R = np.corrcoef(dists, iters)[0, 1]
        print(f'Metric {metric}: R = {R}')
        ax.set_title(metric)
        ax.scatter(dists, iters, label=('R = %.2f' % R))
        ax.legend()

    fig.tight_layout()
    plt.show()
