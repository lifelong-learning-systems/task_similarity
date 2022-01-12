import numpy as np
from matplotlib import pyplot as plt

RESULTS_DIR='results_tmp'

file_names = ['new_res.txt', 'song_res.txt']


if __name__ == '__main__':
    results = {}
    for f in file_names:
        f2 = f'{RESULTS_DIR}/{f}'
        with open(f2) as ptr:
            lines = ptr.readlines()
        
        def process_line(line):
            line = line.rstrip()
            line = line[1:-1]
            tokens = line.split(', ')
            return np.array([float(token) for token in tokens])
        y = process_line(lines[0])
        x = process_line(lines[1])
        results[f.split('_res.txt')[0].title()] = (x, y)

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
    plt.title('Speedup over Song Baseline')
    plt.legend()
    plt.show()



    fig, axs = plt.subplots(1, len(results))
    for ax, data in zip(axs, results.items()):
        metric, vals = data
        dists, iters = vals
        print(f'Metric {metric}: R = {np.corrcoef(dists, iters)[0, 1]}')
        ax.set_title(metric)
        ax.scatter(dists, iters)

    fig.tight_layout()
    plt.show()