import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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

def check_shuffled(matrix, graphs, decimals=10):
    # using default np random 
    boundary = 10**-decimals
    graphs_shuffled = [g.copy().shuffle_states() for g in graphs]
    for (i, j), comp_value in np.ndenumerate(matrix):
        G1_shuffled, G2_shuffled = graphs_shuffled[i], graphs_shuffled[j]
        s_score = G1_shuffled.compare2(G2_shuffled)
        if abs(s_score - comp_value) >= boundary:
            return False
    return True

def check_metric_properties(comparisons, graphs, decimals=10, output=False):
    precision_check = decimals
    def output_print(x):
        if output:
            print(x)
    # ensure symmetric
    boundary = -10**-decimals
    if np.abs(comparisons - comparisons.T).max() > abs(boundary):
        output_print('Symmetry check failed')
    else:
        output_print('Symmetry check passed!')
    order = check_diffs(comparisons, precision_check)
    if not order:
        output_print('Order check failed')
    else:
        output_print('Order check passed!')
    ans, counter = check_triangle_inequality(comparisons, precision_check)
    if not ans:
        output_print(f'Triangle inequality check ({precision_check} decimals) failed at: {counter}')
    else:
        output_print(f'Triangle inequality check ({precision_check} decimals) passed!')
    return order, ans
    # shuffle = check_shuffled(comparisons, graphs, precision_check)
    # if not shuffle:
    #     print('Shuffle check failed')
    # else:
    #     print('Shuffle check passed!')

# TODO: labels, ticks, & titles
def heatmap(matrix, title=None, xticks=None, yticks=None, new_figure=True, flip_orientation=True, upper=True, standard_range=True):
    upper_mask = np.triu(matrix)
    lower_mask = np.tril(matrix)
    np.fill_diagonal(upper_mask, 0)
    if flip_orientation:
        upper_mask = np.fliplr(upper_mask)
        lower_mask = np.fliplr(lower_mask)
    if new_figure:
        plt.figure()
    keep = lower_mask
    mask = upper_mask
    if upper:
        if standard_range:
            ax = sns.heatmap(1 - keep, mask=mask, linewidth=0.5, cmap="rainbow", vmin=0, vmax=1)
        else:
            ax = sns.heatmap(1 - keep, mask=mask, linewidth=0.5, cmap="rainbow")
    else:
        keep = np.flipud(matrix) if flip_orientation else matrix
        if standard_range:
            ax = sns.heatmap(1 - keep, linewidth=0.5, cmap="rainbow", vmin=0, vmax=1)
        else:
            ax = sns.heatmap(1 - keep, linewidth=0.5, cmap="rainbow")
    if title is not None:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticklabels(xticks)
    if yticks is not None:
        yticks = np.array(yticks).copy()
        if flip_orientation:
            yticks = np.flip(yticks)
        ax.set_yticklabels(yticks)
    return ax

# From Stackoverflow: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress_bar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()