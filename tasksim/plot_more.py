from os import name
import numpy as np
import glob
import dill
import argparse
from collections import namedtuple
import pingouin as pg

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from plot_transfer import N_CHUNKS, PERF_ITER

OUT_DIR = None
DPI = 300

Condition = namedtuple('Condition', 'dim reward rot method')
def cond_to_str(x: Condition):
    rot_str = ', Rot' if x.rot else ''
    dim_str = 'Lg' if x.dim == 13 else 'Sm'
    reward_str = f', R{int(x.reward)}'
    return f'{dim_str}{reward_str}{rot_str}'

def metric_name(x, method):
    if x in ['Song', 'Uniform']:
        ret = x
    elif x == 'New':
        ret = 'Ours'
    else:
        ret = 'Ours (+ Action Sim)'
    ret += f', {method.title()}'
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Top level dir to read from', default='nested_results')
    args = parser.parse_args()
    OUT_DIR = args.results

    files = glob.glob(f'{OUT_DIR}/**/*.dill')
    files.sort()

    all_data = {}
    conds = []
    for file_name in files:
        with open(file_name, 'rb') as f:
            obj = dill.load(f)
        meta = obj['meta']
        method = 'weight' if 'weight' in file_name else 'state'
        N = int(meta['num'])
        cond = Condition(int(meta['dim']), int(float(meta['reward'])), meta['rotate'], method)
        conds.append(cond)
        all_data[cond] = obj

    # plot with seaborn barplot
    def plot_bar(df, cond_col, group_col, val_col, title, out='mean', filter=None):
        sns.set_context("notebook", font_scale=1.4)
        y_max = df[val_col].max()*1.3
        if filter is not None:
            df = filter(df)
        df = df[[cond_col, group_col, val_col]].drop_duplicates()
        plt.clf()
        ax = sns.barplot(data=df, x=cond_col, y=val_col, hue=group_col)
        if 'relative' in out:
            ax.set(ylim=(0, 130))
        else:
            ax.set(ylim=(0, y_max))
        for container in ax.containers:
            if 'relative' in out:
                #labels = [f"{('%d' % x)}%" for x in container.datavalues]
                labels = [f"{('%d' % x)}" for x in container.datavalues]
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, fmt='%d')

        plt.title(title)
        fig = plt.gcf()
        fig.set_size_inches(16, 10)
        plt.savefig(f'{OUT_DIR}/{out}_transfer.png', dpi=DPI, bbox_inches = 'tight', pad_inches = 0.1)
        return df
    
    perfs = [[metric_name(metric, cond.method), cond_to_str(cond), cond.dim, cond.reward, cond.rot, metric, cond.method, idx, val, \
              all_data[cond]['scores'][metric][idx], all_data[cond]['optimals'][cond.method][metric][idx]] \
              for cond in conds \
              for metric, perf_dict in all_data[cond]['final_perfs'].items() \
              for idx, val in perf_dict.items()]
    df = pd.DataFrame(perfs, columns=['Algorithm', 'Condition', 'Dimension', 'Reward', 'Rotate', 'Metric', 'Method', 'Source', 'Avg. Final Performance', 'Score', 'Optimal'])
    df['Avg. Episode Performance'] = PERF_ITER / df['Avg. Final Performance']
    # Storing as percentage
    df['Relative Performance'] = 100*df['Optimal'] / df['Avg. Episode Performance']
    group_keys = ['Algorithm', 'Condition']
    df['Metric ID'] = df['Metric'].apply(lambda x: 1 if x == 'New_Action' else 2 if x == 'New' else 3 if x == 'Song' else 4)
    df = df.sort_values(['Source', 'Dimension', 'Rotate', 'Reward', 'Metric ID', 'Method']) 

    df['Median Performance'] = df.groupby(group_keys)['Avg. Final Performance'].transform(lambda x: x.median())
    df['Mean Performance'] = df.groupby(group_keys)['Avg. Final Performance'].transform(lambda x: x.mean())
    df['Avg. Relative Performance %'] = df.groupby(group_keys)['Relative Performance'].transform(lambda x: x.mean())
    df['Med. Relative Performance %'] = df.groupby(group_keys)['Relative Performance'].transform(lambda x: x.median())

    title_base = f'Number of Episodes Completed in First {PERF_ITER} Steps (N = {N})'
    rel_title_base = f'Percent of Optimal Performance in First {PERF_ITER} Steps (N = {N})'
    plot_bar(df, 'Condition', 'Algorithm', 'Median Performance', title = f'Median {title_base}', out='median')
    plot_bar(df, 'Condition', 'Algorithm', 'Mean Performance', title = f'Mean {title_base}', out='mean')
    plot_bar(df, 'Condition', 'Algorithm', 'Avg. Relative Performance %', title = f'Mean {rel_title_base}', out='avg_relative')
    plot_bar(df, 'Condition', 'Algorithm', 'Med. Relative Performance %', title = f'Median {rel_title_base}', out='med_relative')

    best_methods = df[((df.Method == 'weight') & (df.Metric == 'Uniform')) | \
                        ((df.Method == 'weight') & (df.Metric == 'Song')) | \
                        ((df.Method == 'state') & (df.Metric == 'New')) | \
                        ((df.Method == 'state') & (df.Metric == 'New_Action'))]
    for main_key in ['weight', 'state', 'best']:
        def filter_func(x):
            if main_key != 'best':
                ret = x[x.Method == main_key]
                ret['Algorithm'] = ret['Algorithm'].str.split(',').str[0]
                return ret
            return best_methods

        title_str = f', {main_key.title()} Transfer'
        out_str = f'{main_key}_'
        plot_bar(df, 'Condition', 'Algorithm', 'Median Performance', title = f'Median {title_base}{title_str}',\
                 out=out_str+'median', filter=filter_func)
        plot_bar(df, 'Condition', 'Algorithm', 'Mean Performance', title = f'Mean {title_base}{title_str}',\
                 out=out_str+'mean', filter=filter_func)
        plot_bar(df, 'Condition', 'Algorithm', 'Avg. Relative Performance %', title = f'Mean {rel_title_base}{title_str}',\
                 out=out_str+'avg_relative', filter=filter_func)
        plot_bar(df, 'Condition', 'Algorithm', 'Med. Relative Performance %', title = f'Median {rel_title_base}{title_str}',\
                 out=out_str+'med_relative', filter=filter_func)



    exp_keys = ['Dimension', 'Rotate', 'Reward']
    #corr_keys = ['Metric', 'Method', *exp_keys]
    corr_keys = ['Metric', 'Method']
    all_perfs = dict(df.groupby(corr_keys)['Relative Performance'].apply(list))
    all_scores = dict(df.groupby(corr_keys)['Score'].apply(list))
    
    algos = {}
    for key in all_perfs.keys():
        perfs = all_perfs[key]
        metric, method = key[:2]
        algo = (metric, method)
        if algo not in algos:
            algos[algo] = []
        scores = all_scores[key]
        #R = np.corrcoef(perfs, scores)[0, 1]
        stats = pg.corr(perfs, scores)
        R = stats['r']
        p_val = stats['p-val']
        algos[algo].append(R)
        print(key, '%.4f' % R, '%.4f' % p_val)
        plt.clf()
        plt.scatter(scores, perfs)
        plt.xlabel('Distance')
        plt.ylabel('Performance')
        plt.title(key)
        #plt.show()

    #print(algos)
    print({key: np.array(x).mean() for key, x in algos.items()})
    #import pdb; pdb.set_trace()

