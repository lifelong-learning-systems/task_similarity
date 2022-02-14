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

from matplotlib import colors

OUT_DIR = None
DPI = 300

Condition = namedtuple('Condition', 'dim reward rot method')
def cond_to_str(x: Condition):
    rot_str = ', Rot' if x.rot else ''
    dim_str = 'Lg' if x.dim == 13 else 'Sm'
    reward_str = f', R{int(x.reward)}'
    return f'{dim_str}{reward_str}{rot_str}'

def metric_name(x, method):
    action=False
    if x in ['Song', 'Uniform']:
        ret = x
    elif x == 'New':
        ret = 'SS2'
    else:
        ret = 'SS2'
        action=True
    action_str = ' + Action' if action else ''
    ret += f', {method.title()}{action_str}'
    return ret

def name_to_metric_method(name):
    method = 'weight' if 'W' in name else 'state'
    if 'Song' in name:
        return 'Song', method
    elif 'Uniform' in name:
        return 'Uniform', method
    elif 'A' in name:
        return 'New_Action', method
    else:
        return 'New', method

def metric_name_short(metric_name):
    metric, method = metric_name.split(', ')
    method_str = ', '
    if 'State' in method:
        method_str += 'S'
    else:
        method_str += 'W'
    
    if 'Action' in method:
        method_str += ' + A'
    return f'{metric}{method_str}'


def shorter_cond(dim, rot):
    dim_str = 'Large (13x13)' if dim == 13 else 'Small (9x9)'
    rot_str = ' + Rotations' if rot else ''
    return f'{dim_str}{rot_str}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Top level dir to read from', default='nested_results')
    args = parser.parse_args()
    OUT_DIR = args.results

    files = glob.glob(f'{OUT_DIR}/**/*.dill')
    files.sort()

    env_files =  glob.glob(f'{OUT_DIR}/**/**/all_envs.dill')   
    # Display an environment from no rotations, dim 13
    env_file = list(filter(lambda x: f'{OUT_DIR}/dim13_reward100_num100' in x and 'state_action' in x, env_files))[0]
    with open(env_file, 'rb') as f:
        all_envs = dill.load(f)
    choice_env = all_envs['target']
    choice_env.do_render = True
    choice_grid = choice_env.reset(center=False)
    palette = sns.color_palette()
    #cmap = colors.ListedColormap(['#420359', '#11618a', '#2ac95f', '#fff419'])
    # grey, black, green, red
    cmap = colors.ListedColormap([palette[7], (0, 0, 0), palette[2], palette[3]])
    #bounds = [0, 1, 2, 3, 4]
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    scale = 50
    new_grid = np.zeros((scale*choice_grid.shape[0], scale*choice_grid.shape[1]))
    for i in range(choice_grid.shape[0]):
        for j in range(choice_grid.shape[1]):
            new_grid[scale*i:scale*(i + 1), scale*j:scale*(j+1)] = choice_grid[i, j]
    plt.imsave(f'{OUT_DIR}/example_grid.png', new_grid, cmap=cmap, dpi=150)

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
        df = df[df.Reward == 100]
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

    def plot_dist(df, group_col, val_col, subplot_cols, title, out='mean', filter=None):
        sns.set_context("notebook", font_scale=1.1)
        if filter is not None:
            df = filter(df)
        reward_set = 100
        df = df[df.Reward == reward_set]
        assert subplot_cols == ['Dimension', 'Rotate'], 'Only 2D subplot supported'

        # for dim in df['Dimension'].unique():
        #     for reward in df['Reward'].unique():
        #         for rot in df['Rotate'].unique():
        #             perf_curves = {}
        #             for method in df['Method'].unique():
        #                 cond = Condition(dim, reward, rot, method)
        #                 cond_df = df[(df.Method == method) & (df.Condition == cond_to_str(cond))]
        #                 keys = [name_to_metric(algo) for algo in cond_df['Algorithm'].unique()]
        #                 data = all_data[cond]
        #                 import pdb; pdb.set_trace()

        subplot_dims = [df[col].nunique() for col in subplot_cols]
        plt.clf()
        fig, axs = plt.subplots(*subplot_dims, sharex=True, sharey=True)
        fig_curve, axs_curve = plt.subplots(*subplot_dims, sharex=True, sharey=True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_curve.tight_layout(rect=[0, 0.03, 1, 0.95])
        for i, col1 in enumerate(df[subplot_cols[0]].unique()):
            for j, col2 in enumerate(df[subplot_cols[1]].unique()):
                sub_df = df[(df[subplot_cols[0]] == col1) & (df[subplot_cols[1]] == col2)]
                ax = axs[i, j]
                if 'relative' in out:
                    ax.set(xlim=(0, 100))
                ax.title.set_text(shorter_cond(col1, col2))
                legend = (i == 0 and j == 1)
                sns.kdeplot(data=sub_df, x=val_col, hue=group_col, fill=True, bw_adjust=1, ax=ax, legend=legend, cut=0)
                groups = sub_df[group_col].unique()
                metric_methods = [name_to_metric_method(x) for x in groups]
                conds = [Condition(col1, reward_set, col2, method) for _, method in metric_methods]
                data = []
                optimals = []
                for (metric, _), cond in zip(metric_methods, conds):
                    data.append(all_data[cond]['avg_perfs'][metric])
                    optimals.append(all_data[cond]['optimals']['state']['Song'][0])
                ax_curve = axs_curve[i, j]
                ax_curve.set_xlabel('Total Iterations')
                ax_curve.set_ylabel(val_col)
                ax_curve.set_title(shorter_cond(col1, col2))
                ax_curve.label_outer()
                for algo, perf in zip(groups, data):
                    x = np.linspace(0, PERF_ITER, N_CHUNKS)
                    if 'relative' not in out:
                        y = perf
                    else:
                        y = np.diff(perf)
                        y = y/((PERF_ITER/N_CHUNKS)/optimals[0])
                        x = x[:-1]
                    ax_curve.plot(x, y, marker='.', label=algo)
                if legend:
                    ax_curve.legend()
        # for ax in axs.flat:
        #     ax.label_outer()

        fig.suptitle(title)
        fig.set_size_inches(12, 9)
        fig.savefig(f'{OUT_DIR}/dist_{out}_transfer.png', dpi=DPI, bbox_inches='tight', pad_inches = 0.1)
        fig_curve.suptitle('Average ' + title)
        fig_curve.set_size_inches(12, 9)
        fig_curve.savefig(f'{OUT_DIR}/curves_{out}_transfer.png', dpi=DPI, bbox_inches='tight', pad_inches = 0.1)

    perfs = [[metric_name(metric, cond.method), cond_to_str(cond), cond.dim, cond.reward, cond.rot, metric, cond.method, idx, val, \
              all_data[cond]['scores'][metric][idx], all_data[cond]['haus_scores'][metric][idx], all_data[cond]['optimals'][cond.method][metric][idx]] \
              for cond in conds \
              for metric, perf_dict in all_data[cond]['final_perfs'].items() \
              for idx, val in perf_dict.items()]
    df = pd.DataFrame(perfs, columns=['Algorithm', 'Condition', 'Dimension', 'Reward', 'Rotate', 'Metric', 'Method', 'Source', 'Episodes Completed', 'Score', 'Haus Score', 'Optimal'])
    df['Avg. Episode Performance'] = PERF_ITER / df['Episodes Completed']
    # Storing as percentage
    df['Relative Performance'] = 100*df['Optimal'] / df['Avg. Episode Performance']
    group_keys = ['Algorithm', 'Condition']
    df['Metric ID'] = df['Metric'].apply(lambda x: 1 if x == 'New_Action' else 2 if x == 'New' else 3 if x == 'Song' else 4)
    df = df.sort_values(['Source', 'Dimension', 'Rotate', 'Reward', 'Metric ID', 'Method']) 

    df['Median Performance'] = df.groupby(group_keys)['Episodes Completed'].transform(lambda x: x.median())
    df['Mean Performance'] = df.groupby(group_keys)['Episodes Completed'].transform(lambda x: x.mean())
    df['Avg. Relative Performance %'] = df.groupby(group_keys)['Relative Performance'].transform(lambda x: x.mean())
    df['Med. Relative Performance %'] = df.groupby(group_keys)['Relative Performance'].transform(lambda x: x.median())
    df.to_csv(f'{OUT_DIR}/all_data.csv')

    title_base = f'Number of Episodes Completed'
    rel_title_base = f'% of Optimal Performance'
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
            else:
                ret = best_methods
            return ret

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

        dist_plot = main_key == 'best'
        subplot_cols = ['Dimension', 'Rotate']
        title_str += ' Method for Each Metric'
        if dist_plot:
            plot_dist(df, 'Algorithm', 'Episodes Completed', subplot_cols, title=f'{title_base}{title_str}', out=out_str.replace('_', ''), filter=filter_func)
            plot_dist(df, 'Algorithm', 'Relative Performance', subplot_cols, title=f'{rel_title_base}{title_str}', out=out_str+'relative', filter=filter_func)


    exp_keys = ['Dimension', 'Rotate', 'Reward']
    corr_keys = ['Metric', 'Method', *exp_keys]
    #corr_keys = ['Metric', 'Method']
    all_perfs = dict(df.groupby(corr_keys)['Relative Performance'].apply(list))
    all_scores = dict(df.groupby(corr_keys)['Score'].apply(list))
    
    algos = {}
    groups = []
    for key in all_perfs.keys():
        perfs = all_perfs[key]
        metric, method = key[:2]
        group = tuple(key[2:])
        if group not in groups:
            groups.append(group)
        algo = (metric, method)
        if algo not in algos:
            algos[algo] = {'R': {}, 'MAD': {}, 'STD': {}, 'Mean': {}}
        scores = all_scores[key]
        #R = np.corrcoef(perfs, scores)[0, 1]
        stats = pg.corr(perfs, scores)
        R = stats['r']
        p_val = stats['p-val']
        algos[algo]['R'][group] = R
        algos[algo]['MAD'][group] = pg.mad(perfs)
        algos[algo]['STD'][group] = np.array(perfs).std()
        algos[algo]['Mean'][group] = np.array(perfs).mean()
        #print(key, '%.4f' % R, '%.4f' % p_val)
        # plt.clf()
        # plt.scatter(scores, perfs)
        # plt.xlabel('Distance')
        # plt.ylabel('Performance')
        # plt.title(key)
        #plt.show()

    rows = []
    for group in groups:
        print(group)
        for algo, stats in algos.items():
            row = [*group]
            metric, method = algo
            R = stats['R'][group].iloc[0]
            MAD = stats['MAD'][group]
            STD = stats['STD'][group]
            Mean = stats['Mean'][group]
            row.extend([metric, method, Mean, STD, MAD, R])
            print(f'\t{algo}: {"%.2f"%Mean} (STD={"%.2f"%STD}, MAD={"%.2f"%MAD})')
            rows.append(row)
        print()
    stats_df = pd.DataFrame(rows, columns=['Dimension', 'Rotate', 'Reward', 'Metric', 'Method', 'Mean', 'STD', 'MAD', 'R'])
    stats_df['Condition'] = stats_df.apply(lambda x: f'{cond_to_str(Condition(x.Dimension, x.Reward, x.Rotate, x.Method))}', axis=1)
    stat_str = lambda x: f'{"%.1f"%x.Mean} ({"%.1f"%x.STD})'
    stats_df['Out'] = stats_df.apply(stat_str, axis=1)
    stats_df['Algorithm'] = stats_df.apply(lambda x: metric_name(x.Metric, x.Method), axis=1)
    stats_df = stats_df.sort_values(['Condition', 'Algorithm'])

    conds = stats_df.Condition.unique()
    algos = stats_df.Algorithm.unique()
    info = stats_df.groupby(['Condition', 'Algorithm'])['Out'].first()
    rows = []
    cols = ['Condition']
    col_init = False
    for cond in conds:
        row = [cond]
        for algo in algos:
            val = info[info.index == (cond, algo)].iloc[0]
            row.append(val)
            if not col_init:
                cols.append(metric_name_short(algo))
        col_init = True
        rows.append(row)
    table_df = pd.DataFrame(rows, columns = cols)
    table_df = table_df.set_index('Condition')
    #table_df = table_df.transpose()
    #column_format='l'*(len(table_df.columns))
    #print(table_df)
    latex = table_df.to_latex(caption='Full Statistical Results. Entries are "mean (std)" for the given algorithm and condition.',
                            label='tab:full_results')
    latex = latex.replace('\\bottomrule', '')
    latex = latex.replace('\\toprule', '')
    latex = latex.replace('\\midrule', '')
    latex = latex.replace('table', 'table*')
    print(latex)
    with open(f'{OUT_DIR}/full_res.tex', 'w+') as f:
        f.write(latex + '\n')

    #print(algos)
    #print({(key, measure): np.array(x).mean() for key, inner_items in algos.items() for measure, x in inner_items.items()})
    #import pdb; pdb.set_trace()

    threshold = 0.2
    algo_anova = {}
    anova_rows = []
    cols = ['Algorithm']
    col_init = False
    for algo in df['Algorithm'].unique():
        sub_df = df[df['Algorithm'] == algo]
        res = pg.anova(data=sub_df, dv='Relative Performance', between=exp_keys)
        filtered = res[res['p-unc'] <= threshold]
        filtered = filtered[['Source', 'F', 'p-unc', 'np2']].values
        algo_anova[algo] = filtered
        # print(algo, 'Significant Effects:', len(filtered))
        for x in filtered:
            source, f, p, eff = x
            # print(f'\t{source}:\tp={"%.4e"%p}\tnp2={"%.4f"%eff}')
        # print()
        row = [algo]
        for key in exp_keys:
            res = pg.welch_anova(data=sub_df, dv='Relative Performance', between=key)
            source, p_unc = res[['Source', 'p-unc']].iloc[0]
            row.extend([p_unc])
            if not col_init:
                cols.extend([f'{key} P-Value'])
        col_init = True
        anova_rows.append(row)
    anova_df = pd.DataFrame(anova_rows, columns=cols)
    anova_df = anova_df.set_index('Algorithm')
    for key in exp_keys:
        anova_df[f'{key} P-Value'] = pg.multicomp(anova_df[f'{key} P-Value'].values, method='bonf')[1]
    #print(anova_df)
    anova_df = anova_df.drop(columns=['Reward P-Value'])
    latex = anova_df.to_latex(caption='Anova Results. Reported p-values are Bonferroni corrected; Reward is omitted as its p-value was 1.0000 in all cases.',
                            label='tab:anova_res', float_format='%.4e')
    latex = latex.replace('\\bottomrule', '')
    latex = latex.replace('\\toprule', '')
    latex = latex.replace('\\midrule', '')
    #latex = latex.replace('table', 'table*')
    print(latex)
    with open(f'{OUT_DIR}/anova.tex', 'w+') as f:
        f.write(latex + '\n')

    corr_rows = []
    # e.g. SS2, S + A; Song, W, etc.
    for cond in df['Condition'].unique():
        row = {}
        row['Condition'] = cond
        for algo in df['Algorithm'].unique():
            sub_df = df[(df['Algorithm'] == algo) & (df['Condition'] == cond)]
            scores = sub_df['Score'].values
            perf = sub_df['Relative Performance'].values
            res = pg.corr(scores, perf)
            R = res['r'].item()
            row[algo] = R
        corr_rows.append(row)
    combined_row = {}
    combined_row['Condition'] = 'All'
    for algo in df['Algorithm'].unique():
        sub_df = df[df['Algorithm'] == algo]
        scores = sub_df['Score'].values
        perf = sub_df['Relative Performance'].values
        res = pg.corr(scores, perf)
        R = res['r'].item()
        combined_row[algo] = R
    corr_rows.append(combined_row)
    corr_df = pd.DataFrame(corr_rows).set_index('Condition')
    corr_df.columns = [metric_name_short(x) for x in corr_df.columns]
    corr_df = corr_df.sort_index()
    latex = corr_df.to_latex(caption="Kantarovich Pearson Correlation Results: Desired relation is negative.", label="tab:corr_res", float_format='%.3f')
    latex = latex.replace('\\bottomrule', '')
    latex = latex.replace('\\toprule', '')
    latex = latex.replace('\\midrule', '')
    latex = latex.replace('table', 'table*')
    print(latex)
    with open(f'{OUT_DIR}/corr.tex', 'w+') as f:
        f.write(latex + '\n')

    corr_rows = []
    # e.g. SS2, S + A; Song, W, etc.
    for cond in df['Condition'].unique():
        row = {}
        row['Condition'] = cond
        for algo in df['Algorithm'].unique():
            sub_df = df[(df['Algorithm'] == algo) & (df['Condition'] == cond)]
            scores = sub_df['Haus Score'].values
            perf = sub_df['Relative Performance'].values
            res = pg.corr(scores, perf)
            R = res['r'].item()
            row[algo] = R
        corr_rows.append(row)
    combined_row = {}
    combined_row['Condition'] = 'All'
    for algo in df['Algorithm'].unique():
        sub_df = df[df['Algorithm'] == algo]
        scores = sub_df['Haus Score'].values
        perf = sub_df['Relative Performance'].values
        res = pg.corr(scores, perf)
        R = res['r'].item()
        combined_row[algo] = R
    corr_rows.append(combined_row)
    corr_df = pd.DataFrame(corr_rows).set_index('Condition')
    corr_df.columns = [metric_name_short(x) for x in corr_df.columns]
    corr_df = corr_df.sort_index()
    latex = corr_df.to_latex(caption="Hausdorff Pearson Correlation Results: Desired relation is negative.", label="tab:corr_haus_res", float_format='%.3f')
    latex = latex.replace('\\bottomrule', '')
    latex = latex.replace('\\toprule', '')
    latex = latex.replace('\\midrule', '')
    latex = latex.replace('table', 'table*')
    print(latex)
    with open(f'{OUT_DIR}/corr_haus.tex', 'w+') as f:
        f.write(latex + '\n')
