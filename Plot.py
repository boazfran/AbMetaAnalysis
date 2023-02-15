"""
function for plotting the AbMetaAnalysis results
"""

# Info
__author__ = 'Boaz Frankel'

# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math
from matplotlib.lines import Line2D
from changeo.Gene import getFamily
import typing


def plot_compare_to_reference_cdr3_df(df, output_file):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    ax1.bar(range(0, len(df) * 2, 2), df['CASE'].absloute)
    ax1.bar(range(1, len(df) * 2, 2), df['CTRL'].absloute)
    ax1.set_xticks([])
    ax1.set_ylabel('Matches', fontsize=20)
    ax1.legend(labels=['CASE', 'CTRL'], fontsize=20)
    ax1.set_yticks(
        range(df.xs('absloute', level=1, axis=1).max().max() + 1),
        range(df.xs('absloute', level=1, axis=1).max().max() + 1),
        fontsize=20
    )
    ax1.grid(axis='y')

    ax2.bar(range(0, len(df) * 2, 2), df['CASE'].support)
    ax2.bar(range(1, len(df) * 2, 2), df['CTRL'].support)
    ax2.set_xticks(np.arange(0.5, len(df) * 2, 2), df.index, fontsize=20)
    ax2.set_ylabel('Support', fontsize=20)
    ax2.set_xlabel('Junction AA length', fontsize=20)
    ax2.set_yticks(
        range(0, math.ceil(df.xs('support', level=1, axis=1).max().max() / 200) * 200, 200),
        range(0, math.ceil(df.xs('support', level=1, axis=1).max().max() / 200) * 200, 200),
        fontsize=20
    )
    ax2.grid(axis='y')
    fig.savefig(output_file)


def boxplot_features(
    by_subj_df,
    by_study_df,
    aliase2study,
    case_aliases,
    ctrl_aliases,
    colors,
    axes,
    case_ctrl_color_diff=3,
    outlier_th=2.5,
    y='prop',
    ylabel="Freq"
):
    positions = np.arange(0, 10, 1)
    for i, col in enumerate(by_subj_df.loc[:, y].columns):
        ax = axes[0, i]
        for alias_itr, alias in enumerate(case_aliases):
            color = colors[alias_itr]
            ax.boxplot(
                x=by_subj_df.loc[('CASE', aliase2study[alias]), (y, col)].loc[
                    by_subj_df.loc[('CASE', aliase2study[alias]), ('zscore', col)].abs() < outlier_th
                ],
                positions=[positions[alias_itr]],
                boxprops=dict(lw=2, color=color),
                medianprops=dict(lw=2, color='black'),
                whiskerprops=dict(lw=2, color=color),
                capprops=dict(lw=2, color=color),
                widths=0.7,
                flierprops=dict(
                    marker='o', markerfacecolor=color, markersize=2, linestyle='none', markeredgecolor=color
                )
            )
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', linewidth=1)
            ax.set_title(col, fontsize=20)
        case_max = alias_itr
        for alias_itr, alias in enumerate(ctrl_aliases):
            color = colors[case_max + 1 + case_ctrl_color_diff + alias_itr]
            ax.boxplot(
                x=by_subj_df.loc[('CTRL', aliase2study[alias]), (y, col)].loc[
                    by_subj_df.loc[('CTRL', aliase2study[alias]), ('zscore', col)].abs() < outlier_th
                ],
                positions=[case_max + 3 + positions[alias_itr]],
                boxprops=dict(lw=2, color=color),
                medianprops=dict(lw=2, color='black'),
                whiskerprops=dict(lw=2, color=color),
                capprops=dict(lw=2, color=color),
                widths=0.7,
                flierprops=dict(
                    marker='o', markerfacecolor=color, markersize=2, linestyle='none', markeredgecolor=color
                )
            )
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', linewidth=1)
            ax.set_title(col, fontsize=20)
        ctrl_max = alias_itr
        if i == 0:
            if ylabel:
                ax.set_ylabel('Sample ' + ylabel, fontsize=20)
            ax.tick_params(labelsize=20)

        ax = axes[1, i]
        ax.boxplot(
            x=by_study_df.loc['CASE', (y, col)],
            positions=[max(ctrl_max, case_max) / 2],
            medianprops=dict(color='gray'),
            widths=max(ctrl_max, case_max) - 2,
            showfliers=False
        )
        case_positions = np.arange(max(ctrl_max, case_max) / 2 - case_max / 2 * 0.3, 100, 0.3)
        ax.scatter(
            x=case_positions[:sum(by_study_df.loc['CASE', ('zscore', col)].abs() < outlier_th)],
            y=by_study_df.loc['CASE', (y, col)].loc[
                by_study_df.loc['CASE', ('zscore', col)].abs() < outlier_th
            ],
            color=colors[:case_max + 1]
        )
        ax.boxplot(
            x=by_study_df.loc['CTRL', (y, col)],
            positions=[max(ctrl_max, case_max) + 2 + max(ctrl_max, case_max) / 2],
            medianprops=dict(color='gray'),
            widths=max(ctrl_max, case_max) - 2,
            showfliers=False
        )
        ctrl_positions = np.arange(2 + max(ctrl_max, case_max) * 1.5 - ctrl_max / 2 * 0.3, 100, 0.3)
        ax.scatter(
            x=ctrl_positions[:sum(by_study_df.loc['CTRL', ('zscore', col)].abs() < outlier_th)],
            y=by_study_df.loc['CTRL', (y, col)].loc[
                by_study_df.loc['CTRL', ('zscore', col)].abs() < outlier_th
            ],
            color=colors[case_max + 1 + case_ctrl_color_diff: case_max + 1 + case_ctrl_color_diff + ctrl_max + 1]
        )
        ax.grid(axis='y', linestyle='--', linewidth=1)
        ax.set_xticks(
            [max(ctrl_max, case_max) / 2, max(ctrl_max, case_max) + 2 + max(ctrl_max, case_max) / 2],
            ['CASE', 'CTRL'], fontsize=20, rotation='vertical'
        )
        if i == 0:
            if ylabel:
                ax.set_ylabel('Study ' + ylabel, fontsize=20)
            ax.tick_params(labelsize=20)
    y_max_row1, y_max_row2 = 0, 0
    for col_itr in range(len(by_subj_df.loc[:, y].columns)):
        y_max_row1 = max(y_max_row1, axes[0, col_itr].get_ylim()[1])
        y_max_row2 = max(y_max_row2, axes[1, col_itr].get_ylim()[1])
    for col_itr in range(len(by_subj_df.loc[:, y].columns)):
        axes[0, col_itr].set_ylim(0, y_max_row1)
        axes[1, col_itr].set_ylim(0, y_max_row2)


def plot_junction_length(ctrl_sequence_df, case_sequence_df, study2alias):
    case_sequence_df['Junction Length'] = case_sequence_df.junction_aa.str.len()
    case_sequence_df['label'] = 'CASE'
    ctrl_sequence_df['Junction Length'] = ctrl_sequence_df.junction_aa.str.len()
    ctrl_sequence_df['label'] = 'CTRL'
    df = case_sequence_df.append(ctrl_sequence_df)
    df['alias'] = df.study_id.apply(lambda x: study2alias[x])
    boxplot_features(
        df,
        df,
        'v_family',
        'Junction Length',
        ['IGHV4', 'IGHV3', 'IGHV1', 'IGHV2', 'IGHV5'],
        'figures/junction_length.pdf',
        supylabel=True
    )


def plot_features(feature_table, labels, n_case, n_control, output_file=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    feature_table = feature_table.loc[labels.index]
    points = pd.value_counts([(x[1].loc[labels.index[labels]].sum(), x[1].loc[labels.index[~labels]].sum()) for x in
                              feature_table.iteritems()])
    if (0, 0) in points.index:
        points = points.drop(index=(0, 0))
    s = points.apply(lambda x: math.log(x) * 100 + 100)
    df = pd.DataFrame(
        [list(map(lambda x: x[0], points.index)), list(map(lambda x: x[1], points.index)), s.to_list()],
        index=['x', 'y', 's']
    ).transpose()
    ax.scatter(
        x='x', y='y', s='s', data=df.loc[(df.y > n_control) & (df.x > n_control)]
    )
    ax.scatter(
        x='x', y='y', s='s', data=df.loc[(df.y > n_case) & (df.x <= n_control)]
    )
    ax.scatter(
        x='x', y='y', s='s', data=df.loc[(df.x > n_case) & (df.y <= n_control)]
    )

    max_tick = int(max(ax.get_xticks().max(), ax.get_yticks().max()))
    ax.set_xticks(range(max_tick), range(max_tick), fontsize=20)
    ax.set_yticks(range(max_tick), range(max_tick), fontsize=20)
    ax.set_aspect('equal')
    ax.set_xlabel("Case")
    ax.set_ylabel("Control")
    ax.set_xlim(-1, max_tick)
    ax.set_ylim(-1, max_tick)
    ax.set_xticks(range(max_tick))
    ax.set_xticklabels(range(max_tick), fontsize='20')
    ax.set_yticks(range(max_tick))
    ax.set_yticklabels(range(max_tick), fontsize='20')
    ax.set_aspect('equal')
    ax.grid(axis='both')

    if output_file:
        plt.savefig(output_file)

    # In[14]:


def plot_results(results, output_file=None):
    res_dict = {
        'Accuracy': results.accuracy_score,
        'Precision': results.precision_score,
        'Recall': results.recall_score
    }
    fig, ax = plt.subplots()
    ax.boxplot(res_dict.values(), flierprops=dict(markerfacecolor='r', marker='d'))
    ax.set_xticklabels(res_dict.keys())
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y', linestyle='--', linewidth=1)


def plot_results_side_by_side(results_l, name_l, figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("Set2", len(results_l))

    bp_l = []
    for i, (results, name) in enumerate(zip(results_l, name_l)):
        res_dict = {
            'Accuracy': results.accuracy_score,
            'Precision': results.precision_score,
            'Recall': results.recall_score
        }
        bp = ax.boxplot(
            res_dict.values(),
            positions=[i + 1, i + 1 + len(results_l) + 1, i + 1 + 2 * len(results_l) + 2],
            boxprops=dict(linewidth=2, color=colors[i]),
            medianprops=dict(linewidth=4, color=colors[i]),
            flierprops=dict(markeredgecolor=colors[i])
        )
        bp_l.append(bp)

    leg = ax.legend(
        [bp['boxes'][0] for bp in bp_l], name_l, handlelength=3, fontsize=20
    )
    for i, legobj in enumerate(leg.legendHandles):
        legobj.set_linewidth(10.0)
        legobj.set_color(colors[i])

    ax.set_xticks([(i + 1 + i * len(results_l) + (len(results_l) - 1) / 2) for i in range(3)])
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall'], fontsize=20, fontweight='bold')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels(np.round(np.arange(0, 1.1, 0.1), 2), fontsize=20)
    plt.grid(True, axis='y', linestyle='--', linewidth=1)


def plot_features_side_by_side(feature_table_dict, labels):
    fig, ax_arr = plt.subplots(1, 2, figsize=(15, 10))

    max_tick = 0
    for i, (name, feature_table) in enumerate(feature_table_dict.items()):

        ax = ax_arr[i]
        points = pd.value_counts(
            [(x[1].loc[labels.index[labels]].sum(), x[1].loc[labels.index[~labels]].sum()) for x in feature_table.iteritems()])
        if (0, 0) in points.index:
            points = points.drop(index=(0, 0))
        # bonus = pd.Series(range(100, 100*len(points.unique())+100, 100), index=np.sort(points.unique()))
        s = points.apply(lambda x: math.log(x) * 100 + 100)  # bonus[x])
        df = pd.DataFrame(
            [list(map(lambda x: x[0], points.index)), list(map(lambda x: x[1], points.index)), s.to_list()],
            index=['x', 'y', 's']
        ).transpose()
        ax.scatter(
            x='x', y='y', s='s', data=df.loc[(df.y > 0) & (df.x > 0)]
        )
        ax.scatter(
            x='x', y='y', s='s', data=df.loc[df.x == 0]
        )
        ax.scatter(
            x='x', y='y', s='s', data=df.loc[df.y == 0]
        )

        max_tick = max(max_tick, int(max(ax.get_xticks().max(), ax.get_yticks().max())))
        ax.set_xticks(range(max_tick), range(max_tick), fontsize=20)
        ax.set_yticks(range(max_tick), range(max_tick), fontsize=20)
        ax.set_aspect('equal')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title('(' + name + ')', fontsize=20, loc='left')

    for ax in ax_arr:
        ax.set_xlim(-1, max_tick)
        ax.set_ylim(-1, max_tick)

        ax.set_xticks(range(max_tick))
        ax.set_xticklabels(range(max_tick), fontsize='20')
        ax.set_yticks(range(max_tick))
        ax.set_yticklabels(range(max_tick), fontsize='20')
        ax.set_aspect('equal')
        ax.grid(axis='both')

    fig.supxlabel('Case', fontsize=20)
    fig.supylabel('Control', fontsize=20)


def create_studies_legend_handles(aliases, colors):

    handles = []
    for alias_itr, alias in enumerate(aliases):
        handles.append(
            Line2D(
                [0], [0], marker='o', color='w', label=alias, markerfacecolor=colors[alias_itr], markersize=10
            )
        )
    return handles


def boxplot_junction_aa_length(
    axes, case_airr_seq_df: pd.DataFrame,
    ctrl_airr_seq_df: pd.DataFrame,
    v_families: typing.Iterable,
    case_aliases: typing.Iterable,
    ctrl_aliases: typing.Iterable,
    aliase2study: dict,
    colors: list,
    case_ctrl_color_diff=3
):
    by_v_family_by_study_junction_len = pd.Series(
        dtype=object, name='v_family', index=pd.MultiIndex.from_product([['CASE', 'CTRL'], v_families])
    )
    for v_family, frame_index in case_airr_seq_df.index.groupby(case_airr_seq_df.v_call_original.apply(getFamily)).items():
        if v_family not in v_families:
            continue
        by_v_family_by_study_junction_len[
            ('CASE', v_family)
        ] = case_airr_seq_df.loc[frame_index].groupby('study_id').apply(lambda x: x.junction_aa_length.astype(int).tolist())
    for v_family, frame_index in ctrl_airr_seq_df.index.groupby(ctrl_airr_seq_df.v_call_original.apply(getFamily)).items():
        if v_family not in v_families:
            continue
        by_v_family_by_study_junction_len[
            ('CTRL', v_family)
        ] = ctrl_airr_seq_df.loc[frame_index].groupby('study_id').apply(lambda x: x.junction_aa_length.astype(int).tolist())
    positions = np.arange(0, 10, 1)
    for i, v_family in enumerate(v_families):
        ax = axes[0, i]
        for alias_itr, alias in enumerate(case_aliases):
            color = colors[alias_itr]
            ax.boxplot(
                x=by_v_family_by_study_junction_len.loc[('CASE', v_family)].loc[aliase2study[alias]],
                positions=[positions[alias_itr]],
                boxprops=dict(lw=2, color=color),
                medianprops=dict(lw=2, color='black'),
                whiskerprops=dict(lw=2, color=color),
                capprops=dict(lw=2, color=color),
                widths=0.7,
                flierprops=dict(
                    marker='o', markerfacecolor=color, markersize=2, linestyle='none', markeredgecolor=color
                )
            )
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', linewidth=1)
            ax.set_title(v_family, fontsize=20)
        case_max = alias_itr
        for alias_itr, alias in enumerate(ctrl_aliases):
            color = colors[case_max + 1 + case_ctrl_color_diff + alias_itr]
            ax.boxplot(
                x=by_v_family_by_study_junction_len.loc[('CTRL', v_family)].loc[aliase2study[alias]],
                positions=[case_max + 3 + positions[alias_itr]],
                boxprops=dict(lw=2, color=color),
                medianprops=dict(lw=2, color='black'),
                whiskerprops=dict(lw=2, color=color),
                capprops=dict(lw=2, color=color),
                widths=0.7,
                flierprops=dict(
                    marker='o', markerfacecolor=color, markersize=2, linestyle='none', markeredgecolor=color
                )
            )
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', linewidth=1)
            ax.set_title(v_family, fontsize=20)
        ctrl_max = alias_itr
        if i == 0:
            ax.set_ylabel('Study Junction Length', fontsize=20)
            ax.tick_params(labelsize=20)

        ax = axes[1, i]
        ax.boxplot(
            x=sum(by_v_family_by_study_junction_len.loc[('CASE', v_family)].tolist(), []),
            positions=[max(ctrl_max, case_max) / 2],
            medianprops=dict(color='gray'),
            widths=max(ctrl_max, case_max) - 2,
            showfliers=True
        )
        ax.boxplot(
            x=sum(by_v_family_by_study_junction_len.loc[('CTRL', v_family)].tolist(), []),
            positions=[max(ctrl_max, case_max) + 2 + max(ctrl_max, case_max) / 2],
            medianprops=dict(color='gray'),
            widths=max(ctrl_max, case_max) - 2,
            showfliers=True
        )
        ax.grid(axis='y', linestyle='--', linewidth=1)
        ax.set_xticks(
            [max(ctrl_max, case_max) / 2, max(ctrl_max, case_max) + 2 + max(ctrl_max, case_max) / 2],
            ['CASE', 'CTRL'], fontsize=20, rotation='vertical'
        )
        if i == 0:
            ax.set_ylabel('Junction Length', fontsize=20)
            ax.tick_params(labelsize=20)