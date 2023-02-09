"""
Utility functions for dataset Elementary Data Analysis
"""

# Info
__author__ = 'Boaz Frankel'

# Imports
from scipy.stats import mannwhitneyu, ttest_ind_from_stats, permutation_test
from statsmodels.stats.proportion import multinomial_proportions_confint
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests
from changeo.Gene import getFamily, getGene, getAllele
import os
import math
import numpy as np
from scipy.spatial.distance import cdist, pdist
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict
from scipy import sparse as sp


# AbMetaAnalysis imports
import sys
sys.path.append('/work/boazfr/dev/packages')
from AbMetaAnalysis.Utilities import build_feature_table, filter_airr_seq_df_by_labels, get_imgt_allele, load_sampled_airr_seq_df
from AbMetaAnalysis.Clustering import add_cluster_id, sequence_series_to_numeric_array, add_cluster_id, save_distance_matrices


def mannwhitneyu_test(
    df: pd.DataFrame,
    confidence_th: float = 0.05,
    alternative: str = 'greater',
    adjustment_method: str = 'holm-sidak',
    y: str = 'prop'
):
    """
    perform none parametric mannwhitneyu test
    :param df: data frame with label field (CASE/CTRL)
    :param confidence_th: the confidence level for which to reject the null hypothesis
    :param alternative: one of greater/lower/two-sided
    :param adjustment_method: what adjustment method to use for the multiple tests p value correction
    :param y: field for which to perform the test statistic
    :return: data frame with the p_value and corrected_p_value for each group
    """
    p_values = df[y].apply(lambda x: getattr(
        mannwhitneyu(x.loc['CASE'], x.loc['CTRL'], alternative=alternative),
        'pvalue'
    ))
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values.to_list(), confidence_th, method=adjustment_method)
    return pd.DataFrame(
        [p_values.to_list(), pvals_corrected.tolist(), reject.tolist()],
        index=['pvalue', 'pvalue_corrected', 'reject'],
        columns=p_values.index
    ).transpose().sort_values('pvalue')


def count_v_family_by_subj(airr_seq_df, v_call_field="v_call_original"):
    return airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda x: pd.DataFrame(x[v_call_field].apply(getFamily).value_counts()).transpose()
    ).droplevel(2).fillna(0)


def count_v_gene_by_subj(airr_seq_df, v_call_field="v_call_original"):
    return airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda x: pd.DataFrame(x[v_call_field].apply(getGene).value_counts()).transpose()
    ).droplevel(2).fillna(0)


def count_d_gene_by_subj(airr_seq_df):
    return airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda x: pd.DataFrame(x.d_call.apply(getGene).value_counts()).transpose()
    ).droplevel(2).fillna(0)


def count_d_allele_by_subj(airr_seq_df):
    return airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda x: pd.DataFrame(x.d_call.apply(getAllele).value_counts()).transpose()
    ).droplevel(2).fillna(0)


def count_j_gene_by_subj(airr_seq_df):
    return airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda x: pd.DataFrame(x.j_call.apply(getGene).value_counts()).transpose()
    ).droplevel(2).fillna(0)


def count_j_allele_by_subj(airr_seq_df):
    return airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda x: pd.DataFrame(x.j_call.apply(getAllele).value_counts()).transpose()
    ).droplevel(2).fillna(0)


def count_junction_length_by_subj(airr_seq_df):
    return airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda x: pd.DataFrame(x.junction_aa.str.len().value_counts()).transpose()
    ).droplevel(2).fillna(0)


def get_kmers_set(airr_seq_df, k):
    kmers = set()
    for i in range(0, len(airr_seq_df), 1000):
        kmers.update(
            sum(
                airr_seq_df.iloc[i:min(i+1000, len(airr_seq_df))].junction_aa.apply(
                    lambda x: [x[j:j+k] for j in range(2, len(x)-k)] if len(x) >= k + 3 else []
                ).to_list(),
                []
            )
        )
    return kmers


def count_kmers_by_subj(airr_seq_df, k):

    kmers_set = get_kmers_set(airr_seq_df, k)
    dv = DictVectorizer(sparse=True).fit([OrderedDict.fromkeys(kmers_set, 1)])
    by_subject_kmers = airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda frame: sp.csr_matrix(
            dv.transform(
                frame.junction_aa.apply(
                    lambda x: Counter([x[j:j+k] for j in range(2, len(x)-k)] if len(x) >= k + 3 else [])
                )
            ).sum(axis=0).getA1()
        )
    )

    return pd.DataFrame(
        [by_subject_kmers[i].tocsr().toarray()[0] for i in range(len(by_subject_kmers))],
        index=by_subject_kmers.index,
        columns=dv.feature_names_
    )


def count_kmers_clusters(airr_seq_df, kmer2id, k):

    dv = DictVectorizer(sparse=True).fit([OrderedDict.fromkeys(kmer2id.unique(), 1)])
    by_subject_kmers = airr_seq_df.groupby(['study_id', 'subject_id']).apply(
        lambda frame: sp.csr_matrix(
            dv.transform(
                frame.junction_aa.apply(
                    lambda x: Counter([kmer2id[x[j:j+k]] for j in range(2, len(x)-k)] if len(x) >= k + 3 else [])
                )
            ).sum(axis=0).getA1()
        )
    )

    return pd.DataFrame(
        [by_subject_kmers[i].tocsr().toarray()[0] for i in range(len(by_subject_kmers))],
        index=by_subject_kmers.index,
        columns=dv.feature_names_
    )


def get_by_subj_prop_df(by_subj_occur_df: pd.DataFrame):

    return by_subj_occur_df.div(by_subj_occur_df.sum(axis=1), axis=0)


def get_by_subj_zscore_df(by_subj_prop_df: pd.DataFrame):

    return pd.concat(
        [
            pd.concat(
                {
                    frame_index: (by_subj_prop_df.loc[frame_index] - by_subj_prop_df.loc[frame_index].mean()).div(
                        by_subj_prop_df.loc[frame_index].std(axis=0), axis=1
                    )
                },
                names=['study_id']
            )
            for frame_index in by_subj_prop_df.index.groupby(by_subj_prop_df.index.get_level_values(0))
        ]
    ).loc[by_subj_prop_df.index].fillna(0)


def get_by_subj_motif_analysis_df(occur_df):
    prop_df = get_by_subj_prop_df(occur_df)
    zscore_df = get_by_subj_zscore_df(prop_df)
    occur_df.columns = pd.MultiIndex.from_product([['occur'], occur_df.columns])
    prop_df.columns = pd.MultiIndex.from_product([['prop'], prop_df.columns])
    zscore_df.columns = pd.MultiIndex.from_product([['zscore'], zscore_df.columns])
    return pd.concat([occur_df, prop_df, zscore_df], axis=1)


def get_by_study_motif_analysis_df(by_subj_motif_df, outlier_th=3):
    occur_df = by_subj_motif_df['occur'][by_subj_motif_df['zscore'].abs() < outlier_th].groupby('study_id').apply(lambda x: x.sum(axis=0))
    prop_df = by_subj_motif_df['prop'][by_subj_motif_df['zscore'].abs() < outlier_th].groupby('study_id').apply(lambda x: x.mean(axis=0))
    zscore_df = (prop_df - prop_df.mean()).div(prop_df.std(axis=0), axis=1)
    occur_df.columns = pd.MultiIndex.from_product([['occur'], occur_df.columns])
    prop_df.columns = pd.MultiIndex.from_product([['prop'], prop_df.columns])
    zscore_df.columns = pd.MultiIndex.from_product([['zscore'], zscore_df.columns])
    return pd.concat([occur_df, prop_df, zscore_df], axis=1)


def get_by_label_motif_analysis_df(by_study_motif_df, label, outlier_th=3, alpha=0.05):
    occur_df = by_study_motif_df['occur'][by_study_motif_df['zscore'].abs() < outlier_th].apply(lambda x: x.sum(axis=0))
    occur_df.name = 'occur'
    ci_df = pd.DataFrame(
        multinomial_proportions_confint(occur_df.values, alpha=alpha),
        index=occur_df.index,
        columns=['lwr_ci', 'upr_ci'],
    )
    ci_df['occur'] = occur_df
    ci_df['prop'] = ci_df.occur / ci_df.occur.sum()
    ci_df.columns = pd.MultiIndex.from_product([[label], ci_df.columns])
    ci_df = ci_df.transpose()
    ci_df.index.rename(['label', 'metric'], inplace=True)
    return ci_df


def get_motif_analysis(
    case_airr_seq_df: pd.DataFrame,
    ctrl_airr_seq_df: pd.DataFrame,
    motif_occur_func,
    outlier_th=3,
    ci_alpha=0.05
):
    """

    :param case_airr_seq_df:
    :param ctrl_airr_seq_df:
    :param motif_occur_func:
    :param outlier_th:
    :param ci_alpha:
    :return:
    """
    case_motif_df = motif_occur_func(case_airr_seq_df)
    ctrl_motif_df = motif_occur_func(ctrl_airr_seq_df)
    case_motif_df = case_motif_df.append(pd.DataFrame(columns=ctrl_motif_df.columns)).fillna(0)
    ctrl_motif_df = ctrl_motif_df.append(pd.DataFrame(columns=case_motif_df.columns)).fillna(0)
    case_motif_df = get_by_subj_motif_analysis_df(case_motif_df)
    case_motif_df = pd.concat({'CASE': case_motif_df}, names=['label'])
    ctrl_motif_df = get_by_subj_motif_analysis_df(ctrl_motif_df)
    ctrl_motif_df = pd.concat({'CTRL': ctrl_motif_df}, names=['label'])
    by_subj_motif_df = case_motif_df.append(ctrl_motif_df)
    del case_motif_df, ctrl_motif_df

    case_motif_df = get_by_study_motif_analysis_df(by_subj_motif_df.loc['CASE'], outlier_th=outlier_th)
    ctrl_motif_df = get_by_study_motif_analysis_df(by_subj_motif_df.loc['CTRL'], outlier_th=outlier_th)
    case_motif_df = pd.concat({'CASE': case_motif_df}, names=['label'])
    ctrl_motif_df = pd.concat({'CTRL': ctrl_motif_df}, names=['label'])
    by_study_motif_df = case_motif_df.append(ctrl_motif_df)
    del case_motif_df, ctrl_motif_df

    by_label_motif_df = pd.concat(
        [
            get_by_label_motif_analysis_df(by_study_motif_df.loc['CASE'], 'CASE', outlier_th=outlier_th, alpha=ci_alpha),
            get_by_label_motif_analysis_df(by_study_motif_df.loc['CTRL'], 'CTRL', outlier_th=outlier_th, alpha=ci_alpha)
        ], axis=0
    )

    return by_subj_motif_df, by_study_motif_df, by_label_motif_df


def load_motifs_analysis(motifs_dir: str, base_name: str, motif: str):
    """

    :param motifs_dir:
    :param base_name:
    :param motif:
    :return:
    """
    by_subj_motif_df, by_study_motif_df, by_label_motif_df = None, None, None
    by_subj_motif_df_file_path = os.path.join(motifs_dir, '_'.join([base_name, 'by_subj', motif + '.csv.gz']))
    if os.path.isfile(by_subj_motif_df_file_path):
        by_subj_motif_df = pd.read_csv(by_subj_motif_df_file_path, header=[0, 1]).set_index(
            [('label', 'Unnamed: 0_level_1'), ('study_id', 'Unnamed: 1_level_1'), ('subject_id', 'Unnamed: 2_level_1')]
        )
        by_subj_motif_df.index.rename(['label', 'study_id', 'subject_id'], inplace=True)
    by_study_motif_df_file_path = os.path.join(motifs_dir, '_'.join([base_name, 'by_study', motif + '.csv.gz']))
    if os.path.isfile(by_study_motif_df_file_path):
        by_study_motif_df = pd.read_csv(by_study_motif_df_file_path, header=[0, 1]).set_index(
            [('label', 'Unnamed: 0_level_1'), ('study_id', 'Unnamed: 1_level_1')]
        )
        by_study_motif_df.index.rename(['label', 'study_id'], inplace=True)
    by_label_motif_df_file_path = os.path.join(motifs_dir, '_'.join([base_name, 'by_label', motif + '.csv.gz']))
    if os.path.isfile(by_label_motif_df_file_path):
        by_label_motif_df = pd.read_csv(by_label_motif_df_file_path).set_index(['label', 'metric'])

    return by_subj_motif_df, by_study_motif_df, by_label_motif_df


def save_motifs_analysis(
    by_subj_motif_df: pd.DataFrame, by_study_motif_df: pd.DataFrame, by_label_motif_df: pd.DataFrame,
    motifs_dir: str, base_name: str, motif: str
):
    """

    :param by_subj_motif_df:
    :param by_study_motif_df:
    :param by_label_motif_df:
    :param motifs_dir:
    :param base_name:
    :param motif:
    :return:
    """

    by_subj_motif_df_file_path = os.path.join(motifs_dir, '_'.join([base_name, 'by_subj', motif + '.csv.gz']))
    by_subj_motif_df.reset_index().to_csv(by_subj_motif_df_file_path, index=False, compression='gzip')
    by_study_motif_df_file_path = os.path.join(motifs_dir, '_'.join([base_name, 'by_study', motif + '.csv.gz']))
    by_study_motif_df.reset_index().to_csv(by_study_motif_df_file_path, index=False, compression='gzip')
    by_label_motif_df_file_path = os.path.join(motifs_dir, '_'.join([base_name, 'by_label', motif + '.csv.gz']))
    by_label_motif_df.to_csv(by_label_motif_df_file_path, compression='gzip')

    return by_subj_motif_df, by_study_motif_df, by_label_motif_df


def compare_to_reference_df(
    case_airr_seq_df: pd.DataFrame,
    ctrl_airr_seq_df: pd.DataFrame,
    ref_airr_seq_df: pd.DataFrame,
    min_dist_th:float=0.1
):
    """

    :param case_airr_seq_df:
    :param ctrl_airr_seq_df:
    :param ref_airr_seq_df:
    :param min_dist_th: normalized hamming distance threshold to consider two junction_aa sequences as a match
    :return:
    """
    case_matched_sequences = pd.DataFrame()
    ctrl_matched_sequences = pd.DataFrame()
    min_dist_df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([['CASE', 'CTRL'], ['support', 'absolute', 'percentage']]),
        index=case_airr_seq_df.junction_aa.str.len().sort_values().unique()
    )
    for junction_aa_length in case_airr_seq_df.junction_aa.str.len().sort_values().unique():
        if sum(ref_airr_seq_df.junction_aa.str.len() == junction_aa_length) == 0:
            continue
        dist_mat = cdist(
            sequence_series_to_numeric_array(case_airr_seq_df.junction_aa.loc[case_airr_seq_df.junction_aa.str.len() == junction_aa_length]),
            sequence_series_to_numeric_array(ref_airr_seq_df.junction_aa.loc[ref_airr_seq_df.junction_aa.str.len() == junction_aa_length]),
            "hamming"
        )
        min_dist = np.min(dist_mat, axis=1)
        min_dist_df.loc[junction_aa_length, ('CASE', 'absolute')] = sum(min_dist <= min_dist_th)
        min_dist_df.loc[junction_aa_length, ('CASE', 'percentage')] = sum(min_dist <= min_dist_th) / sum(
            case_airr_seq_df.junction_aa.str.len() == junction_aa_length
        )
        min_dist_df.loc[junction_aa_length, ('CASE', 'support')] = sum(case_airr_seq_df.junction_aa.str.len() == junction_aa_length)
        case_matched_sequences = case_matched_sequences.append(
            case_airr_seq_df.loc[case_airr_seq_df.junction_aa.str.len() == junction_aa_length].loc[min_dist <= min_dist_th]
        )

    for junction_aa_length in ctrl_airr_seq_df.junction_aa.str.len().sort_values().unique():
        if sum(ref_airr_seq_df.junction_aa.str.len() == junction_aa_length) == 0:
            continue
        dist_mat = cdist(
            sequence_series_to_numeric_array(ctrl_airr_seq_df.junction_aa.loc[ctrl_airr_seq_df.junction_aa.str.len() == junction_aa_length]),
            sequence_series_to_numeric_array(ref_airr_seq_df.junction_aa.loc[ref_airr_seq_df.junction_aa.str.len() == junction_aa_length]),
            "hamming"
        )
        min_dist = np.min(dist_mat, axis=1)
        min_dist_df.loc[junction_aa_length, ('CTRL', 'absolute')] = sum(min_dist <= min_dist_th)
        min_dist_df.loc[junction_aa_length, ('CTRL', 'percentage')] = sum(min_dist <= min_dist_th) / sum(
            ctrl_airr_seq_df.junction_aa.str.len() == junction_aa_length
        )
        min_dist_df.loc[junction_aa_length, ('CTRL', 'support')] = sum(ctrl_airr_seq_df.junction_aa.str.len() == junction_aa_length)
        ctrl_matched_sequences = ctrl_matched_sequences.append(
            ctrl_airr_seq_df.loc[ctrl_airr_seq_df.junction_aa.str.len() == junction_aa_length].loc[min_dist <= min_dist_th]
        )

    min_dist_df = min_dist_df.loc[(min_dist_df[('CASE', 'absolute')] > 0) | (min_dist_df[('CTRL', 'absolute')] > 0)]

    return min_dist_df, case_matched_sequences, ctrl_matched_sequences


def get_case_control_sequence_df(airr_seq_df, labels, dist_mat_dir, dist_th):

    airr_seq_df['cluster_id'] = add_cluster_id(
        airr_seq_df, dist_mat_dir, dist_th
    )
    feature_table = build_feature_table(airr_seq_df, airr_seq_df['cluster_id'])
    case_features = feature_table.columns[
        (feature_table.loc[labels.index[labels]].sum() > 0) & (feature_table.loc[labels.index[~labels]].sum() == 0)
    ]
    ctrl_features = feature_table.columns[
        (feature_table.loc[labels.index[~labels]].sum() > 0) & (feature_table.loc[labels.index[labels]].sum() == 0)
    ]
    case_airr_seq_df = airr_seq_df.loc[airr_seq_df.cluster_id.isin(case_features)]
    ctrl_airr_seq_df = airr_seq_df.loc[airr_seq_df.cluster_id.isin(ctrl_features)]

    return case_airr_seq_df, ctrl_airr_seq_df, feature_table



