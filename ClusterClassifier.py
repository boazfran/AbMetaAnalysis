"""
Classify junction_aa clusters
"""

# Info
__author__ = 'Boaz Frankel'

# Imports
import pandas
import pandas as pd
import os
import numpy as np
from changeo.Gene import getFamily, getGene
from sklearn.naive_bayes import BernoulliNB
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.proportion import multinomial_proportions_confint

# MetaAnalysis imports
import sys
sys.path.append('/work/boazfr/dev/packages')
from MetaAnalysis.Clustering import add_cluster_id, match_cluster_id
from MetaAnalysis.Utilities import build_feature_table
from MetaAnalysis.EDA import mannwhitneyu_test


class ClusterClassifier:

    def __init__(self, clf, selected_v_gene, selected_kmers, k, kmer2cluster_func, default_label):
        self.clf = clf
        self.selected_v_gene = selected_v_gene
        self.selected_kmers = selected_kmers
        self.k = k
        self.kmer2cluster_func = kmer2cluster_func
        self.default_label = default_label

    def predict(self, sequences):

        v_gene = sequences.v_gene.apply(lambda x: pd.Series(x == self.selected_v_gene, index=self.selected_v_gene)).astype(int)
        kmers = sequences.junction_aa.apply(
            lambda x: pd.value_counts(
                list(filter(lambda x: x in self.selected_kmers, [self.kmer2cluster_func(x[i:i+self.k]) for i in range(2, len(x)-self.k)]))
            ) if len(x) >= self.k+3 else pd.Series()
        ).append(pd.DataFrame(columns=self.selected_kmers)).loc[:, self.selected_kmers].fillna(0).astype(bool).astype(int)
        embedding = pd.concat([v_gene, kmers], axis=1)
        predicts = self.clf.predict(embedding) == 1
        predicts[embedding.sum(axis=1) == 0] = self.default_label
        return pd.Series(sequences.cluster_id.unique(), index=sequences.cluster_id.astype(str).unique()).apply(
            lambda x: sum(sequences.loc[predicts].cluster_id == x) > 0
        )
        

def create_cluster_classifier(
        train_sequence_df: pd.DataFrame,
        train_feature_table: pd.DataFrame,
        train_labels: pd.Series,
        default_label: bool,
        k: int,
        kmer2cluster_func=lambda x: x,
) -> ClusterClassifier:
    """

    :param train_sequence_df:
    :param train_feature_table:
    :param train_labels:
    :param default_label: default cluster label when all features are zero
    :param k:
    :param kmer2cluster_func:
    :return:
    """
    case_sequence_df = train_sequence_df.loc[
        train_sequence_df.cluster_id.isin(
            train_feature_table.loc[
                :,
                (
                   (train_feature_table.loc[train_labels.index[train_labels]].sum() > 0) &
                   (train_feature_table.loc[train_labels.index[~train_labels]].sum() == 0)
                )
            ].columns
        )
    ]
    ctrl_sequence_df = train_sequence_df.loc[
        train_sequence_df.cluster_id.isin(
            train_feature_table.loc[
            :, (train_feature_table.loc[train_labels.index[~train_labels]].sum() > 0) & (
                        train_feature_table.loc[train_labels.index[train_labels]].sum() == 0)
            ].columns
        )
    ]

    print(f'segmenting to {k}-mers')
    case_kmers = pd.Series(dtype=int)
    for s, e in zip(range(0, len(case_sequence_df), 1000), range(1000, len(case_sequence_df), 1000)):
        case_kmers = case_kmers.add(
            case_sequence_df.junction_aa[s:e].apply(
                lambda x: pd.value_counts([kmer2cluster_func(x[i:i + k]) for i in range(2, len(x) - k)]) if len(
                    x) >= k + 3 else pd.Series()).sum(), fill_value=0
        )
    case_kmers = case_kmers.add(
        case_sequence_df.junction_aa[e:len(case_sequence_df)].apply(
            lambda x: pd.value_counts([kmer2cluster_func(x[i:i + k]) for i in range(2, len(x) - k)]) if len(
                x) >= k + 3 else pd.Series()).sum(), fill_value=0
    )
    case_kmers = pd.DataFrame(case_kmers).transpose().astype(int)
    ctrl_kmers = pd.Series(dtype=int)
    for s, e in zip(range(0, len(ctrl_sequence_df), 1000), range(1000, len(ctrl_sequence_df), 1000)):
        ctrl_kmers = ctrl_kmers.add(
            ctrl_sequence_df.junction_aa[s:e].apply(
                lambda x: pd.value_counts([kmer2cluster_func(x[i:i + k]) for i in range(2, len(x) - k)]) if len(
                    x) >= k + 3 else pd.Series()).sum(), fill_value=0
        )
    ctrl_kmers = ctrl_kmers.add(
        ctrl_sequence_df.junction_aa[e:len(ctrl_sequence_df)].apply(
            lambda x: pd.value_counts([kmer2cluster_func(x[i:i + k]) for i in range(2, len(x) - k)]) if len(
                x) >= k + 3 else pd.Series()).sum(), fill_value=0
    )
    ctrl_kmers = pd.DataFrame(ctrl_kmers).transpose().astype(int)

    case_kmers = case_kmers.append(pd.DataFrame(columns=ctrl_kmers.columns)).fillna(0)
    case_kmers = case_kmers.loc[:, np.sort(case_kmers.columns)]
    ctrl_kmers = ctrl_kmers.append(pd.DataFrame(columns=case_kmers.columns)).fillna(0)
    ctrl_kmers = ctrl_kmers.loc[:, case_kmers.columns]

    kmers_ci = pd.DataFrame(columns=['case_lwr_ci', 'case_upr_ci', 'ctrl_lwr_ci', 'ctrl_upr_ci'],
                            index=case_kmers.columns)
    kmers_ci.loc[:, ['case_lwr_ci', 'case_upr_ci']] = multinomial_proportions_confint(case_kmers.iloc[0], alpha=0.05)
    kmers_ci.loc[:, ['ctrl_lwr_ci', 'ctrl_upr_ci']] = multinomial_proportions_confint(ctrl_kmers.iloc[0], alpha=0.05)
    selected_kmers = kmers_ci.loc[kmers_ci.case_lwr_ci > kmers_ci.ctrl_upr_ci].index.tolist()
    print(f'selected {k}-mers: {selected_kmers}')

    case_kmers = case_sequence_df.junction_aa.apply(
        lambda x: pd.value_counts(list(filter(lambda x: x in selected_kmers,
                                              [kmer2cluster_func(x[i:i + k]) for i in range(2, len(x) - k)]))) if len(
            x) >= k + 3 else pd.Series()
    ).append(pd.DataFrame(columns=selected_kmers)).loc[:, selected_kmers].fillna(0).astype(bool).astype(int)
    ctrl_kmers = ctrl_sequence_df.junction_aa.apply(
        lambda x: pd.value_counts(list(filter(lambda x: x in selected_kmers,
                                              [kmer2cluster_func(x[i:i + k]) for i in range(2, len(x) - k)]))) if len(
            x) >= k + 3 else pd.Series()
    ).append(pd.DataFrame(columns=selected_kmers)).loc[:, selected_kmers].fillna(0).astype(bool).astype(int)

    print('selecting v_gene features')
    case_v_gene = pd.DataFrame(
        case_sequence_df.groupby(['study_id', 'subject_id']).v_gene.value_counts(normalize=True)).transpose()
    ctrl_v_gene = pd.DataFrame(
        ctrl_sequence_df.groupby(['study_id', 'subject_id']).v_gene.value_counts(normalize=True)).transpose()
    case_v_gene = case_v_gene.append(pd.DataFrame(columns=ctrl_v_gene.columns)).fillna(0)
    ctrl_v_gene = ctrl_v_gene.append(pd.DataFrame(columns=case_v_gene.columns)).fillna(0)
    case_v_gene = case_v_gene.transpose().rename(columns={'v_gene': 'Freq'}).reset_index().sort_values('v_gene')
    ctrl_v_gene = ctrl_v_gene.transpose().rename(columns={'v_gene': 'Freq'}).reset_index().sort_values('v_gene')

    case_v_gene.loc[:, 'zscore'] = case_v_gene.groupby('study_id').apply(
        lambda x: pd.Series(((x.Freq - x.Freq.mean()) / x.Freq.std()) if (x.Freq.std() != 0) else 0, index=x.index,
                            name='zscore')
    ).reset_index().set_index('level_1').zscore
    case_v_gene = case_v_gene.loc[case_v_gene.zscore.apply(abs) < 3]
    ctrl_v_gene.loc[:, 'zscore'] = ctrl_v_gene.groupby('study_id').apply(
        lambda x: pd.Series(((x.Freq - x.Freq.mean()) / x.Freq.std()) if (x.Freq.std() != 0) else 0, index=x.index,
                            name='zscore')
    ).reset_index().set_index('level_1').zscore
    ctrl_v_gene = ctrl_v_gene.loc[ctrl_v_gene.zscore.apply(abs) < 3]

    case_v_gene['label'] = 'CASE'
    ctrl_v_gene['label'] = 'CTRL'
    selected_v_gene = mannwhitneyu_test(case_v_gene.append(ctrl_v_gene), 'v_gene', 0.1, 'two-sided')
    selected_v_gene = selected_v_gene.loc[selected_v_gene.reject].index
    print(f'selected v_genes: {selected_v_gene}')

    case_v_gene = case_sequence_df.v_gene.apply(
        lambda x: pd.Series(x == selected_v_gene, index=selected_v_gene)).astype(int)
    ctrl_v_gene = ctrl_sequence_df.v_gene.apply(
        lambda x: pd.Series(x == selected_v_gene, index=selected_v_gene)).astype(int)

    train_embedding = pd.concat(
        [
            case_v_gene.append(ctrl_v_gene),
            case_kmers.append(ctrl_kmers)
        ], axis=1
    )
    train_labeling = np.concatenate([np.ones(len(case_v_gene)), np.zeros(len(ctrl_v_gene))]) == 1

    return ClusterClassifier(
        BernoulliNB(fit_prior=True).fit(train_embedding, train_labeling),
        selected_v_gene,
        selected_kmers,
        k,
        kmer2cluster_func,
        default_label
    )
