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
import ray
from changeo.Gene import getFamily, getGene
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from multiprocessing import cpu_count


# MetaAnalysis imports
import sys
sys.path.append('/work/boazfr/dev/')
from MetaAnalysis.ClusterClassifier import create_cluster_classifier
from MetaAnalysis.Clustering import add_cluster_id, match_cluster_id
from MetaAnalysis.Utilities import build_feature_table, filter_airr_seq_df_by_labels
from MetaAnalysis.Defaults import default_random_state


if not ray.is_initialized():
    ray.init(
        ignore_reinit_error=True,
        runtime_env={'working_dir': '/work/boazfr/dev/', 'includes': ['/work/boazfr/dev/MetaAnalysis']},
        num_cpus=cpu_count()-1
    )


def test_fold(
    airr_seq_df: pandas.DataFrame,
    train_labels: pd.Series,
    validation_labels: pd.Series,
    dist_mat_dir: str,
    case_th: int,
    default_label: bool,
    k: int,
    kmer2cluster: dict = None,
) -> pd.DataFrame:
    """
    Test fold of cluster classification
    :param airr_seq_df: airr seq dataframe
    :param train_labels: labels of the train set
    :param validation_labels: labels of the validation set
    :param dist_mat_dir: directory where the dataframe pairwise distance matrices are saved
    :param case_th: threshold for which test the cluster classification
    :param default_label: default cluster label when all features are zero
    :param k: k by which to perform k-mers segmentation
    :param kmer2cluster: mapping of k-mer to k-mer cluster_id, if None identity mapping will be used
    :return: data frame with the classification results, case/ctrl support, precision and recall
    """
    train_airr_seq_df = filter_airr_seq_df_by_labels(airr_seq_df, train_labels)
    validation_sequence_df = filter_airr_seq_df_by_labels(airr_seq_df, validation_labels)

    print('adding cluster id')
    train_airr_seq_df = add_cluster_id(train_airr_seq_df, dist_mat_dir, 0.2)

    print('building train_feature_table')
    train_feature_table = build_feature_table(train_airr_seq_df)

    print('selecting features')
    selected_features = train_feature_table.columns[
        (
            (train_feature_table.loc[train_labels.index[train_labels]].sum() == case_th) &
            (train_feature_table.loc[train_labels.index[~train_labels]].sum() == 0)
        )
    ]
    print('matching cluster id')
    validation_sequence_df = match_cluster_id(
        train_airr_seq_df.loc[train_airr_seq_df.cluster_id.isin(selected_features)],
        validation_sequence_df,
        dist_mat_dir,
        dist_th=0.2
    )

    print('building validation_feature_table')
    validation_feature_table = validation_sequence_df.groupby(['study_id', 'subject_id']).apply(
        lambda x: pd.Series(selected_features, index=selected_features).apply(
            lambda i: sum(x.matched_clusters.str.find(f';{i};') != -1) > 0
        )
    )

    positive_validation_features = validation_feature_table.columns[
        (
            (validation_feature_table.loc[validation_labels.index[validation_labels]].sum() > 1) &
            (validation_feature_table.loc[validation_labels.index[~validation_labels]].sum() == 0)
        )
    ]
    positive_validation_features = pd.Series(positive_validation_features).astype(str)
    negative_validation_features = validation_feature_table.columns[
       validation_feature_table.loc[validation_labels.index[~validation_labels]].sum() > 0
    ].colmns
    negative_validation_features = pd.Series(negative_validation_features).astype(str)

    if (len(positive_validation_features) == 0) & (len(negative_validation_features) == 0):
        return pd.DataFrame()

    positive_validation_samples = train_airr_seq_df.loc[
        train_airr_seq_df.cluster_id.astype(str).isin(positive_validation_features)
    ]
    negative_validation_samples = train_airr_seq_df.loc[
        train_airr_seq_df.cluster_id.astype(str).isin(negative_validation_features)
    ]

    clf = create_cluster_classifier(
        train_airr_seq_df,
        train_feature_table,
        train_labels,
        default_label,
        k,
        (lambda x: x) if kmer2cluster is None else (lambda x: kmer2cluster[x])
    )
    feature_labels = pd.Series(True, index=positive_validation_features).append(
        pd.Series(False, index=negative_validation_features))
    predict_labels = clf.predict(positive_validation_samples.append(negative_validation_samples)).loc[
        feature_labels.index]
    res = pd.DataFrame(
        [
            sum(feature_labels),
            sum(~feature_labels),
            recall_score(feature_labels, predict_labels, pos_label=True, zero_division=0) if sum(
                feature_labels) else np.nan,
            recall_score(feature_labels, predict_labels, pos_label=False, zero_division=0) if sum(
                ~feature_labels) else np.nan,
            precision_score(feature_labels, predict_labels, pos_label=True, zero_division=0) if sum(
                feature_labels) else np.nan,
            precision_score(feature_labels, predict_labels, pos_label=False, zero_division=0) if sum(
                ~feature_labels) else np.nan,
        ],
        index=['case_support', 'control_support', 'case_recall', 'ctrl_recall', 'case_precision', 'ctrl_precision']
    ).transpose()

    print(res)

    return res


@ray.remote
def remote_test_fold(sequence_df, train_labels, validation_labels, dist_mat_dir, case_th, k, kmer2cluster):
    test_fold(sequence_df, train_labels, validation_labels, dist_mat_dir, case_th, k, kmer2cluster)


def test_cluster_classification(
    airr_seq_df: pd.DataFrame,
    labels: pd.Series,
    output_dir: str,
    n_splits: int = 10,
    n_repeats: int = 10,
    case_th: int = 2,
    k: int = 5,
    kmer2cluster: dict = None
):
    """
    Run repeated cross validation test folds of cluster classification and return data frame with all folds results
    :param airr_seq_df: airr-seq data frame
    :param labels: samples labels for the
    :param n_splits: number of cross validation splits
    :param n_repeats: number of cross validation iterations
    :param dist_mat_dir: directory where the dataframe pairwise distance matrices are saved
    :param kmer2cluster: mapping of k-mer to k-mer cluster_id, if None identity mapping will be used
    :param case_th: threshold for which test the cluster classification
    :param k: k by which to perform k-mers segmentation
    :return: data frame with the classification results, case/ctrl support, precision and recall
    """
    sequence_df_id = ray.put(airr_seq_df)
    if kmer2cluster is not None:
        kmer2cluster = ray.put(kmer2cluster)
    result_ids = []
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=default_random_state)
    for i, (train_index, validation_index) in enumerate(rskf.split(labels, labels)):
        validation_labels = labels.iloc[validation_index]
        train_labels = labels.drop(index=validation_labels.index)
        result_ids.append(
            test_fold.remote(sequence_df_id, validation_labels, train_labels, case_th, k, kmer2cluster)
        )
    results = pd.concat([ray.get(result_id) for result_id in result_ids])
    results.to_csv(
        os.path.join(output_dir, f'cluster_classification_k-{k}_kmer_clustering-{kmer2cluster is not None}_results.csv')
    )

    return results
