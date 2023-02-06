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
import psutil
from changeo.Gene import getFamily, getGene
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold


# MetaAnalysis imports
import sys
sys.path.append('/work/boazfr/dev/packages')
from MetaAnalysis.ClusterClassifier import create_cluster_classifier
from MetaAnalysis.Clustering import add_cluster_id, match_cluster_id
from MetaAnalysis.Utilities import build_feature_table, filter_airr_seq_df_by_labels
from MetaAnalysis.Defaults import default_random_state
from MetaAnalysis.Defaults import ray_num_cpus_percentage, ray_object_store_memory_percentage


if not ray.is_initialized():
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            'working_dir': '/work/boazfr/dev/packages',
        },
        object_store_memory=int(psutil.virtual_memory().total*ray_object_store_memory_percentage),
        num_cpus=max(int(os.cpu_count()*ray_num_cpus_percentage), 1)
    )


def test_fold(
    airr_seq_df: pandas.DataFrame,
    train_labels: pd.Series,
    test_labels: pd.Series,
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
    :param test_labels: labels of the validation set
    :param dist_mat_dir: directory where the dataframe pairwise distance matrices are saved
    :param case_th: threshold for which test the cluster classification
    :param default_label: default cluster label when all features are zero
    :param k: k by which to perform k-mers segmentation
    :param kmer2cluster: mapping of k-mer to k-mer cluster_id, if None identity mapping will be used
    :return: data frame with the classification results, case/ctrl support, precision and recall
    """
    train_airr_seq_df = filter_airr_seq_df_by_labels(airr_seq_df, train_labels)
    test_sequence_df = filter_airr_seq_df_by_labels(airr_seq_df, test_labels)

    print('adding cluster id')
    train_cluster_assignment = add_cluster_id(train_airr_seq_df, dist_mat_dir, 0.2)

    print('building train_feature_table')
    train_feature_table = build_feature_table(train_airr_seq_df, train_cluster_assignment)

    print('selecting features')
    selected_features = train_feature_table.columns[
        (
            (train_feature_table.loc[train_labels.index[train_labels]].sum() == case_th) &
            (train_feature_table.loc[train_labels.index[~train_labels]].sum() == 0)
        )
    ]
    print('matching cluster id')
    test_cluster_assignment = match_cluster_id(
        train_airr_seq_df.loc[train_cluster_assignment.isin(selected_features)],
        train_cluster_assignment[train_cluster_assignment.isin(selected_features)],
        test_sequence_df,
        dist_mat_dir,
        dist_th=0.2
    )

    print('building test_feature_table')
    test_feature_table = test_sequence_df.groupby(['study_id', 'subject_id']).apply(
        lambda frame: pd.Series(selected_features, index=selected_features).apply(
            lambda cluster_id: sum(test_cluster_assignment[frame.index].str.find(f';{cluster_id};') != -1) > 0
        )
    )

    positive_test_features = test_feature_table.columns[
        (
            (test_feature_table.loc[test_labels.index[test_labels]].sum() > 1) &
            (test_feature_table.loc[test_labels.index[~test_labels]].sum() == 0)
        )
    ]
    positive_test_features = pd.Series(positive_test_features).astype(str)
    negative_test_features = test_feature_table.columns[
       test_feature_table.loc[test_labels.index[~test_labels]].sum() > 0
    ]
    negative_test_features = pd.Series(negative_test_features).astype(str)

    if (len(positive_test_features) == 0) & (len(negative_test_features) == 0):
        return pd.DataFrame()

    positive_test_samples = train_airr_seq_df.loc[
        train_cluster_assignment.astype(str).isin(positive_test_features)
    ].copy()
    positive_test_samples['cluster_id'] = train_cluster_assignment[
        train_cluster_assignment.astype(str).isin(positive_test_features)
    ]
    negative_test_samples = train_airr_seq_df.loc[
        train_cluster_assignment.astype(str).isin(negative_test_features)
    ].copy()
    negative_test_samples['cluster_id'] = train_cluster_assignment[
        train_cluster_assignment.astype(str).isin(negative_test_features)
    ]

    cluster_clf = create_cluster_classifier(
        train_airr_seq_df,
        train_cluster_assignment,
        train_feature_table,
        train_labels,
        default_label,
        k,
        (lambda x: x) if kmer2cluster is None else (lambda x: kmer2cluster[x])
    )
    feature_labels = pd.Series(True, index=positive_test_features).append(
        pd.Series(False, index=negative_test_features))
    predict_labels = cluster_clf.predict(positive_test_samples.append(negative_test_samples)).loc[feature_labels.index]
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


@ray.remote(max_retries=0)
def remote_test_fold(sequence_df, train_labels, validation_labels, dist_mat_dir, case_th, default_label, k, kmer2cluster):
    test_fold(sequence_df, train_labels, validation_labels, dist_mat_dir, case_th, default_label, k, kmer2cluster)


def test_cluster_classification(
    airr_seq_df: pd.DataFrame,
    labels: pd.Series,
    dist_mat_dir: str,
    output_dir: str,
    n_splits: int = 10,
    n_repeats: int = 10,
    case_th: int = 2,
    default_label: bool = True,
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
    :param output_dir: directory path to save the results 
    :param kmer2cluster: mapping of k-mer to k-mer cluster_id, if None identity mapping will be used
    :param case_th: threshold for which test the cluster classification
    :param default_label: default label to use for sequence with no found features (sort of primitive bias)
    :param k: k by which to perform k-mers segmentation
    :return: data frame with the classification results, case/ctrl support, precision and recall
    """
    airr_seq_df = ray.put(airr_seq_df)
    if kmer2cluster is not None:
        kmer2cluster = ray.put(kmer2cluster)
    result_ids = []
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=default_random_state)
    for i, (train_index, validation_index) in enumerate(rskf.split(labels, labels)):
        validation_labels = labels.iloc[validation_index]
        train_labels = labels.drop(index=validation_labels.index)
        result_ids.append(
            remote_test_fold.remote(
                airr_seq_df, train_labels, validation_labels, dist_mat_dir, case_th, default_label, k, kmer2cluster
            )
        )
    results = pd.concat([ray.get(result_id) for result_id in result_ids])
    results.to_csv(
        os.path.join(output_dir, f'cluster_classification_k-{k}_kmer_clustering-{kmer2cluster is not None}_results.csv')
    )

    return results
