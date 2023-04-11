# Info
__author__ = 'Boaz Frankel'

# Imports
import pandas as pd
import numpy as np
import sys
import ray
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
import os
import ray
import psutil

# AbMetaAnalysis Imports
sys.path.append('/work/boazfr/dev/packages/')
from AbMetaAnalysis.RepClassifier import RepClassifier
from AbMetaAnalysis.Utilities import filter_airr_seq_df_by_labels, build_feature_table, load_sampled_airr_seq_df
from AbMetaAnalysis.Clustering import add_cluster_id, match_cluster_id, save_distance_matrices
from AbMetaAnalysis.SubSample import sample_by_n_clusters, sample_by_n_sequences
from AbMetaAnalysis.Defaults import ray_num_cpus_percentage, ray_object_store_memory_percentage


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
    airr_seq_df,
    train_labels,
    test_labels,
    dist_mat_dir,
    dist_th,
    case_th,
    ctrl_th,
    feature_selection_cfg_values,
):
    """
    train and test rep classifier fold
    :param airr_seq_df: airr-seq data frame
    :param train_labels: the labels of the train set repertoire samples
    :param test_labels: the labels of the test set repertoire samples
    :param dist_mat_dir: directory path to the distance matrices files
    :param dist_th: normalized hamming distance cut off value for the hierarchical clustering
    :param case_th: number of shared case repertoire samples to exceed for cluster to be selected as feature
    :param ctrl_th: maximal number of shared ctrl repertoire samples for cluster to be selected as feature
    :param feature_selection_cfg_values: list of dictionaries with feature selection configuration
    :return: (data frame with the fold classification results, data frame with the fold test samples)
    """
    result_metrics = pd.DataFrame(
        columns=[
            'support', 'accuracy_score', 'recall_score', 'precision_score', 'f1-score', 'case_th', 'ctrl_th', 'dist_th', 'fs_method',
            'k', 'kmer_clustering'
        ]
    )
    result_folds = pd.DataFrame(index=test_labels.index.tolist() + ['case_th', 'ctrl_th', 'dist_th', 'fs_method', 'k', 'kmer_clustering'])

    train_airr_seq_df = filter_airr_seq_df_by_labels(airr_seq_df, train_labels)
    train_cluster_assignment = add_cluster_id(train_airr_seq_df, dist_mat_dir, dist_th=dist_th)
    test_airr_sq_df = filter_airr_seq_df_by_labels(airr_seq_df, test_labels)
    print('building feature table')
    train_feature_table = build_feature_table(train_airr_seq_df, train_cluster_assignment)

    for i, fs_cfg in enumerate(feature_selection_cfg_values):
        fs_method = fs_cfg['method']
        k = fs_cfg['k'] if fs_cfg == 'similar' else np.nan
        kmer2cluster = fs_cfg['kmer2cluster'] if (fs_cfg == 'similar') & ('kmer2cluster' in fs_cfg) else None

        print('creating rep classifier')
        rep_clf = RepClassifier(
            train_airr_seq_df, train_cluster_assignment, dist_th, case_th, ctrl_th, fs_method, k, kmer2cluster
        ).fit(
            train_feature_table, train_labels
        )
        test_cluster_assignment = match_cluster_id(
            train_airr_seq_df.loc[train_cluster_assignment.isin(rep_clf.selected_features)],
            train_cluster_assignment[train_cluster_assignment.isin(rep_clf.selected_features)],
            test_airr_sq_df,
            dist_mat_dir,
            dist_th
        )
        test_feature_table = test_airr_sq_df.groupby(['study_id', 'subject_id']).apply(
            lambda frame: pd.Series(rep_clf.selected_features, index=rep_clf.selected_features).apply(
                lambda cluster_id: sum(test_cluster_assignment[frame.index].str.find(f';{cluster_id};') != -1) > 0
            )
        ).loc[test_labels.index]
        predict_labels = rep_clf.predict(test_feature_table)
        result_metrics.loc[len(result_metrics), :] = [
            sum(test_labels),
            accuracy_score(test_labels, predict_labels),
            recall_score(test_labels, predict_labels),
            precision_score(test_labels, predict_labels, zero_division=0),
            f1_score(test_labels, predict_labels, zero_division=0),
            case_th,
            ctrl_th,
            dist_th,
            fs_method,
            k,
            True if kmer2cluster else None
        ]
        print(result_metrics.iloc[-1])

        # collect statistics on selected clusters and subjects classification
        result_folds.loc[
            np.array(test_labels.index.tolist() + ['case_th', 'ctrl_th', 'dist_th', 'fs_method', 'k', 'kmer_clustering'], dtype=object), i
        ] = (test_labels == predict_labels).to_list() + [case_th, ctrl_th, dist_th, fs_method, k, True if kmer2cluster else None]

    return result_metrics, result_folds.transpose()


@ray.remote(max_retries=0)
def remote_test_fold(
    airr_seq_df, train_labels, test_labels, dist_mat_dir, dist_th, case_th, ctrl_th, feature_selection_cfg_values
):

    return test_fold(
        airr_seq_df, train_labels, test_labels, dist_mat_dir, dist_th, case_th, ctrl_th, feature_selection_cfg_values
    )


def test_rep_classifier(
    airr_seq_df: pd.DataFrame,
    labels: pd.Series,
    dist_mat_dir: str,
    train_only_labels: pd.Series = None,
    n_splits: int = 10,
    n_repeats: int = 10,
    case_th_values: list = [1],
    ctrl_th_values: list = [0],
    dist_th_values: list = [0.2],
    feature_selection_cfg_values: list = [{'method': 'naive'}],
):
    """

    :param airr_seq_df: airr-seq data frame
    :param labels: repertoire samples labels
    :param dist_mat_dir: directory path to load/save distance matrices files
    :param train_only_labels: labels of repertoire samples that can only be used for training set and not for test
    :param n_splits: number of cross validation splits
    :param n_repeats: number of cross validation iterations
    :param case_th_values: list of shared case repertoire samples threshold for the feature selection
    :param ctrl_th_values: list of shared ctrl repertoire samples threshold for the feature selection
    :param dist_th_values: list of normalized hamming distance cut off values for the hierarchical clustering
    :param feature_selection_cfg_values: list of dictionaries with feature selection configuration
    :return: (data frame with the folds classification results, data frame with the folds test samples)
    """
    airr_seq_df = ray.put(airr_seq_df)
    for fs_cfg in feature_selection_cfg_values:
        if 'kmer_cluster_map' in fs_cfg:
            fs_cfg['kmer2cluster'] = ray.put(fs_cfg['kmer2cluster'])
    result_ids = []
    if (n_splits == len(labels)) and (n_repeats == 1):
        # leave one out cross validation
        train_validation_splits = [(list(filter(lambda x: x != i, range(len(labels)))), [i]) for i in range(len(labels))]
    else:
        # stratified repeated cross validation
        train_validation_splits = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42).split(labels, labels)

    for train_index, validation_index in train_validation_splits:
        validation_labels = labels[validation_index]
        train_labels = labels.drop(index=validation_labels.index)
        if train_only_labels is not None:
            train_labels = pd.concat([train_labels, train_only_labels])
        for case_th in case_th_values:
            for ctrl_th in ctrl_th_values:
                if ctrl_th > case_th:
                    continue
                for dist_th in dist_th_values:
                    result_ids.append(
                        remote_test_fold.remote(
                            airr_seq_df, train_labels, validation_labels, dist_mat_dir, dist_th, case_th, ctrl_th, feature_selection_cfg_values
                        )
                    )

    result_metrics, result_folds = pd.DataFrame(), pd.DataFrame()
    for result_id in result_ids:
        result_metrics_itr, result_folds_itr = ray.get(result_id)
        result_metrics = pd.concat([result_metrics, result_metrics_itr], ignore_index=True)
        result_folds = pd.concat([result_folds, result_folds_itr], ignore_index=True)

    return result_metrics, result_folds


def save_rep_clf_results(
    result_metrics: pd.DataFrame, result_folds: pd.DataFrame, output_dir: str, base_name: str
):
    """
    group the classification result data frames by hyper-parameters and saves frames as different files
    :param result_metrics: the data frame with the classification metrics
    :param result_folds: the data frame with the folds samples split information
    :param output_dir: output directory to save the files
    :param base_name: prefix for the saved files
    :return:
    """
    hyper_parameters = result_metrics.head(0).drop(
        columns=['support', 'accuracy_score', 'recall_score', 'precision_score', 'f1-score']
    ).columns.tolist()

    for params, frame in result_metrics.groupby(hyper_parameters, dropna=False):
        output_path = os.path.join(
            output_dir, '_'.join(
                [base_name] + [f'{k}-{v}' for k, v in filter(lambda kv: not pd.isna(kv[1]), zip(hyper_parameters, params))]
            ) + '_result_metrics.csv'
        )
        frame.drop(columns=hyper_parameters).to_csv(output_path, index=False)

    for params, frame in result_folds.groupby(hyper_parameters, dropna=False):
        output_path = os.path.join(
            output_dir, '_'.join(
                [base_name] + [f'{k}-{v}' for k, v in filter(lambda kv: not pd.isna(kv[1]), zip(hyper_parameters, params))]
            ) + '_result_folds.csv'
        )
        frame = frame.drop(columns=hyper_parameters).transpose().reset_index()
        frame.loc[:, ['study_id', 'subject_id']] = frame['index'].apply(lambda x: [x[0], x[1]]).to_list()
        frame.drop(columns='index').to_csv(output_path, index=False)


def load_rep_clf_results(
    results_dir: str,
    base_name: str,
    hyper_parameters_dict
):
    result_metrics, result_folds = None, None
    results_file_path = os.path.join(
        results_dir, '_'.join(
            [base_name] + [f'{k}-{v}' for k, v in hyper_parameters_dict.items()]
        ) + '_result_metrics.csv'
    )
    if os.path.isfile(results_file_path):
        result_metrics = pd.read_csv(results_file_path)
    folds_file_path = os.path.join(
        results_dir, '_'.join([base_name] + [f'{k}-{v}' for k, v in hyper_parameters_dict.items()]) + '_result_folds.csv'
    )
    if os.path.isfile(folds_file_path):
        result_folds = pd.read_csv(folds_file_path).set_index(['study_id', 'subject_id'])

    return result_metrics, result_folds
