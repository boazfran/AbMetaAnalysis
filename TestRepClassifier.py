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

# MetaAnalysis Imports
sys.path.append('/work/boazfr/dev/packages/')
from MetaAnalysis.RepClassifier import RepClassifier
from MetaAnalysis.Utilities import filter_airr_seq_df_by_labels, build_feature_table, load_sampled_airr_seq_df
from MetaAnalysis.Clustering import add_cluster_id, match_cluster_id, save_distance_matrices
from MetaAnalysis.SubSample import sample_by_n_clusters, sample_by_n_sequences


if not ray.is_initialized():
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            'working_dir': '/work/boazfr/dev/packages',
        },
        object_store_memory=int(psutil.virtual_memory().total*0.1),
        num_cpus=max(int(os.cpu_count()*0.75), 1)
    )


def test_fold(
    airr_seq_df,
    train_labels,
    test_labels,
    dist_mat_dir,
    dist_th,
    case_th,
    ctrl_th,
    feature_selection_mode_values: list,
    k_values: list,
    kmer2cluster_values: list
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
    :param feature_selection_mode_values: select feature only using thresholds (naive) or use cluster clf (similar)
    :param k_values: k values for the k-mers segmentation, relevant only if similar is in feature_selection_mode_values
    :param kmer2cluster_values: k-mers to k-mers cluster id mapping, relevant only if similar is in feature_selection_mode_values
    :return: (data frame with the fold classification results, data frame with the fold test samples)
    """
    results = pd.DataFrame(
        columns=[
            'support', 'accuracy_score', 'recall_score', 'precision_score', 'f1-score', 'case_th', 'ctrl_th', 'dist_th', 'fs_mode',
            'k', 'kmer_clustering'
        ]
    )
    subject_table = pd.DataFrame(index=test_labels.index.tolist() + ['case_th', 'ctrl_th', 'dist_th', 'fs_mode', 'k', 'kmer_clustering'])

    train_airr_seq_df = filter_airr_seq_df_by_labels(airr_seq_df, train_labels)
    train_cluster_assignment = add_cluster_id(train_airr_seq_df, dist_mat_dir, dist_th=dist_th)
    test_airr_sq_df = filter_airr_seq_df_by_labels(airr_seq_df, test_labels)
    print('building feature table')
    train_feature_table = build_feature_table(train_airr_seq_df, train_cluster_assignment)

    if len(k_values) == 0:
        k_values = [-1]
    for i, fs_mode in enumerate(feature_selection_mode_values):
        k_l = [-1] if (fs_mode == "naive") else k_values
        for j, k in enumerate(k_l):
            kmer2cluster = kmer2cluster_values[j] if j < len(kmer2cluster_values) else None
            print('creating rep classifier')
            rep_clf = RepClassifier(
                train_airr_seq_df, train_cluster_assignment, dist_th, case_th, ctrl_th, fs_mode, k, kmer2cluster
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
            results.loc[len(results), :] = [
                sum(test_labels),
                accuracy_score(test_labels, predict_labels),
                recall_score(test_labels, predict_labels),
                precision_score(test_labels, predict_labels, zero_division=0),
                f1_score(test_labels, predict_labels, zero_division=0),
                case_th,
                ctrl_th,
                dist_th,
                fs_mode,
                k,
                kmer2cluster is not None
            ]
            print(results.iloc[-1])

            # collect statistics on selected clusters and subjects classification
            subject_table.loc[
                np.array(test_labels.index.tolist() + ['case_th', 'ctrl_th', 'dist_th', 'fs_mode', 'k', 'kmer_clustering'], dtype=object), i
            ] = (test_labels == predict_labels).to_list() + [case_th, ctrl_th, dist_th, fs_mode, k, kmer2cluster is not None]

    return results, subject_table.transpose()


@ray.remote(max_retries=0)
def remote_test_fold(
        airr_seq_df, train_labels, test_labels, dist_mat_dir, dist_th, case_th, ctrl_th,
        feature_selection_mode_values, k_values, kmer2cluster_values
):

    return test_fold(
        airr_seq_df, train_labels, test_labels, dist_mat_dir, dist_th, case_th, ctrl_th,
        feature_selection_mode_values, k_values, kmer2cluster_values,
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
    feature_selection_mode_values: list = ['naive'],
    k_values: list = [],
    kmer2cluster_values: list = []
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
    :param feature_selection_mode_values: list of method to use for the feature selection ("naive" or "similar")
    :param k_values: list of k values to use for the k-mers segmentation. relevant only when feature selection mode is similar
    :param kmer2cluster_values: list of k-mers to k-mers cluster id maps relevant only when feature selection mode is similar
    :return: (data frame with the folds classification results, data frame with the folds test samples)
    """
    airr_seq_df = ray.put(airr_seq_df)
    for i in range(len(kmer2cluster_values)):
        kmer2cluster_values[i] = ray.put(kmer2cluster_values[i])
    result_ids = []
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    for train_index, validation_index in rskf.split(labels, labels):

        validation_labels = labels[validation_index]
        train_labels = labels.drop(index=validation_labels.index)
        if train_only_labels is not None:
            train_labels = train_labels.append(train_only_labels)
        for case_th in case_th_values:
            for ctrl_th in ctrl_th_values:
                if ctrl_th >= case_th:
                    continue
                for dist_th in dist_th_values:
                    result_ids.append(
                        remote_test_fold.remote(
                            airr_seq_df, train_labels, validation_labels, dist_mat_dir, dist_th, case_th, ctrl_th,
                            feature_selection_mode_values, k_values, kmer2cluster_values
                        )
                    )

    results, subject_table = pd.DataFrame(), pd.DataFrame()
    for result_id in result_ids:
        results_itr, subject_table_itr = ray.get(result_id)
        results = results.append(results_itr, ignore_index=True)
        subject_table = subject_table.append(subject_table_itr, ignore_index=True)

    return results, subject_table


def save_results(
    results: pd.DataFrame, subject_table:pd.DataFrame, output_dir: str, base_name: str
):
    """
    group the classification result data frames by hyper-parameters and saves frames as different files
    :param results: the data frame with the classification metrics
    :param subject_table: the data frame with the folds samples split information
    :param output_dir: output directory to save the files
    :param base_name: prefix for the saved files
    :return:
    """
    hyper_parameters = results.head(0).drop(columns=['support', 'accuracy_score', 'recall_score', 'precision_score', 'f1-score']).columns.tolist()

    for params, frame in results.groupby(hyper_parameters):
        output_path = os.path.join(
            output_dir, '_'.join([base_name] + [f'{key}-{val}' for key, val in zip(hyper_parameters, params)]) + '_results.csv'
        )
        frame.drop(columns=hyper_parameters).to_csv(output_path, index=False)

    for params, frame in subject_table.groupby(hyper_parameters):
        output_path = os.path.join(
            output_dir, '_'.join([base_name] + [f'{key}-{val}' for key, val in zip(hyper_parameters, params)]) + '_subject_table.csv'
        )
        frame.drop(columns=hyper_parameters).to_csv(output_path, index=False)


def test_by_n_clusters(
    input_dir: str,
    output_dir: str,
    case_metadata: pd.DataFrame,
    ctrl_metadata: pd.DataFrame,
    labels: pd.Series,
    train_only_labels: pd.Series = None,
    n_splits: int = 10,
    n_repeats: int = 10,
    v_call_field: str = "v_call_original",
    group_mode: str = "family",
    max_seq_per_cluster_values: list = [1],
    n_clusters_values: list = [100],
    case_th_values: list = [1],
    ctrl_th_values: list = [0],
    dist_th_values: list = [0.2],
    feature_selection_mode_values: list = ['naive'],
    within_sample_clustering_values: list = ['complete_linkage_0.0'],
    k_values: list = [],
    kmer2cluster_values: list = []
):
    """
    :param input_dir:
    :param output_dir:
    :param case_metadata:
    :param ctrl_metadata:
    :param labels:
    :param train_only_labels: labels of repertoire samples that can only be used for training set and not for test
    :param n_splits:
    :param n_repeats:
    :param v_call_field:
    :param group_mode:
    :param max_seq_per_cluster_values:
    :param n_clusters_values:
    :param case_th_values:
    :param ctrl_th_values:
    :param dist_th_values:
    :param feature_selection_mode_values:
    :param within_sample_clustering_values:
    :param k_values:
    :param kmer2cluster_values:
    :return:
    """
    for within_sample_clustering in within_sample_clustering_values:
        for n_clusters in n_clusters_values:
            for max_seq_per_cluster in max_seq_per_cluster_values:
                base_name = f'top_{n_clusters}_{within_sample_clustering}_clusters_max_seq_per_cluster_{max_seq_per_cluster}'
                airr_seq_df_file_path = os.path.join(output_dir, 'sampled_datasets', base_name + '.tsv')
                if not os.path.isfile(airr_seq_df_file_path):
                    sample_by_n_clusters(
                        case_metadata.append(ctrl_metadata),
                        input_dir,
                        os.path.join(output_dir, 'sampled_datasets'),
                        [n_clusters],
                        [max_seq_per_cluster],
                        [f'subject_cluster_id_{within_sample_clustering}']
                    )
                airr_seq_df = load_sampled_airr_seq_df(airr_seq_df_file_path, v_call_field=v_call_field, group_mode=group_mode)
                save_distance_matrices(airr_seq_df, os.path.join(output_dir, "distance_matrices", base_name))
                results, subject_table = test_rep_classifier(
                    airr_seq_df,
                    labels,
                    os.path.join(output_dir, "distance_matrices", base_name),
                    train_only_labels,
                    n_splits,
                    n_repeats,
                    case_th_values,
                    ctrl_th_values,
                    dist_th_values,
                    feature_selection_mode_values,
                    k_values,
                    kmer2cluster_values
                )
                save_results(results, subject_table, os.path.join(output_dir, 'rep_clf_output'), base_name)


def test_by_n_sequences(
        case_metadata,
        ctrl_metadata,
        labels,
        input_dir,
        output_dir,
        n_repeats: int = 10,
        n_splits: int = 10,
        v_call_field: str = "v_call_original",
        group_mode: str = "family",
        n_sequences_values: list = [100],
        case_th_values: list = [1],
        ctrl_th_values: list = [0],
        dist_th_values: list = [0.2],
        feature_selection_mode_values: list = ['naive'],
        k_values: list = [],
        kmer2cluster_values: list = []
):

    for n_sequences in n_sequences_values:
        base_name = f'random_{n_sequences}_sequences_samples'
        airr_seq_df_file_path = os.path.join(output_dir, 'sampled_datasets', base_name + '.tsv')
        if not os.path.isfile(airr_seq_df_file_path):
            sample_by_n_sequences(
                case_metadata.append(ctrl_metadata),
                input_dir,
                output_dir,
                [n_sequences]
            )
        save_distance_matrices(airr_seq_df, os.path.join(output_dir, "distance_matrices", base_name))
        airr_seq_df = load_sampled_airr_seq_df(airr_seq_df_file_path, v_call_field=v_call_field, group_mode=group_mode)
        results, subject_table = test_rep_classifier(
            airr_seq_df,
            labels,
            os.path.join(output_dir, "distance_matrices", base_name),
            n_splits,
            n_repeats,
            case_th_values,
            ctrl_th_values,
            dist_th_values,
            feature_selection_mode_values,
            k_values,
            kmer2cluster_values
        )
        save_results(results, subject_table, os.path.join(output_dir, 'rep_clf_output'), base_name)


