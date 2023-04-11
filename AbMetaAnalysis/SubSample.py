import pandas as pd
import os
from IPython.display import clear_output


def subsample_n_clusters(
    airr_seq_df, n_clusters, max_seq_per_cluster, cluster_id_col
):
    airr_seq_df = airr_seq_df.loc[airr_seq_df[cluster_id_col].notna()]
    clusters_freq = airr_seq_df[cluster_id_col].value_counts(ascending=False)
    clusters_freq.name = 'cluster_freq'
    clusters_df = pd.DataFrame(clusters_freq)
    del clusters_freq
    clusters_freq_dup_cnt = airr_seq_df.groupby(cluster_id_col).apply(
        lambda x: x.duplicate_count.astype(int).sum()
    )
    clusters_df.loc[
        clusters_freq_dup_cnt.index, 'clusters_freq_dup_cnt'] = clusters_freq_dup_cnt.to_list()
    del clusters_freq_dup_cnt
    clusters_uniq_freq = airr_seq_df.groupby(cluster_id_col).apply(
        lambda x: len(x.sequence.unique())
    )
    clusters_df.loc[clusters_uniq_freq.index, 'clusters_uniq_freq'] = clusters_uniq_freq.to_list()
    del clusters_uniq_freq

    clusters_df.sort_values(['cluster_freq', 'clusters_freq_dup_cnt'], ascending=False)
    top_n_clusters = clusters_df.iloc[:min(n_clusters, len(clusters_df))].index
    # from each cluster take the max_seq_per_cluster sequences with the most frequent junction_aa
    airr_seq_df = airr_seq_df.loc[airr_seq_df[cluster_id_col].isin(top_n_clusters)].groupby(cluster_id_col).apply(
        lambda x: x.groupby('junction_aa').apply(
            lambda y: y.iloc[0].append(
                pd.Series(len(y) / len(x), index=['junction_aa_freq_in_cluster']))
        ).sort_values('junction_aa_freq_in_cluster', ascending=False).reset_index(
            drop=True
        ).iloc[:min(max_seq_per_cluster, len(clusters_df))]
    ).reset_index(drop=True).copy(True)

    return airr_seq_df


def sample_by_n_clusters(
    metadata,
    input_dir: str,
    output_dir: str,
    n_clusters_values: list = [100],
    max_seq_per_cluster_values: list = [1],
    cluster_id_col_values: list = ['subject_cluster_id_complete_linkage_dist_0.0'],
    force: bool = False
):
    # to be more efficient we will go over the files only once and simultaneously do all subsampling
    samples = pd.Series(
        dtype=object,
        index=[(n_clusters, max_seq_per_cluster, cluster_id_col) for n_clusters in n_clusters_values for
               max_seq_per_cluster in max_seq_per_cluster_values for cluster_id_col in cluster_id_col_values]
    )
    for i, ((study_id, subject_id), sample) in enumerate(metadata.iterrows()):
        single_sample_airr_seq_df = None
        for (n_clusters, max_seq_per_cluster, cluster_id_col) in samples.index:
            multi_sample_airr_seq_df_file_path = os.path.join(
                output_dir,
                f'top_{n_clusters}_{cluster_id_col.replace("subject_cluster_id_", "")}_clusters_max_seq_per_cluster_{max_seq_per_cluster}.tsv'
            )
            if not force and os.path.isfile(multi_sample_airr_seq_df_file_path):
                if i == 0:
                    print(f'file {multi_sample_airr_seq_df_file_path} already exists - skipping sampling')
                continue
            if single_sample_airr_seq_df is None:
                print(f'Sampling file {i + 1}: {sample.input_file}')
                single_sample_airr_seq_df = pd.read_csv(
                    os.path.join(input_dir, sample.input_file), sep='\t', dtype={'sequence_id': 'str'}
                )
                single_sample_airr_seq_df = single_sample_airr_seq_df.loc[single_sample_airr_seq_df.junction_aa.notna()]
                single_sample_airr_seq_df = single_sample_airr_seq_df.loc[single_sample_airr_seq_df.junction_aa.str.find('*') == -1]
                single_sample_airr_seq_df = single_sample_airr_seq_df.loc[single_sample_airr_seq_df.junction_aa.str.len() >= 9]
            if type(samples[(n_clusters, max_seq_per_cluster, cluster_id_col)]) == pd.core.frame.DataFrame:
                multi_sample_airr_seq_df = samples[(n_clusters, max_seq_per_cluster, cluster_id_col)]
            else:
                multi_sample_airr_seq_df = pd.DataFrame()
            sampled_df = subsample_n_clusters(single_sample_airr_seq_df, n_clusters, max_seq_per_cluster, cluster_id_col)
            sampled_df['subject_id'] = str(subject_id)
            sampled_df['study_id'] = str(study_id)
            samples[(n_clusters, max_seq_per_cluster, cluster_id_col)] = pd.concat(
                [multi_sample_airr_seq_df, sampled_df]
            )
            clear_output(wait=True)
        del single_sample_airr_seq_df

    samples = samples.loc[samples.notna()]
    for (n_clusters, max_seq_per_cluster, cluster_id_col), multi_sample_airr_seq_df in samples.items():
        multi_sample_airr_seq_df_file_path = os.path.join(
            output_dir,
            f'top_{n_clusters}_{cluster_id_col.replace("subject_cluster_id_", "")}_clusters_max_seq_per_cluster_{max_seq_per_cluster}.tsv'
        )
        multi_sample_airr_seq_df.to_csv(multi_sample_airr_seq_df_file_path, sep='\t', index=False)


def sample_by_n_sequences(
    metadata: pd.DataFrame,
    input_dir: str,
    output_dir: str,
    n_sequences_values:list = [100],
    force: bool = False
):
    samples = pd.Series(
        dtype=object,
        index=n_sequences_values
    )
    for i, ((study_id, subject_id), sample) in enumerate(metadata.iterrows()):
        single_sample_airr_seq_df = None
        for n_sequences in samples.index:
            multi_sample_airr_seq_df_file_path = os.path.join(
                output_dir, f'random_{n_sequences}_sequences_samples.tsv'
            )
            if not force and os.path.isfile(multi_sample_airr_seq_df_file_path):
                if i == 0:
                    print(f'file {multi_sample_airr_seq_df_file_path} already exists - skipping sampling')
                continue
            elif single_sample_airr_seq_df is None:
                print(f'Sampling file {i + 1}: {sample.input_file}')
                single_sample_airr_seq_df = pd.read_csv(os.path.join(input_dir, sample.input_file), sep='\t', dtype={'sequence_id': 'str'})
                single_sample_airr_seq_df = single_sample_airr_seq_df.loc[single_sample_airr_seq_df.junction_aa.notna()]
                single_sample_airr_seq_df = single_sample_airr_seq_df.loc[single_sample_airr_seq_df.junction_aa.str.len() >= 9]
                if type(samples[n_sequences]) == pd.core.frame.DataFrame:
                    multi_sample_airr_seq_df = samples[n_sequences]
                else:
                    multi_sample_airr_seq_df = pd.DataFrame()
                sampled_df = single_sample_airr_seq_df.sample(min(len(single_sample_airr_seq_df), n_sequences), random_state=42).copy(True)
                sampled_df['subject_id'] = str(subject_id)
                sampled_df['study_id'] = str(study_id)
                samples[n_sequences] = pd.concat([multi_sample_airr_seq_df, sampled_df])
                clear_output(wait=True)
        del single_sample_airr_seq_df
    samples = samples.loc[samples.notna()]
    for n_sequences, multi_sample_airr_seq_df in samples.items():
        multi_sample_airr_seq_df_file_path = os.path.join(
            output_dir, f'random_{n_sequences}_sequences_samples.tsv'
        )
        multi_sample_airr_seq_df.to_csv(multi_sample_airr_seq_df_file_path, sep='\t', index=False)
