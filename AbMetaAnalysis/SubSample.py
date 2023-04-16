import pandas as pd
import os
from IPython.display import clear_output


def subsample_top_n_abundant_clusters(
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

    clusters_df.sort_values(['cluster_freq', 'clusters_freq_dup_cnt'], ascending=False, inplace=True)
    top_n_clusters = clusters_df.iloc[:min(n_clusters, len(clusters_df))].index
    # from each cluster take the max_seq_per_cluster sequences with the most frequent junction_aa
    airr_seq_df = airr_seq_df.loc[airr_seq_df[cluster_id_col].isin(top_n_clusters)].groupby(cluster_id_col).apply(
        lambda x: x.groupby('junction_aa').apply(
            lambda y: pd.concat(
                [y.iloc[0], pd.Series(len(y) / len(x), index=['junction_aa_freq_in_cluster'])])
        ).sort_values('junction_aa_freq_in_cluster', ascending=False).reset_index(
            drop=True
        ).iloc[:max_seq_per_cluster]
    ).reset_index(drop=True).copy(True)

    return airr_seq_df


def subsample_top_n_mutated_clusters(
    airr_seq_df, n_clusters, max_seq_per_cluster, cluster_id_col, min_cluster_size
):
    airr_seq_df = airr_seq_df.loc[airr_seq_df[cluster_id_col].notna()]
    clusters_freq = airr_seq_df[cluster_id_col].value_counts(ascending=False, normalize=round(min_cluster_size) != min_cluster_size)
    clusters_freq.name = 'cluster_freq'
    clusters_df = pd.DataFrame(clusters_freq)
    del clusters_freq
    mu_freq = airr_seq_df.groupby(cluster_id_col).apply(lambda x: x.mu_freq.mean())
    mu_freq.name = 'mu_freq'
    clusters_df.loc[mu_freq.index, 'mu_freq'] = mu_freq.to_list()
    del mu_freq
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

    clusters_df['min_cluster_size'] = clusters_df['cluster_freq'] >= min_cluster_size
    clusters_df.sort_values(['min_cluster_size', 'mu_freq', 'cluster_freq', 'clusters_freq_dup_cnt'], ascending=False, inplace=True)
    top_n_clusters = clusters_df.iloc[:min(n_clusters, len(clusters_df))].index
    # from each cluster take the max_seq_per_cluster sequences with the most frequent junction_aa
    airr_seq_df = airr_seq_df.loc[airr_seq_df[cluster_id_col].isin(top_n_clusters)].groupby(cluster_id_col).apply(
        lambda x: x.groupby('junction_aa').apply(
            lambda y: pd.concat([y.iloc[0], pd.Series(y.mu_freq.mean(), index=['junction_aa_mu_freq_in_cluster'])])
        ).sort_values('junction_aa_mu_freq_in_cluster', ascending=False).reset_index(
            drop=True
        ).iloc[:max_seq_per_cluster]
    ).reset_index(drop=True).copy(True)

    return airr_seq_df


def subsample_random_n_sequences(airr_seq_df, n_sequences):

    return airr_seq_df.sample(min(len(airr_seq_df), n_sequences), random_state=42).copy(True)


def subsample(
    metadata: pd.DataFrame,
    input_dir: str,
    output_dir: str,
    subsample_configs: list
):
    # to be more efficient we will go over the files only once and simultaneously do all subsampling
    output_files = pd.Series(None, index=range(len(subsample_configs)), dtype=object)
    # first aggregate all the columns from all files
    base_columns = set(["study_id", "subject_id"])
    for input_file in metadata.input_file:
        with open(os.path.join(input_dir, input_file), 'r') as f_in:
            for chunk_df in pd.read_csv(f_in, sep='\t', chunksize=1):
                base_columns = base_columns.union(chunk_df.columns)
                break
    for input_file_idx, ((study_id, subject_id), sample) in enumerate(metadata.iterrows()):
        single_sample_airr_seq_df = None
        for cfg_idx, cfg in enumerate(subsample_configs):

            if cfg['subsample_method'] == 'top_n_abundant_clusters':
                output_file_path = os.path.join(
                    output_dir,
                    f'top_{cfg["n_clusters"]}_{cfg["cluster_id_col"].replace("subject_cluster_id_", "")}_clusters_max_seq_per_cluster_'
                    f'{cfg["max_seq_per_cluster"]}.tsv'
                )
                columns = base_columns.union(['junction_aa_freq_in_cluster'])
            elif cfg['subsample_method'] == 'top_n_mutated_clusters':
                output_file_path = os.path.join(
                    output_dir,
                    f'top_{cfg["n_clusters"]}_mutated_'
                    f'{cfg["cluster_id_col"].replace("subject_cluster_id_", "")}_clusters_max_seq_per_cluster_'
                    f'{cfg["max_seq_per_cluster"]}_min_cluster_size_{cfg["min_cluster_size"]}.tsv'
                )
                columns = base_columns.union(['junction_aa_mu_freq_in_cluster'])
            elif cfg['subsample_method'] == 'random_n_sequences':
                output_file_path = os.path.join(
                    output_dir,
                    f'random_{cfg["n_sequences"]}_sequences_samples.tsv'
                )
            else:
                assert False, f'Unknown subsampling method {cfg.subsample_method}'

            if not cfg["force"] and os.path.isfile(output_file_path):
                if input_file_idx == 0:
                    print(f'file {output_file_path} already exists - skipping sampling')
                continue
            if single_sample_airr_seq_df is None:
                print(f'Sampling file {input_file_idx + 1}: {sample.input_file}')
                single_sample_airr_seq_df = pd.read_csv(
                    os.path.join(input_dir, sample.input_file), sep='\t', dtype={'sequence_id': 'str'}
                )
                single_sample_airr_seq_df = single_sample_airr_seq_df.loc[single_sample_airr_seq_df.junction_aa.notna()]
                single_sample_airr_seq_df = single_sample_airr_seq_df.loc[single_sample_airr_seq_df.junction_aa.str.find('*') == -1]
                single_sample_airr_seq_df = single_sample_airr_seq_df.loc[single_sample_airr_seq_df.junction_aa.str.len() >= 9]

            if cfg["subsample_method"] == 'top_n_abundant_clusters':
                sampled_df = subsample_top_n_abundant_clusters(
                    single_sample_airr_seq_df, cfg["n_clusters"], cfg["max_seq_per_cluster"], cfg["cluster_id_col"]
                )
            elif cfg["subsample_method"] == 'top_n_mutated_clusters':
                sampled_df = subsample_top_n_mutated_clusters(
                    single_sample_airr_seq_df, cfg["n_clusters"], cfg["max_seq_per_cluster"], cfg["cluster_id_col"], cfg["min_cluster_size"]
                )
            elif cfg["subsample_method"] == 'random_n_sequences':
                sampled_df = subsample_random_n_sequences(
                    single_sample_airr_seq_df, cfg["n_sequences"]
                )
            else:
                assert False, f'Unknown subsampling method {cfg.subsample_method}'

            sampled_df = pd.concat(
                [sampled_df, pd.DataFrame(columns=list(set(columns).difference(single_sample_airr_seq_df.columns)))]
            )
            sampled_df = sampled_df[list(columns)]
            sampled_df['subject_id'] = str(subject_id)
            sampled_df['study_id'] = str(study_id)
            f_out = output_files[cfg_idx]
            if pd.isna(f_out):
                f_out = open(output_file_path, 'w')
                f_out.write('\t'.join(list(columns)) + '\n')
                output_files[cfg_idx] = f_out
            sampled_df.to_csv(f_out, sep='\t', index=False, header=False)
            clear_output(wait=True)
        del single_sample_airr_seq_df

    output_files = output_files.loc[output_files.notna()]
    for _, f_out in output_files.items():
        f_out.close()


def sample_by_n_clusters(
    metadata: pd.DataFrame,
    input_dir: str,
    output_dir: str,
    n_clusters_values: list = [100],
    max_seq_per_cluster_values: list = [1],
    cluster_id_col_values: list = ['subject_cluster_id_complete_linkage_dist_0.0'],
    force: bool = False
):
    # backwards compatibility implementation
    subsample_configs = []
    for (n_clusters, max_seq_per_cluster, cluster_id_col) in [
        (n_clusters, max_seq_per_cluster, cluster_id_col) for
        n_clusters in n_clusters_values for max_seq_per_cluster in max_seq_per_cluster_values for cluster_id_col in cluster_id_col_values
    ]:
        subsample_configs.append({
            'subsample_method': 'top_n_abundant_clusters',
            'n_clusters': n_clusters,
            'max_seq_per_cluster': max_seq_per_cluster,
            'cluster_id_col': cluster_id_col,
            'force': force
        })

    subsample(metadata, input_dir, output_dir, subsample_configs)


def sample_by_n_sequences(
    metadata: pd.DataFrame,
    input_dir: str,
    output_dir: str,
    n_sequences_values: list = [100],
    force: bool = False
):
    # backwards compatibility implementation
    for n_sequences in n_sequences_values:
        subsample_configs.append({
            'n_sequences': n_sequences,
            'force': force
        })
    subsample(metadata, input_dir, output_dir, subsample_configs)