"""
Utility functions for clustering
"""

# Info
__author__ = 'Boaz Frankel'

# Imports
import pandas
import pandas as pd
import os
import numpy as np
from changeo.Gene import getFamily, getGene, getAllele
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import squareform

# MetaAnalysis imports
import sys
sys.path.append('/work/boazfr/dev/packages')
from MetaAnalysis.Utilities import get_imgt_allele


def sequence_series_to_numeric_array(sequence_series: pd.Series) -> np.ndarray:
    """
    translates a series of equal length amino acid sequences to 2d numpy integer array
    :param sequence_series: a series of equal length amino acid sequences
    :return: a 2d numpy integer array
    """
    assert len(sequence_series.str.len().value_counts()) == 1, "all strings must have the same length"
    sequence_series = np.array(sequence_series.apply(list).to_list())
    arr = np.zeros(sequence_series.shape, dtype=int)
    for val, key in enumerate(['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']):
        arr[sequence_series == key] = val
    return arr


def save_distance_matrices(airr_seq_df: pd.DataFrame, dist_mat_dir: str, force: bool = False):
    """
    Compute and save pairwise normalized hamming distance matrices of junction_aa groups, grouped by the v and j
    genes or families
    :param airr_seq_df: airr-seq data frame to compute the pairwise distance matrices
    :param dist_mat_dir: directory to save the distance matrices
    :param force: recompute and overwrite even if folder already exists
    :return:
    """
    n = 0
    if not os.path.isdir(dist_mat_dir):
        os.makedirs(os.path.join(dist_mat_dir))
    for (v_group, j_group, junction_aa_length), frame in airr_seq_df.groupby(['v_group', 'j_group', 'junction_aa_length']):
        n += len(frame)
        save_dir = os.path.join(dist_mat_dir, v_group, j_group, junction_aa_length)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if os.path.isfile(os.path.join(save_dir, 'pdist.tsv.gz')) and not force:
            continue
        # square form matrix of the pairwise hamming distance - possible as we grouped by junction_aa_length
        pdist_df = pd.DataFrame(
            squareform(pdist(sequence_series_to_numeric_array(frame.junction_aa), 'hamming')),
            index=frame.index,
            columns=frame.index
        )
        # give penlty to sequences from same studise and/or same subject - the purpose is to priortize the creation of public clusters
        # the penalty is set to be small enough so that it will only be a brake even for between sequence pairs with the same hamming distance.
        pdist_df += (pdist_df.apply(lambda x: x.name.split(';')[0] == pd.Series(x.index, index=x.index).apply(
            lambda y: y.split(';')[0])) * 0.0001)
        pdist_df += (pdist_df.apply(lambda x: x.name[:x.name.rfind(';') + 1] == pd.Series(x.index, index=x.index).apply(
            lambda y: y[:y.rfind(';') + 1])) * 0.0001)
        np.fill_diagonal(pdist_df.values, 0)
        pdist_df.reset_index().to_csv(os.path.join(save_dir, 'pdist.tsv.gz'), sep='\t', index=False, compression='gzip')


def add_cluster_id(
        airr_seq_df: pandas.DataFrame, dist_mat_dir: str, dist_th: float = 0.2
) -> pandas.DataFrame:
    """
    cluster the sequences of airr-seq dataframe, first the sequences are grouped by their v gene, j gene and junction
    length and then clustered by their hamming distance using a complete linkage hierarchical clustering with cutoff
    :param airr_seq_df: airr-seq dataframe to cluster
    :param dist_mat_dir: directory were pre-saved distance matrix are saved
    :param dist_th: normalized hamming distance cutoff of the complete linkage hierarchical clustering
    :return: a series with the cluster assignment
    """
    res = pd.Series(np.nan, index=airr_seq_df.index, name='cluster_id')
    max_cluster_id = 0
    for (v_group, j_group, junction_aa_length), frame in airr_seq_df.groupby(['v_group', 'j_group', 'junction_aa_length']):
        dist_map_folder = os.path.join(dist_mat_dir, v_group, j_group, junction_aa_length)
        dist_df = pd.read_csv(
            os.path.join(dist_map_folder, 'pdist.tsv.gz'), sep='\t'
        ).set_index(['id']).loc[frame.index, frame.index]
        if dist_df.shape[0] == 1:
            res.loc[frame.index] = max_cluster_id
        else:
            # We add a small extra to the distance threshold because when computing the distances we add penalty to
            # sequences from same study and/or same subject. The addition is small enough to only be relevant as
            # break even rule for sequences pairs with same hamming distance
            res.loc[frame.index] = max_cluster_id + AgglomerativeClustering(
                affinity='precomputed', linkage='complete', distance_threshold=dist_th + 0.0002, n_clusters=None
            ).fit_predict(dist_df)
        max_cluster_id = res.max() + 1
    res = res.astype(int)

    return airr_seq_df


# In[6]:


def match_cluster_id(
        clustered_airr_seq_df: pandas.DataFrame,
        cluster_assignment: pd.Series,
        to_match_airr_seq_df: pandas.DataFrame,
        dist_mat_dir: str,
        dist_th: float = 0.2
) -> pandas.DataFrame:
    """
    match sequences from not clustered df to clusters in a clustered df. note that a sequence can match more than one
    cluster. the matches will be added in string frame where each matched cluster id will be surrounded by ';' chars
    :param clustered_airr_seq_df: airr-seq data frame with cluster_id column
    :param cluster_assignment:
    :param to_match_airr_seq_df: airr-seq data frame to match to the clusters in the clustered_df
    :param dist_mat_dir: directory were pre-saved distance matrix are saved
    :param dist_th: normalized hamming distance cutoff of the complete linkage hierarchical clustering
    :return: series with the matched clusters strings, each matched cluster_id is wrapped with ';' chars
    """

    res = pd.Series(';', index=to_match_airr_seq_df.index, name='matched_clusters')

    for (v_group, j_group, junction_aa_length), to_match_frame in to_match_airr_seq_df.groupby(
            ['v_group', 'j_group', 'junction_aa_length']
    ):
        clustered_frame = cluster_assignment[
            (clustered_airr_seq_df.v_group == v_group) & (clustered_airr_seq_df.j_group == j_group) & (clustered_airr_seq_df.junction_aa_length == junction_aa_length)
        ]
        if len(clustered_frame) == 0:
            continue
        dist_df = pd.read_csv(
            os.path.join(dist_mat_dir, v_group, j_group, junction_aa_length, 'pdist.tsv.gz'),
            sep='\t'
        ).set_index(['id']).loc[clustered_frame.index, to_match_frame.index]
        for cluster_id, cluster_id_frame in clustered_frame.groupby(clustered_frame):
            matched_sequences = to_match_frame.index[
                dist_df.loc[cluster_id_frame.index].apply(lambda x: x > dist_th + 0.0002).sum() == 0
            ]
            res.loc[matched_sequences] = res.loc[matched_sequences] + str(cluster_id) + ';'

    return res

