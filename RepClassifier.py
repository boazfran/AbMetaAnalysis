# Info
__author__ = 'Boaz Frankel'

# Imports
import pandas
import pandas as pd
import os
import numpy as np


# AbMetaAnalysis imports
from AbMetaAnalysis.Clustering import add_cluster_id, match_cluster_id, save_distance_matrices
from AbMetaAnalysis.Utilities import build_feature_table
from AbMetaAnalysis.EDA import mannwhitneyu_test


def select_features(
    airr_seq_df: pd.DataFrame,
    cluster_assignment: pd.Series,
    feature_table: pd.DataFrame,
    train_labels: pd.Series,
    case_th: int,
    ctrl_th: int,
    mode: str ='naive',
    k: int = 5,
    kmer2cluster: dict = None
) -> np.ndarray:
    if mode == 'naive':
        features = feature_table.columns[
            (
                    (feature_table.loc[train_labels.index[train_labels]].sum() > case_th) &
                    (feature_table.loc[train_labels.index[~train_labels]].sum() <= ctrl_th)
            )
        ]
    else:
        assert False, 'unsupported feature selection mode - supported modes are naive or similar'

    return features


class RepClassifier:

    def __init__(
        self,
        train_airr_seq_df: pd.DataFrame,
        train_cluster_assignment: pd.Series,
        dist_th: float = 0.2,
        case_th: float = 1.0,
        ctrl_th: float = 0.0,
        feature_selection_mode: str = "naive",
        k: int = 5,
        kmer2cluster: dict = None
    ):
        self.dist_th = dist_th
        self.case_th = case_th
        self.ctrl_th = ctrl_th
        self.feature_selection_mode = feature_selection_mode
        self.train_airr_seq_df = train_airr_seq_df
        self.train_cluster_assignment = train_cluster_assignment
        self.k = k
        self.kmer2cluster = kmer2cluster
        self.selected_features = None
        self.train_airr_seq_df = None

    def fit(self, feature_table: pd.DataFrame, labels):
        self.selected_features = select_features(
            self.train_airr_seq_df, self.train_cluster_assignment, feature_table, labels,
            self.case_th, self.ctrl_th, self.feature_selection_mode, self.k, self.kmer2cluster
        )
        return self

    @staticmethod
    def predict(test_feature_table: pd.DataFrame) -> pd.Series:
        return test_feature_table.sum(axis=1) > 0




