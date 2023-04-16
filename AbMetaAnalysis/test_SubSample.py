from unittest import TestCase
import unittest
import tempfile
from SubSample import subsample_top_n_mutated_clusters, subsample_top_n_abundant_clusters, subsample
import pandas as pd
import os


class TestSubSample(unittest.TestCase):

    def setUp(self):
        self.airr_seq_df = pd.DataFrame(
            [
                ["AAAAAAAAA", 1, 0.2, 1, "AAAAAAAAA"],
                ["AAAAAAAAA", 1, 0.2, 1, "AAAAAAAAA"],
                ["AAAAAAAAB", 2, 0.3, 1, "AAAAAAAAB"],
                ["CCCCCCCCC", 1, 0.3, 2, "CCCCCCCCC"],
                ["CCCCCCCCA", 2, 0.31, 2, "CCCCCCCCA"],
                ["DDDDDDDDD", 1, 0.4, 3, "DDDDDDDDD"]
            ],
            columns=["junction_aa", "duplicate_count", "mu_freq", "cluster_id", "sequence"]
        )

    def test_subsample_top_n_mutated_clusters(self):
        sample = subsample_top_n_mutated_clusters(self.airr_seq_df, 2, 1, "cluster_id", 2)
        self.assertEqual(len(sample), 2)
        self.assertEqual(sum((sample.cluster_id == 1) & (sample.sequence == "AAAAAAAAB")), 1)
        self.assertEqual(sum((sample.cluster_id == 2) & (sample.sequence == "CCCCCCCCA")), 1)

        sample = subsample_top_n_mutated_clusters(self.airr_seq_df, 2, 1, "cluster_id", 1)
        self.assertEqual(len(sample), 2)
        self.assertEqual(sum((sample.cluster_id == 3) & (sample.sequence == "DDDDDDDDD")), 1)
        self.assertEqual(sum((sample.cluster_id == 2) & (sample.sequence == "CCCCCCCCA")), 1)

        sample = subsample_top_n_mutated_clusters(self.airr_seq_df, 2, 2, "cluster_id", 2)
        self.assertEqual(len(sample), 4)
        self.assertEqual(sum((sample.cluster_id == 1) & (sample.sequence == "AAAAAAAAB")), 1)
        self.assertEqual(sum((sample.cluster_id == 1) & (sample.sequence == "AAAAAAAAA")), 1)
        self.assertEqual(sum((sample.cluster_id == 2) & (sample.sequence == "CCCCCCCCA")), 1)
        self.assertEqual(sum((sample.cluster_id == 2) & (sample.sequence == "CCCCCCCCC")), 1)

        sample = subsample_top_n_mutated_clusters(self.airr_seq_df, 1, 2, "cluster_id", 1)
        self.assertEqual(len(sample), 1)
        self.assertEqual(sum((sample.cluster_id == 3) & (sample.sequence == "DDDDDDDDD")), 1)

    def test_subsample_top_n_abundant_clusters(self):
        sample = subsample_top_n_abundant_clusters(self.airr_seq_df, 2, 1, "cluster_id")
        self.assertEqual(len(sample), 2)
        self.assertEqual(sum((sample.cluster_id == 1) & (sample.sequence == "AAAAAAAAA")), 1)
        self.assertEqual(sum((sample.cluster_id == 2) & (sample.sequence == "CCCCCCCCA")), 1)

        sample = subsample_top_n_mutated_clusters(self.airr_seq_df, 2, 2, "cluster_id", 2)
        self.assertEqual(len(sample), 4)
        self.assertEqual(sum((sample.cluster_id == 1) & (sample.sequence == "AAAAAAAAB")), 1)
        self.assertEqual(sum((sample.cluster_id == 1) & (sample.sequence == "AAAAAAAAA")), 1)
        self.assertEqual(sum((sample.cluster_id == 2) & (sample.sequence == "CCCCCCCCA")), 1)
        self.assertEqual(sum((sample.cluster_id == 2) & (sample.sequence == "CCCCCCCCC")), 1)

    def test_subsample(self):

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.airr_seq_df.to_csv(os.path.join(tmp_dir, '1.tsv'), sep='\t', index=False)
            self.airr_seq_df.to_csv(os.path.join(tmp_dir, '2.tsv'), sep='\t', index=False)
            metadata = pd.DataFrame(
                [
                    ['test', '1', '1.tsv'],
                    ['test', '2', '2.tsv']
                ],
                columns=['study_id', 'subject_id', 'input_file']
            )
            subsample(
                metadata.set_index(['study_id', 'subject_id']),
                tmp_dir,
                tmp_dir,
                [
                    {
                        'subsample_method': 'top_n_mutated_clusters',
                        'n_clusters': 2,
                        'max_seq_per_cluster': 1,
                        'min_cluster_size': 2,
                        'cluster_id_col': 'cluster_id',
                        'force': True
                    },
                    {
                        'subsample_method': 'top_n_abundant_clusters',
                        'n_clusters': 2,
                        'max_seq_per_cluster': 1,
                        'cluster_id_col': 'cluster_id',
                        'force': True
                    },
                    {
                        'subsample_method': 'random_n_sequences',
                        'n_sequences': 1,
                        'cluster_id_col': 'cluster_id',
                        'force': True
                    },
                ]
            )

            self.assertTrue(os.path.isfile(
                os.path.join(tmp_dir, 'top_2_mutated_cluster_id_clusters_max_seq_per_cluster_1_min_cluster_size_2.tsv'))
            )
            df = pd.read_csv(
                os.path.join(tmp_dir, 'top_2_mutated_cluster_id_clusters_max_seq_per_cluster_1_min_cluster_size_2.tsv'), sep='\t',
                dtype={'subject_id': str, 'cluster_id': str, 'subject_id': str}
            )
            self.assertEqual(len(df), 4)
            self.assertTrue('study_id' in df.columns)
            self.assertEqual(sum((df.cluster_id == "1.0") & (df.sequence == "AAAAAAAAB") & (df.subject_id == "1")), 1)
            self.assertEqual(sum((df.cluster_id == "2.0") & (df.sequence == "CCCCCCCCA") & (df.subject_id == "1")), 1)
            self.assertEqual(sum((df.cluster_id == "1.0") & (df.sequence == "AAAAAAAAB") & (df.subject_id == "2")), 1)
            self.assertEqual(sum((df.cluster_id == "2.0") & (df.sequence == "CCCCCCCCA") & (df.subject_id == "2")), 1)

            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, 'top_2_cluster_id_clusters_max_seq_per_cluster_1.tsv')))
            df = pd.read_csv(
                os.path.join(tmp_dir, 'top_2_cluster_id_clusters_max_seq_per_cluster_1.tsv'), sep='\t',
                dtype={'subject_id': str, 'cluster_id': str, 'subject_id': str}
            )
            self.assertEqual(len(df), 4)
            self.assertTrue('study_id' in df.columns)
            self.assertEqual(sum((df.cluster_id == "1.0") & (df.sequence == "AAAAAAAAA") & (df.subject_id == "1")), 1)
            self.assertEqual(sum((df.cluster_id == "2.0") & (df.sequence == "CCCCCCCCA") & (df.subject_id == "1")), 1)
            self.assertEqual(sum((df.cluster_id == "1.0") & (df.sequence == "AAAAAAAAA") & (df.subject_id == "2")), 1)
            self.assertEqual(sum((df.cluster_id == "2.0") & (df.sequence == "CCCCCCCCA") & (df.subject_id == "2")), 1)

            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, 'random_1_sequences_samples.tsv')))
            df = pd.read_csv(
                os.path.join(tmp_dir, 'random_1_sequences_samples.tsv'), sep='\t',
                dtype={'subject_id': str, 'cluster_id': str, 'subject_id': str}
            )
            self.assertEqual(len(df), 2)
            self.assertEqual(sum(df.subject_id == "1"), 1)
            self.assertEqual(sum(df.subject_id == "2"), 1)


unittest.main()

