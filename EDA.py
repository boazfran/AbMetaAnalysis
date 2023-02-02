"""
Utility functions for dataset Elementary Data Analysis
"""

# Info
__author__ = 'Boaz Frankel'

# Imports
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import multinomial_proportions_confint
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests


def mannwhitneyu_test(df, col, confidence_th, alternative='greater', adjustment_method='holm-sidak', y='Freq'):
    df = df.reset_index().set_index(['label'])
    p_values = df.groupby(col).apply(
        lambda x: getattr(
            mannwhitneyu(
                x.loc['CASE'][y],
                x.loc['CTRL'][y],
                alternative=alternative
            ), 'pvalue'
        )
    )
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values.to_list(), confidence_th, method=adjustment_method)
    return pd.DataFrame(
        [p_values.to_list(), pvals_corrected.tolist(), reject.tolist()],
        index=['pvalue', 'pvalue_corrected', 'reject'],
        columns=p_values.index
    ).transpose().sort_values('pvalue')