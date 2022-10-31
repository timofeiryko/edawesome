"""Helper functions for EDA which are not just sns or pd wrappers."""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway

import matplotlib.pyplot as plt
import seaborn as sns

import pyspark

from .str_help import snake_to_title

def _print_statistical_test(test_results):
    """Print results of statistical test"""
    print(f'p-value: {test_results.pvalue}')
    if test_results.pvalue < 0.05:
        print('Reject null hypothesis!')
    else:
        print('Accept null hypothesis!')

def compare_distributions(df, num_col_name: str, cat_col_name: str, treat_nans: str = 'drop'):
    """Compare distributions of numeric feauture by categorical feature"""

    # TODO some decorator for checking state
    # TODO some decorator for checking possible values of the arguments and raising errors

    if treat_nans not in ['drop', 'mean', 'median']:
        raise ValueError(f'Unknown treatment method {treat_nans}! Use "drop", "mean" or "median"')

    sns.displot(
        data=df,
        x=num_col_name, hue=cat_col_name,
        kind='kde', fill=True
    )
    
    plt.title(f'{snake_to_title(num_col_name)} distribution by {snake_to_title(cat_col_name)}')

    sns.despine()
    plt.show()

    sns.boxplot(
        data=df,
        x=cat_col_name, y=num_col_name
    )

    # add mean labels
    means = df.groupby(cat_col_name)[num_col_name].mean()
    for i, mean in enumerate(means):
        plt.text(i, mean, round(mean, 2), horizontalalignment='center', size='large', color='w', weight='semibold')

    sns.despine()
    plt.show()

    if treat_nans == 'drop':
        df = df.dropna()
    elif treat_nans == 'mean':
        df = df.fillna(df.mean())
    elif treat_nans == 'median':
        df = df.fillna(df.median())

    num_col = df[num_col_name]
    cat_col = df[cat_col_name]


    if df[cat_col_name].nunique() > 2:

        raise NotImplementedError('Comparison of more than 2 categories is not implemented yet!')

        # ANOVA, comparing num_col distribution by cat_col
        print('Compare means with ANOVA:')
        test_results = f_oneway(*[num_col[cat_col == cat] for cat in cat_col.unique()])
        _print_statistical_test(test_results)
        
    elif df[cat_col_name].nunique() == 2:
        print('Compare means with t-test:')

        first = num_col[cat_col == cat_col.unique()[0]]
        second = num_col[cat_col == cat_col.unique()[1]]
        _print_statistical_test(ttest_ind(first, second))
    else:
        raise ValueError('Categorical feature has only one category!')