<<<<<<< HEAD
"""
Shortcuts and helper functions for seaborn.
"""

=======
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
from turtle import color
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from typing import Optional, Tuple

from .str_help import snake_to_title

def kde_boxen_qq(
    dataframe: pd.DataFrame, col_name: str,
    title: Optional[str] = None,
    figsize: Optional[Tuple[str, str]] = None
    # It's not good practice to "inherit" figsize without actual inheritance or composition
) -> None:

    if title is None:
        title = snake_to_title(col_name)

    fig, axes = plt.subplot_mosaic([['up', 'right'],['down', 'right']],
                                    constrained_layout=True, figsize=(10,6),
                                    gridspec_kw={
                                        'height_ratios': (0.3, 0.7),
                                        'width_ratios': (0.6, 0.4)
                                    })
    
    fig.suptitle(title, fontsize='xx-large')

    sns.histplot(data=dataframe, x=col_name, ax=axes['down'], kde=True)

    sns.boxenplot(data=dataframe, x=col_name, ax=axes['up'])
    axes['up'].set_xlabel('')
    axes['up'].set_xticklabels('')

    sm.qqplot(dataframe[col_name], fit=True, line='45', alpha=0.2, ax=axes['right'])
    axes['right'].set_title('QQ plot')

    if figsize:
        x, y = figsize
        fig.set_figwidth(x)
        fig.set_figheight(y)

    sns.despine()
    plt.show()

<<<<<<< HEAD


#TODO: typing
def corr_heatmap(df: pd.DataFrame, variables=None, **kwargs):
=======
#TODO: typing
def corr_heatmap(df: pd.DataFrame, variables=None):
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
    
    if variables is None:
        variables = df.columns

<<<<<<< HEAD
    corr_map = df[variables].corr(**kwargs)
=======
    corr_map = df[variables].corr(numeric_only=True)
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
    # Zeros instead of diagonal
    corr_map.values[tuple([np.arange(corr_map.shape[0])]) * 2] = np.nan

    sns.heatmap(corr_map, cmap='coolwarm', annot=True, vmin=-1, vmax=1)
    plt.show()