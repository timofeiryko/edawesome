"""Shortcuts and helper functions for pandas."""

import pandas as pd

from IPython.display import display, Markdown


def explore_df(dataframe: pd.DataFrame) -> None:
    display(pd.concat([dataframe.head(), dataframe.tail()]))
    display(dataframe.info())

def full_display_rows(series: pd.Series, n: int = 5) -> None:
    
    count = 0
    for id, row in series.iteritems():
        count += 1
        if count > n:
            break
        print(f'{id}\t{row}')
    # Can I do it more efficiently?

def display_df(name: str, df: pd.DataFrame) -> None:
    """Display dataframe with shape."""
    display(Markdown(f'### {name}'))
    display(df.head())  
    display(df.tail())
    display(Markdown(f'**{df.shape[0]:,} rows × {df.shape[1]} columns**'))