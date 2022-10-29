import pandas as pd

from IPython.display import display


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