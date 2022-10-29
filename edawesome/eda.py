from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional, List

from glob import glob
import os
import sqlite3
import zipfile
import tarfile

from kaggle.api.kaggle_api_extended import KaggleApi

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

from scipy.stats import ttest_ind

from transitions import Machine

from str_help import generate_attr, snake_to_title
from sns_help import kde_boxen_qq

# TODO: add dimensions concept
# TODO: configure logging, not IPython display

CATEGORY_MAX = 5

def basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

@dataclass
class EDA:
    """Main class for simple and robust EDA with pandas and seaborn."""

    data_dir_path: str
    kaggle_datasets: Optional[List[str]] = None
    # TODO: dictionaty with csv files as keys and separator as values

    state: Machine = field(init=False)
    # TODO implement .next() functionality, prohibit calling functions in wrong order

    silent: bool = False
    _concat: bool = False
    _use_polars: bool = False
    _lazy_sqlite: bool = False
    _load_limit: Optional[int] = None

    @property
    def data_list(self) -> List[pd.DataFrame]:
        return list(self._dataframes_dict.keys())

    def __post_init__(self) -> None:

        if self._concat or self._use_polars:
            raise NotImplementedError

        states = [
            'init',
            'loaded',
            'categorized',
            'cleaned'
        ]
        self.state = Machine(states=states, initial='init')
        self.state.add_ordered_transitions(loop=False)

        max_rows = pd.get_option('display.max_rows')
        pd.set_option('display.max_rows', 10)

    @staticmethod
    def prettify_col_names(df: pd.DataFrame) -> None:

        # detect auto index
        first_col = df.columns[0]
        if first_col == 'Unnamed: 0':
            df.rename(columns={first_col: 'index'}, inplace=True)
            first_col = 'index'

        if (df[first_col] == df.index).all():
            df.set_index(first_col, inplace=True)

        # columns to snake case
        df.columns = df.columns.to_series().apply(generate_attr)

    def _add_df(self, name: str, df: pd.DataFrame):

        if name in self._dataframes_dict.keys():
            raise ValueError(f'Dataframe with name {name} already exists. All filenames must be unique!')

        self.prettify_col_names(df)
        self._dataframes_dict[name] = df

        if not self.silent:
            display(Markdown(f'### {name}'))
            display(df)
            display(df.info())


    def _load_sqlite(self) -> None:

        sqlite_extensions = ('.db', '.sqlite', '.sqlite3')
        sqlite_files = []
        for ext in sqlite_extensions:
            # recursive search for sqlite files
            os.chdir(self.data_dir_path)
            sqlite_files += glob(f'*{ext}', recursive=True)
        
        for file in sqlite_files:

            connection = sqlite3.connect(file)
            
            tables_query = '''
            --sql
            SELECT name FROM sqlite_schema
            WHERE type ='table' AND name NOT LIKE 'sqlite_%';
            '''

            tables = pd.read_sql(tables_query, connection)
            
            for table in tables.name:
                query = f'''
                --sql
                SELECT * FROM {table};
                '''
                if self._load_limit:
                    query.replace(';', f'\nLIMIT {self._load_limit};')
                df = pd.read_sql(query, connection)
                self._add_df(table, df)


    def _load_csv(self) -> None:
        
        os.chdir(self.data_dir_path)

        tsv_files = glob(f'*.tsv', recursive=True)
        for file in tsv_files:

            if self._load_limit:
                df = pd.read_csv(file, sep='\t', nrows=self._load_limit)
            else:
                df = pd.read_csv(file, sep='\t')
            self._add_df(generate_attr(basename_no_ext(file)), df)

        csv_files = glob(f'*.csv', recursive=True)
        for file in csv_files:
            df = pd.read_csv(file, sep=None, engine='python')
            self._add_df(generate_attr(basename_no_ext(file)), df)

        # TODO: automatic separator detection, files without extension
        # TODO: automatic header and index detection

    def _download_kaggle(self) -> None:

        api = KaggleApi()
        api.authenticate()
        os.chdir(self.data_dir_path)

        if not self.kaggle_datasets:
            return ValueError('No kaggle datasets provided')

        for dataset in self.kaggle_datasets:
            api.dataset_download_files(dataset, path=self.data_dir_path)

            zip_files = glob(f'*.zip', recursive=True)
            for file in zip_files:
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir_path)
                os.remove(file)

            tar_files = glob(f'*.tar.gz', recursive=True)
            for file in tar_files:
                with tarfile.open(file, 'r:gz') as tar_ref:
                    tar_ref.extractall(self.data_dir_path)
                os.remove(file)

    def load_data(self) -> None:

        if not self.silent:
            display(Markdown('# Loading data'))
        
        self._dataframes_dict = {}
        self._download_kaggle()
        self._load_sqlite()
        self._load_csv()

        Data = namedtuple('Data', self._dataframes_dict)
        self.data = Data(**self._dataframes_dict)

    def clean_check(self, subset: Optional[List[str]] = None) -> None:
        
        display(Markdown('# Check for bad values'))

        for name, df in self._dataframes_dict.items():
            
            display(Markdown(f'### Nulls in {name}'))
            null_nums = df.isnull().sum()
            if null_nums.any():
                # show rows with some nulls
                display(df[df.isnull().any(axis='rows')])
            else:
                print('No nulls found...')

            display(Markdown(f'### Duplicates in {name}'))
            duplicates_nums = df.duplicated().sum()
            if duplicates_nums.any():
                # show rows with some duplicates
                display(df[~df.duplicated(keep=False)])
            else:
                print('No duplicates found...')

    def clean(self):
        display(Markdown('# Cleaning data'))

        for name, df in self._dataframes_dict.items():
            
            display(Markdown(f'### Drop nulls in {name}'))
            null_nums = df.isnull().sum()
            if null_nums.any():
                df.dropna(inplace=True, axis='rows')
                print(f'Dropped {null_nums.sum()} rows with nulls')
                print(f'No subset is used by default!')
            else:
                print('No nulls found...')

            display(Markdown(f'### Drop duplicates in {name}'))
            duplicates_nums = df.duplicated().sum()
            if duplicates_nums.any():
                df.drop_duplicates(inplace=True)
                print(f'Dropped {duplicates_nums.sum()} duplicates')
                print(f'No subset is used by default!')
            else:
                print('No duplicates found...')

    def categorize(self, subset: Optional[List[str]] = None) -> None:
        
        display(Markdown('# Identifiying categorical features'))

        for name, df in self._dataframes_dict.items():
                
                display(Markdown(f'### Categories in {name}'))

                already_cat_features = list(df.select_dtypes(include='category').columns)
                cat_features = [cat_name for cat_name in df.columns if df[cat_name].nunique() <= CATEGORY_MAX]
                new_cat_features = list(set(cat_features) - set(already_cat_features))

                if already_cat_features:
                    print(f'Already categorical:')
                    display(already_cat_features)

                if new_cat_features:
                    df[cat_features] = df[cat_features].astype('category')
                    print('Converted to categorical:')
                    display(new_cat_features)
                else:
                    print('No new categorical features found...')

                not_treated = list(df.select_dtypes(include='object').columns)
                if not_treated:
                    print('Not treated:')
                    display(not_treated)

    def explore_numerics(self, subset: Optional[List[str]] = None) -> None:
        
        display(Markdown('# Exploring numeric features'))

        for name, df in self._dataframes_dict.items():
            
            display(Markdown(f'### Numeric features in {name}'))

            num_features = list(df.select_dtypes(include='number').columns)
            for feature in num_features:
                kde_boxen_qq(df, feature)

    @staticmethod
    def _print_statistical_test(test_results):
        print(f'p-value: {test_results.pvalue}')
        if test_results.pvalue < 0.05:
            print('Reject null hypothesis!')
        else:
            print('Accept null hypothesis!')

    def compare_distributions(self, df_name, num_col_name, cat_col_name):


        sns.displot(
            data=self._dataframes_dict[df_name],
            x=num_col_name, hue=cat_col_name,
            kind='kde', fill=True
        )
        
        plt.title(f'{snake_to_title(num_col_name)} distribution by {snake_to_title(cat_col_name)}')

        sns.despine()
        plt.show()

        sns.boxplot(
            data=self._dataframes_dict[df_name],
            x=cat_col_name, y=num_col_name
        )

        # add mean labels
        means = self._dataframes_dict[df_name].groupby(cat_col_name)[num_col_name].mean()
        for i, mean in enumerate(means):
            plt.text(i, mean, round(mean, 2), horizontalalignment='center', size='large', color='w', weight='semibold')

        sns.despine()
        plt.show()

        if self._dataframes_dict[df_name][cat_col_name].nunique() > 2:
            raise NotImplementedError('Only 2 categories are supported!')

        # t-test to compare 2 means
        # TODO when dimensions will be implemented, select these dimensions at the beginning for DRY
        print('Compare means with t-test:')
        first = self._dataframes_dict[df_name][self._dataframes_dict[df_name][cat_col_name] == self._dataframes_dict[df_name][cat_col_name].unique()[0]][num_col_name]
        second = self._dataframes_dict[df_name][self._dataframes_dict[df_name][cat_col_name] == self._dataframes_dict[df_name][cat_col_name].unique()[1]][num_col_name]
        self._print_statistical_test(ttest_ind(first, second))

        # TODO: ANOVA or something else?



    # later it will be incorporated into dimensions concept
    @property
    def categorical_features(self):
        return [col for df in self._dataframes_dict.values() for col in df.select_dtypes(include='category').columns]
    
    @property
    def numeric_features(self):
        return [col for df in self._dataframes_dict.values() for col in df.select_dtypes(include='number').columns]