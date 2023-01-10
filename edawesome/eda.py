"""Main module with EDA class."""

__docformat__ = "markdown"

from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional, List
import functools
import warnings

from glob import glob
import os
import sqlite3

from kaggle.api.kaggle_api_extended import KaggleApi
import patoolib

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
# It is required to set this environment variable to '1'
# in both driver and executor sides if you use pyarrow>=2.0.0.


from pyspark.sql import SparkSession
from pyspark.pandas import read_csv as spark_read_csv
import pyspark

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

<<<<<<< HEAD
=======
from scipy.stats import ttest_ind, f_oneway
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

from transitions import Machine

from .str_help import generate_attr, snake_to_title, get_files
from .sns_help import kde_boxen_qq
<<<<<<< HEAD
from .pd_help import display_df
from .eda_help import compare_distributions

from .configs import CATEGORY_MAX

# TODO: add dimensions concept
# TODO: configure logging as a possible alternative to IPython display
=======
from .configs import CATEGORY_MAX

# TODO: add dimensions concept
# TODO: configure logging, not IPython display
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

def basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _check_duplicated_features(func):
    """Decorator for checking duplicated features"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        all_list = [col for df in self._dataframes_dict.values() for col in df.columns]
        if len(all_list) != len(set(all_list)):
            warnings.warn('There are duplicated features in the dataframes!')
        return func(self, *args, **kwargs)
    return wrapper

@dataclass
class EDA:
    """
    Main class for simple and robust EDA with pandas and seaborn.
    
    The only required argument is `data_dir_path`, which is the full path to the folder, where all the data files will be placed. EDAwesome will find all csv/tsv and also sqlite files in this folder and load these files.

    Possible data sources:
    - `kaggle_datasets`: list of kaggle datasets to download and extract (like `"swaptr/covid19-state-data"`)
    - `archives`: list of full paths to archives to extract (like `"/home/user/data.zip"`) into the data folder, it supports various formats with patool
    
    Boolean options:

    - `silent`: if True, during loading the dataframes will be shown
    - `concat`: if True, all the loaded dataframes will be vertically concateated into one (not implemented yet)
    - `use_polars`: if True, polars will be used for loading instead of pandas (not implemented yet)
    - `use_pyspark`: if True, pyspark will be used for loading instead of pandas
    - `lazy_sqlite`: if True, sqlite3 won't be loaded, but edawesome will query the database when needed (not implemented yet)
    
    Numerical options:

    - `load_limit`: maximum number of rows to load in each dataframe
    - `pandas_mem_limit`: memory treashold after which to use pyspark or polars
    - `pyspark_mem_limit`: memory limit for pyspark
    """

    data_dir_path: str
    kaggle_datasets: Optional[List[str]] = None
<<<<<<< HEAD
    archives: Optional[List[str]] = None
    # List of supported archives can be found on patool github page: https://github.com/wummel/patool
=======
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
    # TODO: dictionary with csv files as keys and separator as values

    state: Machine = field(init=False)

    silent: Optional[bool] = False
    concat: Optional[bool] = False
    use_polars: Optional[bool] = False
    use_pyspark: Optional[bool] = False
    lazy_sqlite: Optional[bool] = False
    
    load_limit: Optional[int] = None
    pandas_mem_limit: Optional[int] = 1024**2
    pyspark_mem_limit: Optional[str] = '4g'

    @property
    def data_list(self) -> List[pd.DataFrame]:
        return list(self._dataframes_dict.keys())

    def __post_init__(self) -> None:

        if self.concat or self.use_polars:
            raise NotImplementedError
            # self.concat will be used to concat all the loaded dataframes into one
            # self.use_polars will be used to load data with polars instead of pandas (it is faster and uses less memory)

        states = [
            'init',
            'loaded',
            'cleaned',
            'categorized'
        ]
        self.state = Machine(states=states, initial='init')
        self.state.add_ordered_transitions(loop=False)

<<<<<<< HEAD
        if self.use_polars and self.use_pyspark:
            raise ValueError('You can use only one of polars or pyspark! Later there will be a map funcionality.')
        
        if self.use_pyspark and self.load_limit:
            raise ValueError('You can use load_limit only with pandas and polars!')

        if self.pandas_mem_limit and not (self.use_polars or self.use_pyspark):
            raise ValueError('You can use pandas_mem_limit only with pandas and polars! It is needed to define minimum number of rows to use pyspark or polars.')

        if self.pyspark_mem_limit and not self.use_pyspark:
            raise ValueError('You can use pyspark_mem_limit only with pyspark!')

        if self.lazy_sqlite:
            raise NotImplementedError

    def next(self) -> None:
        """Change state to the next one."""
        self.state.next_state()
        display(Markdown(f'### EDA is in {self.state.state} state now'))
        if self.state.state == 'loaded':
            # Separate messages into some "front-end" class or module
            message = '\n'.join([
                'Now you can drop or impute missing values. Here are some useful methods:',
                '- `clean_check` - check number for missing values in each column',
                '- `drop_empty_columns` -  drop columns with a lot of missing values',
                '- `clean` - drop duplicates and missing values (deprecated)'
            ])
            display(Markdown(message))
=======
        pd.set_option('display.max_rows', 10)
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

    def next(self) -> None:
        self.state.next_state()
        display(Markdown(f'### EDA is in {self.state.state} state now'))

    @staticmethod
    def prettify_col_names(df: pd.DataFrame) -> None:
        """Rename columns to snake case and set index if it is auto index."""

        # detect auto index
        first_col = df.columns[0]
        if first_col == 'Unnamed: 0':
            df.rename(columns={first_col: 'index'}, inplace=True)
            first_col = 'index'

        if (df[first_col] == df.index).all():
            df.set_index(first_col, inplace=True)

        # columns to snake case
        df.columns = df.columns.to_series().apply(generate_attr)

<<<<<<< HEAD
    @staticmethod
    def display_df(name: str, df: pd.DataFrame) -> None:
        """Display dataframe with shape."""
        display_df(name, df)

    def _add_df(self, name: str, df):
        """Add dataframe to the EDA object."""

=======
    def add_df(self, name: str, df: pd.DataFrame):

>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
        if self.state.state != 'init':
            raise ValueError(f'Dataframes can be added only in init state, not {self.state.state}!')

        if hasattr(self, name):
            raise ValueError(f'Dataframe or other attribute with name {name} already exists. All filenames must be unique!')

        self.prettify_col_names(df)
        self._dataframes_dict[name] = df

        if not self.silent:
<<<<<<< HEAD
            self.display_df(name, df)

    def rename_df(self, old_name: str, new_name: str) -> None:
        """Rename the dataframe."""
=======
            display(Markdown(f'### {name}'))
            display(df)

    def rename_df(self, old_name: str, new_name: str) -> None:
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

        if old_name not in self._dataframes_dict.keys():
            raise ValueError(f'Dataframe with name {old_name} does not exist!')
            
        if new_name in self._dataframes_dict.keys():
            raise ValueError(f'Dataframe with name {new_name} already exists. All filenames must be unique!')
<<<<<<< HEAD
            # TODO: DRY
            # Separate routine checks (like this one) from the main logic into decorators
=======
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
    
        self._dataframes_dict[new_name] = self._dataframes_dict.pop(old_name)


    def _load_sqlite(self) -> None:
        """Load sqlite files from data_dir_path."""

        sqlite_extensions = ('.db', '.sqlite', '.sqlite3')
        # TODO move extensions and othe settings-like variables to config file (TOML?)
        sqlite_files = []
        for ext in sqlite_extensions:
            # recursive search for sqlite files
            os.chdir(self.data_dir_path)
            sqlite_files += glob(f'*{ext}', recursive=True)
        sqlite_files = get_files(sqlite_files)
        
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
                if self.load_limit:
                    query.replace(';', f'\nLIMIT {self.load_limit};')
                df = pd.read_sql(query, connection)
                self.add_df(table, df)

    
    def _load_csv_pyspark(self, file, sep: Optional[str] = None):
        """Load single csv file with pyspark."""

        # set memory limit for pyspark (to avoid crashing of kernel)
        os.environ['PYSPARK_SUBMIT_ARGS'] = f'--driver-memory {self.pyspark_mem_limit} pyspark-shell'

        spark = SparkSession.builder.getOrCreate()
        spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
        # set log level to ERROR to avoid unnecessary messages
        # sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel)
        spark.sparkContext.setLogLevel('ERROR')

        spark_read_csv_args = {
            'inferSchema': True
        }

        if sep:
            spark_read_csv_args['sep'] = sep

        if self.load_limit:
            spark_read_csv_args['nrowsint'] = self.load_limit

        df = spark_read_csv(file, **spark_read_csv_args)
        
        return df

    def _load_csv(self) -> None:
        """Load csv files from data_dir_path."""
        
        os.chdir(self.data_dir_path)

        csv_files = get_files(glob(f'*.csv', recursive=True))
        tsv_files = get_files(glob(f'*.tsv', recursive=True))

        for file in tsv_files:
            
            # TODO: implement polars support
            # TODO: doesn't check each iteration of for loop
            file_size = os.path.getsize(file)
            if file_size > self.pandas_mem_limit and self.use_pyspark:
                df = self._load_csv_pyspark(file, sep='\t')
            else:
<<<<<<< HEAD
                if self.load_limit:
                    df = pd.read_csv(file, sep='\t', nrows=self.load_limit)
                else:
                    df = pd.read_csv(file, sep='\t')
            
            self._add_df(generate_attr(basename_no_ext(file)), df)
=======
                df = pd.read_csv(file, sep='\t')
            self.add_df(generate_attr(basename_no_ext(file)), df)
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

        for file in csv_files:
<<<<<<< HEAD
            
            # TODO: DRY!
            file_size = os.path.getsize(file)
            if file_size > self.pandas_mem_limit and self.use_pyspark:
                df = self._load_csv_pyspark(file)
            else:

                try:
                    if self.load_limit:
                        df = pd.read_csv(file, nrows=self.load_limit, sep=None, engine='python')
                    else:
                        df = pd.read_csv(file, sep=None, engine='python')
                except UnicodeDecodeError:
                    if self.load_limit:
                        df = pd.read_csv(file, nrows=self.load_limit, sep=None, engine='python', encoding='latin1')
                    else:
                        df = pd.read_csv(file, sep=None, engine='python', encoding='latin1')
            
            self._add_df(generate_attr(basename_no_ext(file)), df)
=======
            df = pd.read_csv(file, sep=None, engine='python')
            self.add_df(generate_attr(basename_no_ext(file)), df)
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

        # TODO: automatic separator detection, files without extension
        # TODO: provide dict with file names or extensions as keys and separators as values
        # TODO: automatic header and index detection, multiple index levels

    def _download_kaggle(self) -> None:
        """Download kaggle datasets from kaggle_datasets list."""

        api = KaggleApi()
        api.authenticate()
        os.chdir(self.data_dir_path)

        for dataset in self.kaggle_datasets:
            api.dataset_download_files(dataset, path=self.data_dir_path)
            dataset_filename = dataset.split('/')[-1] + '.zip'
            dataset_path = os.path.join(self.data_dir_path, dataset_filename)
            self.archives.append(dataset_path)

    def _extract_archives(self) -> None:
        """Extract archives from archives list."""
        for archive in self.archives:
            ext = os.path.splitext(archive)[1].replace('.', '')
            if ext not in patoolib.ArchiveFormats:
                raise ValueError(f'Unsupported archive format {ext}! Supported formats: {patoolib.ArchiveFormats}')
            patoolib.extract_archive(archive, outdir=self.data_dir_path)

    def load_data(self) -> None:
        """Load data from data_dir_path, kaggle_datasets and archives lists into EDA object."""

        if self.state.state != 'init':
            raise ValueError(f'Data can be loaded only in init state, not {self.state.state}!')

        if self.state.state != 'init':
            raise ValueError(f'Data can be loaded only in init state, not {self.state.state}!')

        if not self.silent:
            display(Markdown('# Loading data'))
        
        self._dataframes_dict = {}
        if self.kaggle_datasets:
            self._download_kaggle()
        if self.archives:
            self._extract_archives()
        self._load_sqlite()
        self._load_csv()

        display(Markdown(f'**Loaded {len(self._dataframes_dict)} dataframes**'))
        display(Markdown(f'**To add dataframes, use `add_df` method**'))
        display(Markdown(f'**After data loading is completed, call `eda.next()`**'))

    def clean_check(self, subset: Optional[List[str]] = None) -> None:
        """Check for nulls and duplicates in the dataframes."""
        
        display(Markdown('# Check for bad values'))

        for name, df in self._dataframes_dict.items():
            
            display(Markdown(f'### Nulls and datatypes in {name}'))
            display(df.info())

            # calculate ratio of nulls in each column
<<<<<<< HEAD
            nulls = (df.isnull().sum() / len(df)).to_numpy()
            for ratio, col in zip(nulls, df.columns):
                if ratio:
                    col_dtype = df[col].dtype
                    message = f'**Column `{col}` ({col_dtype}) has {ratio*100:.2f}% nulls.**'
                    if ratio > 0.1:
                        message += ' **Consider dropping this column before calling `clean`**'
                    display(Markdown(message))
=======
            nulls = df.isnull().sum() / len(df)
            for ratio, col in zip(nulls, df.columns):
                if ratio:
                    display(Markdown(f'**Column {col} has {ratio*100:.2f}% nulls**'))
                    if ratio > 0.1:
                        display(Markdown(f'**Consider dropping this column before calling `clean`**'))
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
            
            display(Markdown(f'### Duplicates in {name}'))
            duplicates_num = df.duplicated().sum()
            if duplicates_num:
                # show rows with some duplicates
                duplicated_df = df[df.duplicated(keep=False)]
                display(Markdown(f'**Found {duplicates_num} duplicated rows in the following columns:**'))
                display(df[df.duplicated(keep=False)].groupby(list(df.columns)).size().reset_index().rename(columns={0:'count'}))
                display(Markdown(f'**Some of the duplicated rows:**'))
                display(duplicated_df.head(20))
            else:
                print('No duplicates found...')

        display(Markdown(f'**Use `clean` method to drop duplicated and containing nulls rows or perform other preprocessing procedures.**'))
        display(Markdown(f'**After cleaning is completed, call `eda.next()`**'))

    def clean(self, nulls: bool, duplicates: bool) -> None:
<<<<<<< HEAD
        """Drop rows with nulls or duplicates"""

        # WARNING: .any() can caause errors, this method is currently deprecated
        raise DeprecationWarning("This method is deprecated, because it can cause errors in some cases, and it's usually not a good idea just to drop the missing values.")
=======
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

        if self.state.state != 'loaded':
            raise ValueError(f'Data can be cleaned only in loaded state, not {self.state.state}!')

        display(Markdown('# Cleaning data'))

        for name, df in self._dataframes_dict.items():
            
            if nulls:
                display(Markdown(f'### Drop nulls in {name}'))
                null_nums = df.isnull().sum()
                if null_nums.any():
                    df.dropna(inplace=True, axis='rows')
                    print(f'Dropped {null_nums.sum()} rows with nulls')
                    print(f'No subset is used by default!')
                else:
                    print('No nulls found...')

            if duplicates:
                display(Markdown(f'### Drop duplicates in {name}'))
                duplicates_nums = df.duplicated().sum()
                if duplicates_nums.any():
                    df.drop_duplicates(inplace=True)
                    print(f'Dropped {duplicates_nums.sum()} duplicates')
                    print(f'No subset is used by default!')
                else:
                    print('No duplicates found...')
<<<<<<< HEAD

    def drop_empty_columns(self, treshold: float = 0.1):
        """Drop columns with more than treshold nulls"""
        for name, df in self._dataframes_dict.items():
            display(Markdown(f'### Drop empty columns in {name}'))
            nulls = df.isnull().sum() / len(df)
            for ratio, col in zip(nulls, df.columns):
                if ratio > treshold:
                    df.drop(col, axis='columns', inplace=True)
                    print(f'Dropped column {col} with {ratio*100:.2f}% nulls')

    def categorize(self, subset: Optional[List[str]] = None) -> None:
        """Categorize columns with less than CATEGORY_MAX unique values"""
=======

    def categorize(self, subset: Optional[List[str]] = None) -> None:
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

        if self.state.state != 'cleaned':
            raise ValueError(f'Data can be categorized only in cleaned state, not {self.state.state}!')
        
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

        display(Markdown(f'**You can change some categories by hands. After categorization is completed, call `eda.next()`**'))

    def explore_numerics(self, subset: Optional[List[str]] = None) -> None:
<<<<<<< HEAD
        """Explore numeric features"""
=======
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

        if self.state.state != 'categorized':
            raise ValueError(f'Data can be explored only in categorized state, not {self.state.state}!')
        
        display(Markdown('# Exploring numeric features'))

        for name, df in self._dataframes_dict.items():
            
            display(Markdown(f'### Numeric features in {name}'))

            num_features = list(df.select_dtypes(include='number').columns)
            for feature in num_features:
                kde_boxen_qq(df, feature)

<<<<<<< HEAD
    def explore_categories(self, df_name: Optional[str], subset: Optional[List[str]] = None) -> None:
        """Explore categorical features with countplots"""

        if df_name is not None:
            dfs_to_explore = {df_name: self._dataframes_dict[df_name]}
=======
    def explore_categories(self, subset: Optional[List[str]] = None) -> None:

        if self.state.state != 'categorized':
            raise ValueError(f'Data can be explored only in categorized state, not {self.state.state}!')
        
        display(Markdown('# Exploring categorical features'))

        for name, df in self._dataframes_dict.items():
            
            display(Markdown(f'### Categorical features in {name}'))

            cat_features = list(df.select_dtypes(include='category').columns)
            for feature in cat_features:
                
                sns.countplot(
                    data=df,
                    x=feature
                )

                plt.bar_label(plt.gca().containers[0])
                plt.title(snake_to_title(feature))
                sns.despine()
                plt.show()

    @staticmethod
    def _print_statistical_test(test_results):
        print(f'p-value: {test_results.pvalue}')
        if test_results.pvalue < 0.05:
            print('Reject null hypothesis!')
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
        else:
            dfs_to_explore = self._dataframes_dict

<<<<<<< HEAD
        if subset is not None:

            if not df_name:
                raise ValueError('If subset is specified, df_name must be specified too!')

            dfs_to_explore = {df_name: dfs_to_explore[df_name][subset]}


        if self.state.state != 'categorized':
            raise ValueError(f'Data can be explored only in categorized state, not {self.state.state}!')
=======
    def _select_dimension(self, df_name, col_name):

        if df_name not in self._dataframes_dict.keys():
            raise ValueError(f'Dataframe {df_name} is not loaded')
        if col_name not in self._dataframes_dict[df_name].columns:
            raise ValueError(f'Column {col_name} is not found in {df_name}')

        return self._dataframes_dict[df_name][col_name]

    def compare_distributions(self, df_name, num_col_name, cat_col_name):

        if self.state.state != 'categorized':
            raise ValueError(f'Data can be compared only in categorized state, not {self.state.state}!')

        sns.displot(
            data=self._dataframes_dict[df_name],
            x=num_col_name, hue=cat_col_name,
            kind='kde', fill=True
        )
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
        
        display(Markdown('# Exploring categorical features'))

        for name, df in dfs_to_explore.items():
            
            display(Markdown(f'### Categorical features in {name}'))

            cat_features = list(df.select_dtypes(include='category').columns)
            for feature in cat_features:
                
                sns.countplot(
                    data=df,
                    x=feature
                )

                plt.bar_label(plt.gca().containers[0])
                plt.title(snake_to_title(feature))
                sns.despine()
                plt.show()

    def _select_dimension(self, df_name, col_name):
        """Select some variable from the dataframe"""

<<<<<<< HEAD
        raise DeprecationWarning('This method is for future use!')

        if df_name not in self._dataframes_dict.keys():
            raise ValueError(f'Dataframe {df_name} is not loaded')
        if col_name not in self._dataframes_dict[df_name].columns:
            raise ValueError(f'Column {col_name} is not found in {df_name}')

        return self._dataframes_dict[df_name][col_name]

    def compare_distributions(self, df_name: str, num_col_name: str, cat_col_name: str, treat_nans: str = 'drop'):
        """Compare distributions of numeric feauture by categorical feature"""

        # TODO some decorator for checking state
        # TODO some decorator for checking possible values of the arguments and raising errors

        if self.state.state != 'categorized':
            raise ValueError(f'Data can be compared only in categorized state, not {self.state.state}!')

        compare_df = self._dataframes_dict[df_name][[num_col_name, cat_col_name]]
        if isinstance(compare_df, pyspark.pandas.frame.DataFrame):
            compare_df = compare_df.to_pandas()

        compare_distributions(compare_df, num_col_name, cat_col_name, treat_nans)
    

    @_check_duplicated_features
    def get_featues_by_type(self, dtype, df_name: Optional[str] = None):
        """Get features by type"""
        if df_name is None:
            features_list = [col for df in self._dataframes_dict.values() for col in df.select_dtypes(include=dtype).columns]
        else:
            features_list = self._dataframes_dict[df_name].select_dtypes(include=dtype).columns.tolist()

        return features_list

=======
        num_col = self._select_dimension(df_name, num_col_name)
        cat_col = self._select_dimension(df_name, cat_col_name)

        if self._dataframes_dict[df_name][cat_col_name].nunique() > 2:
            # ANOVA, comparing num_col distribution by cat_col
            print('Compare means with ANOVA:')
            test_results = f_oneway(*[num_col[cat_col == cat] for cat in cat_col.unique()])
            self._print_statistical_test(test_results)
        elif self._dataframes_dict[df_name][cat_col_name].nunique() == 2:
            print('Compare means with t-test:')

            first = num_col[cat_col == cat_col.unique()[0]]
            second = num_col[cat_col == cat_col.unique()[1]]
            self._print_statistical_test(ttest_ind(first, second))
        else:
            raise ValueError('Categorical feature has only one category!')

>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
    @property
    def categorical_features(self, df_name: Optional[str] = None):
        return self.get_featues_by_type('category', df_name=df_name)
    
    @property
<<<<<<< HEAD
    def numeric_features(self, df_name: Optional[str] = None):
        return self.get_featues_by_type('number', df_name=df_name)

    @property
    def not_treated_features(self, df_name: Optional[str] = None):
        return self.get_featues_by_type('object', df_name=df_name)
    
    @property
    @_check_duplicated_features
    def all_features(self, df_name: Optional[str] = None):
        if df_name is None:
            features_list = [col for df in self._dataframes_dict.values() for col in df.columns]
        else:
            features_list = self._dataframes_dict[df_name].columns.tolist()

        return features_list
=======
    def numeric_features(self):
        return [col for df in self._dataframes_dict.values() for col in df.select_dtypes(include='number').columns]
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1

    @property
    def data(self):
        Data = namedtuple('Data', self._dataframes_dict)
<<<<<<< HEAD
        return Data(**self._dataframes_dict)
=======
        return Data(**self._dataframes_dict)
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
