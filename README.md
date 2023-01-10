# EDAwesome

This is a package for quick, easy and customizable data analysis with pandas and seaborn. We automate all the routine so you can focus on the data. Clear and cuztomizable workflow is proposed.

## Installation

EDAwesome is generally compatible with standard Anaconda environment in therms of dependencies. So, you can just install it in you environment with pip:

```bash
pip install edawesome
```

You can also install the dependencies, using `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you use Poetry, just include the depedencies in your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = ">=3.8,<3.11"
seaborn = "^0.12.1"
kaggle = "^1.5.12"
ipython = "^8.5.0"
transitions = "^0.9.0"
patool = "^1.12"
pyspark = "^3.3.1"
pandas = "^1.5.2"
statsmodels = "^0.13.5"
scikit-learn = ">=1.2.0"
scipy = "~1.8.0"
```

## Usage

This package is designed to be used in Jupyter Notebook. You can use step-by-step workflow or just import the functions you need. Below is the example of the step-by-step workflow:

### Quick start

```python
from edawesome.eda import EDA

eda = EDA(
    data_dir_path='/home/dreamtim/Desktop/Coding/turing-ds/MachineLearning/tiryko-ML1.4/data',
    archives=['/home/dreamtim//Downloads/home-credit-default-risk.zip'],
    use_pyspark=True,
    pandas_mem_limit=1024**2,
    pyspark_mem_limit='4g'   
)
```

This will create the `EDA` object. Now you can load the data into your EDA:

```python
eda.load_data()
```

This will display the dataframes and their shapes. You can also use `eda.dataframes` to see the dataframes. Now you can go to the next step:

```python
eda.next()
eda.clean_check()
```

Let us say, that we don't want to do any cleaning in this case. So, we just go to the next step:

```python
eda.next()
eda.categorize()
```

Now you can compare some numerical column by category just in one line:

```python
eda.compare_distributions('application_train', 'ext_source_3', 'target')
```

### Real-world example

Full notebook which was used for examples above can be found in one of my real ML projects.

There is also an example `quickstart.ipynb` notebook in this repo.

### Documentation

You can find the documentation [here](https://timofeiryko.github.io/edawesome).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.