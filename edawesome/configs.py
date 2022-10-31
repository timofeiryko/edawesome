"""
This file contains all the constants used in the project. In the future, TOML file outside of the package will be used.
"""

# Use TOML file which can be changed by user

SUPPORTED_INPUT_FORMATS = (
    'csv', 'tsv',
    'xls', 'xlsx'
)

# TODO add other formats (like in pd) and test it

CATEGORY_MAX = 5