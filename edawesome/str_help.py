"""Just useful scripts for string manipulation."""
import string
import re
from typing import List, Union
import os

def remove_punctuation(s: str) -> str:
    # exclude _ from punctuation
    return s.translate(str.maketrans('', '', string.punctuation.replace('_', '')))

def to_snake_case(s: str) -> str:

<<<<<<< HEAD
    s = s.strip().lower()

=======
>>>>>>> be2797a5b11fbcebb53306d503ac0244060d9ce1
    # split by whitespace symbols
    words = re.split(r'\s+', s)

    # from camel case to snake case
    words = [re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower() for name in words]

    # replace - and . with _
    words = [re.sub(r'[\-\.]', '_', name).lower() for name in words]

    # remove punctuation
    words = [remove_punctuation(name) for name in words]

    return '_'.join(words)

def generate_attr(name: Union[str, List[str]]) -> str:
    if isinstance(name, str):
        return to_snake_case(name)
    elif isinstance(name, list):
        return '_'.join([to_snake_case(n) for n in name])
    else:
        raise TypeError('name must be str or List[str]')

def snake_to_title(col_name: str) -> str:
    return col_name.replace('_', ' ').capitalize()

# function to check is path a directory for all paths in a list and return only files
def get_files(path: Union[str, List[str]]) -> List[str]:
    if isinstance(path, str):
        if os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        else:
            return [path]
    elif isinstance(path, list):
        return [f for p in path for f in get_files(p)]
    else:
        raise TypeError('path must be str or List[str]')

    