import string
import re
from typing import List, Union

def remove_punctuation(s: str) -> str:
    return s.translate(str.maketrans('', '', string.punctuation))

def to_snake_case(s: str) -> str:
    # split by whitespace symbols
    words = re.split(r'\s+', s)

    # from camel case to snake case
    words = [re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower() for name in words]

    return '_'.join(words)

def generate_attr(name: Union[str, List[str]]) -> str:
    if isinstance(name, str):
        return to_snake_case(remove_punctuation(name))
    elif isinstance(name, list):
        return '_'.join([to_snake_case(remove_punctuation(n)) for n in name])
    else:
        raise TypeError('name must be str List[str]')

def snake_to_title(col_name: str) -> str:
    return col_name.replace('_', ' ').capitalize()