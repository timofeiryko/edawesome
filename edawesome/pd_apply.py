"""Functions designed to apply to pandas dataframes."""

from typing import Dict, List, Optional, Sequence

# TODO Better mapper functionality (inversed etc)
def multi_replace(
    row: str, mapper: Dict[str, Sequence[str]],
    default: Optional[str] = None
) -> str:
    """Replace a string with another string based on a mapper."""

    if default is None:
        default = row

    for output_str, inputs in mapper.items():
        if row in inputs:
            return output_str
    
    return default

def cat_to_num(
    row: str, order: List[str]
) -> float:
    """Convert a categorical variable to a number based on an order in provided list."""
    return order.index(row) / (len(order) - 1)