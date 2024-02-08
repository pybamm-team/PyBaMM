"""Common type definitions for PyBaMM"""

from typing import Union
import numpy as np
import pybamm

# numbers.Number should not be used for type hints
Numeric = Union[int, float, np.number]

# expression tree
ChildValue = Union[float, np.ndarray]
ChildSymbol = Union[float, np.ndarray, pybamm.Symbol]

DomainType = Union[list[str], str, None]
AuxiliaryDomainType = Union[dict[str, str], None]
DomainsType = Union[dict[str, Union[list[str], str]], None]
