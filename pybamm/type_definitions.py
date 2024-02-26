#
# Common type definitions for PyBaMM
#
from __future__ import annotations

from typing import Union, List, Dict
import numpy as np
import pybamm
from pybamm.util import have_optional_dependency

TypeAlias = have_optional_dependency("typing_extensions", "TypeAlias")

# numbers.Number should not be used for type hints
Numeric: TypeAlias = Union[int, float, np.number]

# expression tree
ChildValue: TypeAlias = Union[float, np.ndarray]
ChildSymbol: TypeAlias = Union[float, np.ndarray, pybamm.Symbol]

DomainType: TypeAlias = Union[List[str], str, None]
AuxiliaryDomainType: TypeAlias = Union[Dict[str, str], None]
DomainsType: TypeAlias = Union[Dict[str, Union[List[str], str]], None]
