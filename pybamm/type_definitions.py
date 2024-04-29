from __future__ import annotations

from typing import Union
from typing_extensions import TypeAlias
import numpy as np
import pybamm

# numbers.Number should not be used for type hints
Numeric: TypeAlias = Union[int, float, np.number]

# expression tree
ChildValue: TypeAlias = Union[float, np.ndarray]
ChildSymbol: TypeAlias = Union[float, np.ndarray, pybamm.Symbol]

DomainType: TypeAlias = Union[list[str], str, None]
AuxiliaryDomainType: TypeAlias = Union[dict[str, str], None]
DomainsType: TypeAlias = Union[dict[str, Union[list[str], str]], None]
