from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

import pybamm

# numbers.Number should not be used for type hints
Numeric: TypeAlias = Union[int, float, np.number]

# expression tree
ChildValue: TypeAlias = Union[float, npt.NDArray[Any]]
ChildSymbol: TypeAlias = Union[float, npt.NDArray[Any], pybamm.Symbol]

DomainType: TypeAlias = Union[list[str], str, None]
AuxiliaryDomainType: TypeAlias = Union[dict[str, str], None]
DomainsType: TypeAlias = Union[dict[str, Union[list[str], str]], None]
