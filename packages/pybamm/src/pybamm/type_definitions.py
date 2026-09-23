from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

import pybamm

# numbers.Number should not be used for type hints
Numeric: TypeAlias = int | float | np.number

# expression tree
ChildValue: TypeAlias = float | npt.NDArray[Any]
ChildSymbol: TypeAlias = float | npt.NDArray[Any] | pybamm.Symbol

DomainType: TypeAlias = list[str] | str | None
AuxiliaryDomainType: TypeAlias = dict[str, str] | None
DomainsType: TypeAlias = dict[str, list[str] | str] | None
