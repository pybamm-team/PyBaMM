# processed_variable_base.py  (new file)
from __future__ import annotations
import pybamm
import abc


class BaseProcessedVariable(abc.ABC):
    """
    Shared API for both 'on-demand' and 'computed' processed variables.
    """

    @abc.abstractmethod
    def __call__(self, **coords): ...

    @abc.abstractmethod
    def as_computed(self) -> pybamm.ProcessedVariableComputed: ...

    def _update(
        self,
        other: BaseProcessedVariable,
        new_sol: pybamm.Solution,
    ) -> pybamm.ProcessedVariableComputed:
        """
        Return a *new* CPV that is `self` followed by `other`.
        Works no matter which concrete types we start with.
        """
        return self.as_computed()._concat(other.as_computed(), new_sol)
