from __future__ import annotations

import abc

import pybamm


def sensitivity_mismatch_message(name_lists: list[list[str]]) -> str:
    """Why a join of segments with these sensitivity parameters has none.

    Parameters
    ----------
    name_lists : list of list of str
        Each segment's sensitivity parameters, in join order.

    Returns
    -------
    str
    """
    described = []
    for names in name_lists:
        description = str(names) if names else "no sensitivities"
        if description not in described:
            described.append(description)
    return (
        "This solution joins segments with sensitivities to different parameters "
        f"({'; '.join(described)}) and no parameter is common to all of them, so "
        "it has no sensitivities. Solve every segment with the same "
        "`calculate_sensitivities`, or read each segment's own sensitivities from "
        "`sub_solutions`."
    )


class BaseProcessedVariable(abc.ABC):
    """
    Shared API for both 'on-demand' and 'computed' processed variables.
    """

    #: Why the sensitivities of a join of segments solved for different
    #: parameters cannot be read, or None. A class default for older pickles.
    _sensitivity_mismatch = None

    @abc.abstractmethod
    def __call__(self, **coords): ...

    @abc.abstractmethod
    def as_computed(self) -> pybamm.ProcessedVariableComputed: ...

    def update(
        self,
        other: BaseProcessedVariable,
        new_sol: pybamm.Solution,
    ) -> pybamm.ProcessedVariableComputed:
        """
        Return a *new* CPV that is `self` followed by `other`.
        Works no matter which concrete types we start with.
        """
        return self.as_computed()._concat(other.as_computed(), new_sol)

    def _check_sensitivities_available(
        self, available: bool, name: str | None = None
    ) -> None:
        """Raise if the variable's sensitivities cannot be read.

        Parameters
        ----------
        available : bool
            Whether the solve calculated sensitivities for the variable.
        name : str, optional
            The variable's name, for the message.

        Raises
        ------
        :class:`pybamm.SolverError`
            If the variable joins segments solved for sensitivities with respect
            to parameters none of which they share, or if its solve calculated
            no sensitivities.
        """
        if self._sensitivity_mismatch is not None:
            raise pybamm.SolverError(self._sensitivity_mismatch)
        if not available:
            variable = "This variable" if name is None else f"Variable {name!r}"
            raise pybamm.SolverError(
                f"{variable} has no sensitivities because its solve did not "
                "calculate any. Pass `calculate_sensitivities=True`, or a list of "
                "input parameter names, to the solve."
            )
