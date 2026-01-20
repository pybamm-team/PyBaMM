from functools import wraps

import pybamm


def iterate_solvers(func):
    """
    Decorator that evaluates a for loop over the sub_solvers to test
    each solver's method until one succeeds.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        errors = []
        for solver in self.sub_solvers:
            try:
                return getattr(solver, func.__name__)(*args, **kwargs)
            except pybamm.SolverError as e:
                errors.append(str(e))
        raise pybamm.SolverError(f"All sub_solvers failed: {errors}") from None

    return wrapper


class CompositeSolver(pybamm.BaseSolver):
    """
    Experimental composite solver that tries each solver in order until one succeeds.
    """

    def __init__(self, sub_solvers: list[pybamm.BaseSolver] | None = None):
        """
        Parameters
        ----------
        sub_solvers : list[pybamm.BaseSolver]
            The list of solvers to try in order.
        """
        if not sub_solvers:
            raise ValueError("No sub_solvers provided")
        if not isinstance(sub_solvers, list) and not all(
            isinstance(s, pybamm.BaseSolver) for s in sub_solvers
        ):
            raise ValueError("sub_solvers must be a list of pybamm.BaseSolver")
        self.sub_solvers: list[pybamm.BaseSolver] = sub_solvers
        self.name = "Composite solver"

    @iterate_solvers
    def solve(self, *args, **kwargs):
        pass
