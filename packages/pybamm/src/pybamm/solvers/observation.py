"""How a :class:`pybamm.Solution` reads its variables.

:func:`build_variable` turns a variable name into a processed variable over a
Solution's sub-solutions, its *segments*, reading each segment through its own
model. A *leaf* is one segment's compiled form of one variable; each model keeps
its leaves in an :class:`ObserverCache`, so every solution of it reuses them.

:class:`OutputAssembly` is the ``output_variables`` counterpart: the solver has
already computed those variables, so it owns the payload's row layout and
populates the Solution eagerly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import accumulate, pairwise

import casadi
import numpy as np
import numpy.typing as npt

import pybamm
from pybamm.solvers.base_processed_variable import BaseProcessedVariable
from pybamm.solvers.variable_observer import (
    CasadiObserver,
    check_variable_in_solve,
    pack_sensitivity_dict,
)


class ObserverCache:
    """The compiled leaves of one model's variables, keyed by variable name.

    A name is assumed to keep one expression for the model's lifetime. Each leaf
    is built for the input layout and solution options (such as ``compile`` and
    ``cse``) of the first solve that reads it, and later solves reuse it as is.
    """

    def __init__(self):
        # Key -> (leaf, the expression it evaluates, time integral or None)
        self._casadi_leaves: dict[str, tuple] = {}

    @classmethod
    def of(cls, model: pybamm.BaseModel) -> ObserverCache:
        """The cache of ``model``, created on first use."""
        if model._observer_cache is None:
            model._observer_cache = cls()
        return model._observer_cache

    def copy(self) -> ObserverCache:
        """A cache that starts with these leaves and grows independently."""
        new = type(self)()
        new._casadi_leaves = self._casadi_leaves.copy()
        return new

    def casadi_leaf(
        self,
        solution: pybamm.Solution,
        key: str,
        var_pybamm: pybamm.Symbol,
        inputs: dict,
        ys_shape: tuple[int, ...],
    ) -> tuple[
        casadi.Function, pybamm.Symbol, pybamm.ProcessedVariableTimeIntegral | None
    ]:
        """One segment's CasADi leaf for ``var_pybamm``, converted on first use.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution being read, whose options the conversion uses.
        key : str
            The cache key: the variable name, or a vector field component's.
        var_pybamm : :class:`pybamm.Symbol`
            The variable's discretised expression.
        inputs : dict
            The segment's inputs, which fix the leaf's parameter layout.
        ys_shape : tuple of int
            The shape of the segment's states.

        Returns
        -------
        tuple
            The CasADi function, the expression it evaluates (a time integral's
            integrand) and the time integral, or None.
        """
        entry = self._casadi_leaves.get(key)
        if entry is None:
            entry = solution._convert_to_casadi(var_pybamm, inputs, ys_shape)
            self._casadi_leaves[key] = entry
        return entry


def build_variable(solution: pybamm.Solution, name: str) -> BaseProcessedVariable:
    """The processed variable ``name`` of ``solution``, ready to evaluate.

    Parameters
    ----------
    solution : :class:`pybamm.Solution`
        The solution to read; each segment is read through its own model.
    name : str
        Variable name, as registered on the models.

    Returns
    -------
    :class:`pybamm.solvers.base_processed_variable.BaseProcessedVariable`
        Evaluable over every segment of ``solution``.
    """
    pybamm.logger.debug(f"Post-processing {name}")
    time_integral = None
    vars_pybamm = [
        model.get_processed_variable_or_event(name) for model in solution.all_models
    ]
    vars_casadi = [None] * len(solution.all_models)
    for i, (model, ys, inputs) in enumerate(
        zip(solution.all_models, solution.all_ys, solution.all_inputs, strict=True)
    ):
        var_pybamm = vars_pybamm[i]
        check_variable_in_solve(solution, name, var_pybamm)
        cache = ObserverCache.of(model)
        if isinstance(var_pybamm, pybamm.VectorField):
            vars_casadi[i] = [
                cache.casadi_leaf(
                    solution, f"{name}[{k}]", component, inputs, ys.shape
                )[0]
                for k, component in enumerate(var_pybamm.components)
            ]
        else:
            vars_casadi[i], vars_pybamm[i], time_integral = cache.casadi_leaf(
                solution, name, var_pybamm, inputs, ys.shape
            )
    return pybamm.process_variable(
        name,
        vars_pybamm,
        CasadiObserver(vars_casadi),
        solution,
        time_integral=time_integral,
    )


class OutputAssembly:
    """The row layout of an ``output_variables`` payload, and its attachment.

    A solve run with ``output_variables`` propagates no state trajectory: the
    solver evaluates the requested variables itself and returns one row per
    non-zero of each variable's CasADi function, in variable order, so a vector
    variable spans several consecutive rows.

    Parameters
    ----------
    names : list of str
        The output variables, in row order.
    casadi_fns : dict
        Map of name to the CasADi function ``f(t, y, p)`` the rows were
        evaluated by. Its non-zeros set the rows each variable owns.
    time_integrals : dict, optional
        Map of name to :class:`pybamm.ProcessedVariableTimeIntegral` for the
        outputs whose rows carry an integrand rather than the variable itself.
    """

    def __init__(
        self,
        names: Sequence[str],
        casadi_fns: Mapping[str, casadi.Function],
        *,
        time_integrals: Mapping[str, pybamm.ProcessedVariableTimeIntegral]
        | None = None,
    ):
        self._names = tuple(names)
        self._casadi_fns = {name: casadi_fns[name] for name in self._names}
        self._time_integrals = dict(time_integrals or {})
        lens = []
        # name -> sparsity of the variables whose rows are sparse
        self._sparse = {}
        for name in self._names:
            sparsity = self._casadi_fns[name].sparsity_out(0)
            lens.append(sparsity.nnz())
            if sparsity.nnz() != sparsity.numel():
                self._sparse[name] = sparsity
        offsets = list(accumulate(lens, initial=0))
        self._n_rows = offsets[-1]
        # One slice of payload rows per output variable
        self._rows = tuple(slice(start, end) for start, end in pairwise(offsets))

    @property
    def n_rows(self) -> int:
        """Rows in one time point of the payload."""
        return self._n_rows

    def attach(
        self,
        solution: pybamm.Solution,
        data: npt.ArrayLike,
        *,
        sensitivities: npt.ArrayLike | None = None,
        sensitivity_names: Sequence[str] = (),
    ) -> None:
        """Populate ``solution``'s variables from one outputs-only payload.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution to populate, built with ``variables_returned=True``.
        data : array-like
            Output trajectory of shape ``(n_t, n_rows)``, time-outer.
        sensitivities : array-like, optional
            Output sensitivities of shape ``(n_t, n_rows, n_p)``. Omit when the
            solve carried none, which leaves every variable's sensitivities empty
            rather than lazily recomputed: an outputs-only solve keeps no state
            to recompute them from.
        sensitivity_names : list of str, optional
            Sensitivity-parameter names, in ``sensitivities``' column order.

        Raises
        ------
        :class:`pybamm.SolverError`
            If the payload does not match this layout, or if sensitivities were
            requested for a variable whose CasADi rows are sparse.
        """
        data = self._checked_rows(data)
        if sensitivities is not None:
            sensitivities = self._checked_sensitivities(
                sensitivities, data.shape[0], sensitivity_names
            )
        model = solution.all_models[0]
        for name, rows in zip(self._names, self._rows, strict=True):
            time_integral = self._time_integrals.get(name)
            entries = np.ascontiguousarray(data[:, rows])
            sparsity = self._sparse.get(name)
            if sparsity is not None:
                # Scatter the returned structural nonzeros to their flat indices
                dense = np.zeros((entries.shape[0], sparsity.numel()))
                dense[:, sparsity.find()] = entries
                entries = dense
            values = entries
            if time_integral is not None:
                # These rows are the integrand's trajectory, not the variable's.
                values = time_integral.postfix(
                    entries.reshape(-1), solution.t, solution.all_inputs[0]
                )
            variable = pybamm.ProcessedVariableComputed(
                [model.get_processed_variable_or_event(name)],
                [self._casadi_fns[name]],
                [values],
                solution,
                time_integral=time_integral,
            )
            variable._sensitivities = (
                {}
                if sensitivities is None
                else self._variable_sensitivities(
                    name,
                    entries,
                    sensitivities[:, rows, :],
                    solution,
                    sensitivity_names,
                )
            )
            solution._variables[name] = variable

    def _checked_rows(self, data):
        """``data`` as a ``(n_t, n_rows)`` array, or a complaint about its width."""
        array = np.asarray(data)
        if array.ndim != 2 or array.shape[1] != self.n_rows:
            raise pybamm.SolverError(
                f"Output row count mismatch: expected {self.n_rows} rows (the total "
                f"flattened length of {len(self._names)} output variables) but the "
                f"solver returned an array of shape {array.shape}."
            )
        return array

    def _checked_sensitivities(self, sensitivities, n_timesteps, sensitivity_names):
        """``sensitivities`` as a ``(n_t, n_rows, n_p)`` array, or a complaint."""
        array = np.asarray(sensitivities)
        expected = (n_timesteps, self.n_rows, len(sensitivity_names))
        if array.shape != expected:
            raise pybamm.SolverError(
                f"Output sensitivity shape mismatch: expected {expected} (times, "
                f"flattened outputs, parameters) but the solver returned "
                f"{array.shape}."
            )
        return array

    def _variable_sensitivities(
        self, name, entries, block, solution, sensitivity_names
    ):
        """One variable's ``"all"`` block plus a flat vector per parameter."""
        sparsity = self._sparse.get(name)
        if sparsity is not None:
            raise pybamm.SolverError(
                f"Sensitivity of sparse variables not supported. {name} is a sparse "
                f"variable with number of non-zeros {sparsity.nnz()} and shape "
                f"{sparsity.shape}"
            )
        n_timesteps, var_len, n_params = block.shape
        all_sens = block.reshape(n_timesteps * var_len, n_params)
        time_integral = self._time_integrals.get(name)
        if time_integral is not None:
            all_sens = time_integral.postfix_sensitivities(
                name,
                entries.reshape(-1),
                solution.t,
                solution.all_inputs[0],
                sensitivity_names,
                all_sens,
            )
        return pack_sensitivity_dict(all_sens, sensitivity_names)
