"""Observation backends: how a :class:`pybamm.Solution` reads its variables.

An :class:`ObservationBackend` turns a variable name into a processed variable
over a run of a Solution's sub-solutions, its *segments*. A *leaf* is one
segment's compiled form of one variable, so a variable spanning ``n``
sub-solutions has ``n`` leaves.

:class:`OutputAssembly` is the ``output_variables`` counterpart: the solver has
already computed those variables, so it owns the payload's row layout and
populates the Solution eagerly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class ObservationBackend(ABC):
    """How a Solution turns a variable name into something it can evaluate.

    A backend covers an ordered run of a Solution's sub-solutions, one
    *segment* each. ``backend[key]`` indexes *segments*, matching the slice the
    Solution applies to its sub-solutions; :func:`join_observations` combines
    the backends of consecutive Solutions.
    """

    @abstractmethod
    def __getitem__(self, key: slice) -> ObservationBackend:
        """This backend restricted to a slice of the Solution's segments."""

    @abstractmethod
    def build_variable(
        self, solution: pybamm.Solution, name: str
    ) -> BaseProcessedVariable:
        """The processed variable for ``name``, ready to evaluate.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution being observed; supplies the trajectories, models and
            inputs. Its segments are 1:1 with this backend's.
        name : str
            Variable name, as registered on the models.

        Returns
        -------
        :class:`pybamm.solvers.base_processed_variable.BaseProcessedVariable`
            Evaluable over this backend's whole run of segments.
        """


class CasadiObservation(ObservationBackend):
    """Variables converted to CasADi and evaluated by the IDAKLU kernels.

    Stateless: every per-segment artifact it needs is reached through the
    Solution it is handed, so any two instances are interchangeable and one,
    :data:`CASADI_OBSERVATION`, covers any run of segments.
    """

    def __getitem__(self, key):
        return self

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return hash(type(self))

    def build_variable(self, solution, name):
        time_integral = None
        pybamm.logger.debug(f"Post-processing {name}")

        # Iterate through all models, some may be in the list several times and
        # therefore only get set up once
        vars_pybamm = [
            model.get_processed_variable_or_event(name) for model in solution.all_models
        ]
        vars_casadi = [None] * len(solution.all_models)
        for i, (model, ys, inputs) in enumerate(
            zip(solution.all_models, solution.all_ys, solution.all_inputs, strict=True)
        ):
            _var_pybamm = vars_pybamm[i]
            check_variable_in_solve(solution, name, _var_pybamm)
            if isinstance(_var_pybamm, pybamm.VectorField):
                comp_casadi = []
                for k, comp in enumerate(_var_pybamm.components):
                    cc, _, _ = self._model_leaf(
                        solution,
                        model,
                        comp,
                        inputs=inputs,
                        ys_shape=ys.shape,
                        time_integral=None,
                        cache_key=f"{name}[{k}]",
                    )
                    comp_casadi.append(cc)
                vars_casadi[i] = comp_casadi
            else:
                var_casadi, var_pybamm, time_integral = self._model_leaf(
                    solution,
                    model,
                    _var_pybamm,
                    inputs=inputs,
                    ys_shape=ys.shape,
                    time_integral=time_integral,
                    cache_key=name,
                )
                vars_pybamm[i] = var_pybamm
                vars_casadi[i] = var_casadi
        return pybamm.process_variable(
            name,
            vars_pybamm,
            CasadiObserver(vars_casadi),
            solution,
            time_integral=time_integral,
        )

    @staticmethod
    def _model_leaf(
        solution,
        model,
        var_pybamm,
        time_integral,
        inputs,
        ys_shape,
        cache_key,
    ):
        """One model's CasADi leaf, memoised on the model unless time-integrated."""
        _var_casadi = model._variables_casadi.get(cache_key)
        if _var_casadi is not None:
            return _var_casadi, var_pybamm, time_integral

        var_casadi, var_pybamm, time_integral = solution._convert_to_casadi(
            var_pybamm, inputs, ys_shape
        )

        # Only cache if it's not a time integral
        if time_integral is None:
            model._variables_casadi[cache_key] = var_casadi
        return var_casadi, var_pybamm, time_integral


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
        # name -> (non-zeros, shape) of the variables whose rows are sparse
        self._sparse = {}
        for name in self._names:
            evaluated = self._casadi_fns[name](0.0, 0.0, 0.0)
            sparsity = evaluated.sparsity()
            lens.append(sparsity.nnz())
            if sparsity.nnz() != sparsity.numel():
                self._sparse[name] = (sparsity.nnz(), evaluated.shape)
        offsets = list(accumulate(lens, initial=0))
        self._n_rows = offsets[-1]
        # The layout itself: one slice of payload rows per output variable.
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
            values = np.ascontiguousarray(data[:, rows])
            if time_integral is not None:
                # These rows are the integrand's trajectory, not the variable's.
                values = time_integral.postfix(
                    values.reshape(-1), solution.t, solution.all_inputs[0]
                )
            variable = pybamm.ProcessedVariableComputed(
                [model.get_processed_variable_or_event(name)],
                [self._casadi_fns[name]],
                [values],
                solution,
                time_indep=time_integral is not None,
            )
            variable._sensitivities = (
                {}
                if sensitivities is None
                else self._variable_sensitivities(
                    name, values, sensitivities[:, rows, :], solution, sensitivity_names
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

    def _variable_sensitivities(self, name, values, block, solution, sensitivity_names):
        """One variable's ``"all"`` block plus a flat vector per parameter."""
        if name in self._sparse:
            nnz, shape = self._sparse[name]
            raise pybamm.SolverError(
                f"Sensitivity of sparse variables not supported. {name} is a sparse "
                f"variable with number of non-zeros {nnz} and shape {shape}"
            )
        n_timesteps, var_len, n_params = block.shape
        all_sens = block.reshape(n_timesteps * var_len, n_params)
        time_integral = self._time_integrals.get(name)
        if time_integral is not None:
            all_sens = time_integral.postfix_sensitivities(
                name, values, solution.t, solution.all_inputs[0], all_sens
            )
        return pack_sensitivity_dict(all_sens, sensitivity_names)


def join_observations(backends: Sequence[ObservationBackend]) -> ObservationBackend:
    """One backend covering consecutive Solutions' segments, in order.

    Parameters
    ----------
    backends : list of :class:`ObservationBackend`
        The backend of each Solution being joined, in order.

    Returns
    -------
    :class:`ObservationBackend`
        The backend every Solution shares.

    Raises
    ------
    :class:`pybamm.SolverError`
        If the Solutions do not all share one backend.
    """
    first, *rest = backends
    for backend in rest:
        if backend != first:
            raise pybamm.SolverError(
                "Cannot join solutions read through different observation "
                f"backends: {type(first).__name__} and {type(backend).__name__}."
            )
    return first


#: The backend a Solution carries by default.
CASADI_OBSERVATION = CasadiObservation()
