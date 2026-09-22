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
    NativeObserver,
    SharedOutputLeaf,
    SharedOutputs,
    check_variable_in_solve,
    pack_sensitivity_dict,
)


class ObserverCache:
    """The compiled leaves of one model's variables, keyed by variable name.

    A name is assumed to keep one expression for the model's lifetime. Each CasADi
    leaf is built for the input layout and solution options (such as ``compile`` and
    ``cse``) of the first solve that reads it, and later solves reuse it as is.
    Native leaves compile into the graph bound by :meth:`bind_rust_model`.
    """

    def __init__(self):
        # Key -> (leaf, the expression it evaluates, time integral or None)
        self._casadi_leaves: dict[str, tuple] = {}
        self._rust_model = None
        self._clear_native()

    def _clear_native(self) -> None:
        self._native_leaves: dict[str, tuple] = {}
        # Names, in row order -> one tape evaluating the whole set
        self._native_sets: dict[tuple[str, ...], object] = {}
        self._post_sum_leaves: dict[str, object] = {}
        # Cached, as from_pybamm_var walks the whole tree
        self._time_integrals: dict[str, pybamm.ProcessedVariableTimeIntegral] = {}

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
        new._rust_model = self._rust_model
        new._native_leaves = self._native_leaves.copy()
        new._native_sets = self._native_sets.copy()
        new._post_sum_leaves = self._post_sum_leaves.copy()
        new._time_integrals = self._time_integrals.copy()
        return new

    @property
    def rust_model(self) -> object | None:
        """The :class:`pybamm.rust.CompiledModel` native leaves compile into, or None."""
        return self._rust_model

    def bind_rust_model(self, rust_model) -> None:
        """Compile native leaves into ``rust_model``'s graph from now on.

        Rebinding drops the previous model's native leaves.
        """
        if rust_model is self._rust_model:
            return
        self._rust_model = rust_model
        self._clear_native()

    def time_integral(
        self, name: str, var_pybamm: pybamm.Symbol
    ) -> pybamm.ProcessedVariableTimeIntegral | None:
        """The time integral ``var_pybamm`` reads, or None, found on first use."""
        if name not in self._time_integrals:
            self._time_integrals[name] = (
                pybamm.ProcessedVariableTimeIntegral.from_pybamm_var(
                    var_pybamm, self._rust_model.n_states
                )
            )
        return self._time_integrals[name]

    def native_leaf(
        self, name: str, var_pybamm: pybamm.Symbol
    ) -> tuple[object, pybamm.Symbol, pybamm.ProcessedVariableTimeIntegral | None]:
        """One segment's native leaf for ``var_pybamm``, compiled on first use."""
        entry = self._native_leaves.get(name)
        if entry is None:
            time_integral = self.time_integral(name, var_pybamm)
            integrand = (
                var_pybamm if time_integral is None else time_integral.sum_node.child
            )
            graph = self._rust_model.graph
            leaf = graph.compile(
                integrand.to_rust(graph), name=name, n_states=self._rust_model.n_states
            )
            entry = (leaf, integrand, time_integral)
            self._native_leaves[name] = entry
        return entry

    def native_set(self, vars_pybamm: Mapping[str, pybamm.Symbol]) -> object:
        """One tape over a set of variables, compiled on first use.

        The set shares one symbol memo, so common subexpressions are lowered once.
        """
        key = tuple(vars_pybamm)
        function = self._native_sets.get(key)
        if function is None:
            graph = self._rust_model.graph
            rust_symbols = {}
            function = graph.compile_outputs(
                {
                    name: var.to_rust(graph, rust_symbols)
                    for name, var in vars_pybamm.items()
                },
                n_states=self._rust_model.n_states,
            )
            self._native_sets[key] = function
        return function

    def post_sum_leaf(
        self, name: str, time_integral: pybamm.ProcessedVariableTimeIntegral
    ) -> object:
        """A time integral's post-sum tape, whose states are the integral's value."""
        function = self._post_sum_leaves.get(name)
        if function is None:
            n_integral = time_integral.sum_node.evaluate_for_shape().shape[0]
            graph = self._rust_model.graph
            function = graph.compile(
                time_integral.post_sum_node.to_rust(graph),
                name="post_sum",
                n_states=n_integral,
            )
            self._post_sum_leaves[name] = function
        return function

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
    caches = _native_caches(solution)
    if caches is not None and not any(
        isinstance(var, pybamm.VectorField) for var in vars_pybamm
    ):
        return _build_native_variable(solution, name, vars_pybamm, caches)
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


def build_variables(
    solution: pybamm.Solution, names: Sequence[str]
) -> dict[str, BaseProcessedVariable]:
    """The processed variables ``names`` of ``solution``, built as one set.

    On a natively read solution, non-time-integral variables share one tape per
    segment, so their common subexpressions are evaluated once.
    """
    caches = _native_caches(solution)
    if caches is None:
        return {name: build_variable(solution, name) for name in names}
    vars_pybamm = {
        name: [
            model.get_processed_variable_or_event(name) for model in solution.all_models
        ]
        for name in names
    }
    shared = [
        name
        for name in names
        if not any(
            isinstance(var, pybamm.VectorField)
            or cache.time_integral(name, var) is not None
            for var, cache in zip(vars_pybamm[name], caches, strict=True)
        )
    ]
    if len(shared) < 2:
        return {name: build_variable(solution, name) for name in names}
    variables = {
        name: build_variable(solution, name) for name in names if name not in shared
    }
    variables.update(_build_native_set(solution, shared, vars_pybamm, caches))
    return {name: variables[name] for name in names}


def _native_caches(solution: pybamm.Solution) -> list[ObserverCache] | None:
    """Each segment's cache, when every segment reads through a bound Rust model."""
    caches = []
    for model in solution.all_models:
        cache = ObserverCache.of(model)
        if model.convert_to_format != "rust" or cache.rust_model is None:
            return None
        caches.append(cache)
    return caches


def _placeholder_states(solution: pybamm.Solution) -> list[int] | None:
    """Each segment's state count when the solve stored no states, else None."""
    if not solution.variables_returned:
        return None
    return [model.len_rhs_and_alg for model in solution.all_models]


def _post_sum_leaf(cache, name, time_integral) -> object | None:
    """The post-sum tape a time integral's sensitivities need, or None."""
    if time_integral is None or time_integral.post_sum_node is None:
        return None
    return cache.post_sum_leaf(name, time_integral)


def _build_native_variable(
    solution, name, vars_pybamm, caches
) -> BaseProcessedVariable:
    """A lazily evaluated variable whose leaves are native tapes."""
    time_integral = None
    leaves = [None] * len(caches)
    for i, cache in enumerate(caches):
        check_variable_in_solve(solution, name, vars_pybamm[i])
        leaves[i], vars_pybamm[i], time_integral = cache.native_leaf(
            name, vars_pybamm[i]
        )
    observer = NativeObserver(
        leaves,
        _placeholder_states(solution),
        _post_sum_leaf(caches[0], name, time_integral),
    )
    return pybamm.process_variable(
        name, vars_pybamm, observer, solution, time_integral=time_integral
    )


def _build_native_set(
    solution, names, vars_pybamm, caches
) -> dict[str, BaseProcessedVariable]:
    """Lazily evaluated variables reading one shared native tape per segment."""
    segments = []
    for i, cache in enumerate(caches):
        for name in names:
            check_variable_in_solve(solution, name, vars_pybamm[name][i])
        function = cache.native_set({name: vars_pybamm[name][i] for name in names})
        segments.append(SharedOutputs(function))
    placeholder_states = _placeholder_states(solution)
    return {
        name: pybamm.process_variable(
            name,
            vars_pybamm[name],
            NativeObserver(
                [SharedOutputLeaf(shared, name) for shared in segments],
                placeholder_states,
            ),
            solution,
        )
        for name in names
    }


class OutputAssembly:
    """The row layout of an ``output_variables`` payload, and its attachment.

    A solve run with ``output_variables`` propagates no state trajectory: the
    solver evaluates the requested variables itself and returns one row per
    non-zero of each variable's CasADi function (per value, for :meth:`dense`),
    in variable order, so a vector variable spans several consecutive rows.

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
        names = tuple(names)
        self._casadi_fns = {name: casadi_fns[name] for name in names}
        lens = []
        # name -> sparsity of the variables whose rows are sparse
        self._sparse = {}
        for name in names:
            sparsity = self._casadi_fns[name].sparsity_out(0)
            lens.append(sparsity.nnz())
            if sparsity.nnz() != sparsity.numel():
                self._sparse[name] = sparsity
        self._set_layout(names, lens, time_integrals)

    @classmethod
    def dense(
        cls,
        names: Sequence[str],
        lens: Sequence[int],
        *,
        time_integrals: Mapping[str, pybamm.ProcessedVariableTimeIntegral]
        | None = None,
    ) -> OutputAssembly:
        """The layout of a payload that carries every value of each variable."""
        assembly = cls.__new__(cls)
        assembly._casadi_fns = {}
        assembly._sparse = {}
        assembly._set_layout(tuple(names), lens, time_integrals)
        return assembly

    def _set_layout(self, names, lens, time_integrals) -> None:
        if len(names) != len(lens):
            raise pybamm.SolverError(
                f"Output layout mismatch: {len(names)} output variables but "
                f"{len(lens)} row lengths."
            )
        self._names = names
        self._time_integrals = dict(time_integrals or {})
        offsets = list(accumulate((int(length) for length in lens), initial=0))
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
                [self._casadi_fns.get(name)],
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
