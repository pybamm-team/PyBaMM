"""Observation backends: how a :class:`pybamm.Solution` reads its variables.

A ``Solution`` holds exactly one :class:`ObservationBackend`, chosen when the
solve finishes. The backend owns everything needed to turn a variable name into
a ready-to-evaluate processed variable — CasADi conversion for the default, or
the retained Rust graph, its compiled tapes and their cache for the native one.
Derived solutions (``first_state``, ``last_state``, ``copy``, ``__add__``,
``from_sub_solutions``) carry that one field, so nothing re-decides which
backend is in play.

An ``output_variables`` solve reads its variables the other way round: the
solver computed them already and hands back a concatenated payload instead of a
state trajectory. :class:`OutputAssembly` is that path's counterpart to a
backend — it owns the payload's row layout and populates the Solution eagerly,
for whichever solver produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import accumulate, pairwise

import numpy as np

import pybamm
from pybamm.solvers.variable_observer import (
    CasadiObserver,
    NativeObserver,
    check_variable_in_solve,
    native_sensitivities,
)


class ObservationBackend(ABC):
    """How a Solution lowers a variable name into evaluable leaves.

    Backends are immutable values covering an ordered run of a Solution's
    sub-solutions. ``backend[key]`` restricts them to a slice of that run and
    :func:`join_observations` concatenates runs, so a derived Solution copies
    one field instead of a bundle of parallel ones.
    """

    @abstractmethod
    def __getitem__(self, key: slice) -> ObservationBackend:
        """This backend restricted to a slice of the Solution's segments."""

    @abstractmethod
    def build_variable(self, solution, name):
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
        """


class CasadiObservation(ObservationBackend):
    """The default backend: variables are converted to CasADi and read by IDAKLU.

    Stateless -- every per-segment artifact it needs is reached through the
    Solution it is handed -- so :data:`CASADI_OBSERVATION` is shared by every
    Solution that has not been given a native backend.
    """

    def __getitem__(self, key):
        return self

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


class NativeObservation(ObservationBackend):
    """Observation through a compiled Rust model's retained expression graph.

    Observed variables are lowered as new roots into the graph the solve was
    built from and compiled to tapes, so no CasADi conversion happens. The
    graph never leaves this class.

    Abstract: a solve that stores state derivatives can be read at arbitrary
    times and wants :class:`NativeInterpolatingObservation`, while one read
    only on its own grid wants :class:`NativeComputedObservation`. Which of
    the two a solver constructs is what picks the kind of processed variable
    its solutions hand back.

    Parameters
    ----------
    models : list of :class:`pybamm.rust.CompiledModel`
        One model per sub-solution, in solve order.
    cache : dict, optional
        Cache of compiled tapes and time-integral analyses, 1:1 with the
        models. Sharing the solver's dict lets repeated solves reuse tapes
        instead of growing the retained graph; omitting it starts a fresh one.
    """

    def __init__(self, models, *, cache=None):
        self._models = list(models)
        self._cache = {} if cache is None else cache

    @classmethod
    def uniform(cls, model, n_segments, **kwargs) -> NativeObservation:
        """One model observing every segment, which is what a single solve produces."""
        return cls([model] * n_segments, **kwargs)

    @classmethod
    def _adopting(cls, models, *, cache) -> NativeObservation:
        """Take ownership of an already-fresh list instead of copying it again."""
        backend = cls.__new__(cls)
        backend._models = models
        backend._cache = cache
        return backend

    @property
    def n_segments(self):
        return len(self._models)

    @property
    def compile_cache(self):
        """The shared cache of compiled tapes and time-integral analyses."""
        return self._cache

    @property
    def segment_models(self):
        """The per-segment compiled models, in solve order."""
        return self._models

    @property
    def primary_model(self):
        """The model owning the graph new observation roots are lowered into."""
        return self._models[0]

    def __getitem__(self, key):
        return self._adopting(self._models[key], cache=self._cache)

    def _segment_leaf(self, solution, name, model, rust_model, nstates):
        """One segment's ``(pybamm variable, time integral, compiled leaf)``.

        A time-integrated variable contributes its integrand, so the leaf is
        what the postfix sum consumes rather than the variable itself.
        """
        var_pybamm = model.get_processed_variable_or_event(name)
        check_variable_in_solve(solution, name, var_pybamm)
        time_integral = self._time_integral(name, model, var_pybamm, nstates)
        integrand = (
            time_integral.sum_node.child if time_integral is not None else var_pybamm
        )
        return var_pybamm, time_integral, self._leaf(name, integrand, rust_model)

    def _time_integral(self, name, model, var_pybamm, nstates):
        """Time-integral classification of `name`, memoised across solves.

        ``from_pybamm_var`` walks the variable's full expression tree, which
        costs more than a small model's solve, so both build paths share this
        memo. ``None`` results are stored too, hence key membership decides a hit.
        """
        ti_key = ("__time_integral__", name, id(model), nstates)
        if ti_key in self._cache:
            return self._cache[ti_key]
        time_integral = pybamm.ProcessedVariableTimeIntegral.from_pybamm_var(
            var_pybamm, nstates
        )
        self._cache[ti_key] = time_integral
        return time_integral

    def _leaf(self, name, integrand, rust_model):
        """Compile `integrand` once into the retained graph, cached by name+model."""
        # model id keys the cache so distinct models don't alias; models are
        # retained for the backend's lifetime, so the id cannot be reused.
        cache_key = (name, id(rust_model))
        fn = self._cache.get(cache_key)
        if fn is None:
            graph = rust_model.graph
            rust_expr = integrand.to_rust(graph)
            fn = graph.compile(rust_expr, name=name, n_states=rust_model.n_states)
            self._cache[cache_key] = fn
        return fn

    def _post_sum_leaf(self, name, time_integral, n_inner):
        """Compile a time-integral's ``post_sum_node`` against the retained graph.

        The post-sum node's synthetic StateVector input is the integrated inner
        value, so the compiled "state" is that ``n_inner`` vector (the eval
        point is the postfix VALUE, not the full state trajectory).
        """
        # Key on the variable name, not id(post_sum_node): the time_integral is a
        # transient local whose id can be reused and alias the wrong compiled fn.
        cache_key = ("__post_sum__", name, n_inner)
        fn = self._cache.get(cache_key)
        if fn is None:
            graph = self.primary_model.graph
            rust_expr = time_integral.post_sum_node.to_rust(graph)
            fn = graph.compile(rust_expr, name="post_sum", n_states=n_inner)
            self._cache[cache_key] = fn
        return fn

    def postfix_sensitivities(
        self,
        times,
        name,
        time_integral,
        postfix_value,
        inner_sens,
        inputs,
        sens_names,
    ):
        """``dvar/dp`` for a time-integral variable.

        Integrates the inner-variable sensitivities over time, then applies the
        post-sum chain rule ``dpost_dy @ s_integral + dpost_dp`` using a
        Rust-lowered jacobian of ``post_sum_node`` evaluated at the postfix value.

        Parameters
        ----------
        times : numpy.ndarray
            The solution's time points, shape ``(n_t,)``.
        name : str
            Variable name; keys the compiled post-sum cache.
        time_integral : pybamm.ProcessedVariableTimeIntegral
            Time-integral descriptor for the variable.
        postfix_value : numpy.ndarray
            Postfix value, shape ``(n_inner,)``; the eval point for the post-sum
            jacobians.
        inner_sens : numpy.ndarray
            Inner-variable sensitivities, shape ``(n_t * n_inner, n_p)``,
            time-outer/inner-inner.
        inputs : dict
            Input-parameter values for the solve.
        sens_names : list[str]
            Sensitivity-parameter names; column order of ``inner_sens`` and the
            returned array.

        Returns
        -------
        numpy.ndarray
            Variable sensitivities ``dvar/dp``, shape ``(n_inner, n_p)``.

        Raises
        ------
        pybamm.SolverError
            If a discrete-time sum's times do not match the solution's.
        """
        time_integral.check_discrete_times(name, times)

        # Integrate the inner sensitivities over time, shape (n_inner, n_p).
        s_integral = time_integral.postfix_sum(inner_sens, times)
        if time_integral.post_sum_node is None:
            return s_integral

        n_inner = int(np.asarray(time_integral.sum_node.evaluate_for_shape()).shape[0])
        post_fn = self._post_sum_leaf(name, time_integral, n_inner)

        # Evaluate dpost_dy / dpost_dp ONCE at the postfix VALUE (the CasADi
        # `entries` eval point), NOT the raw trajectory.
        entries = np.ascontiguousarray(
            np.asarray(postfix_value, dtype=np.float64).ravel()
        )
        p_stacked = post_fn.pack(inputs)
        dpost_dy = post_fn.jacobian("y")(0.0, entries, p_stacked).toarray()
        dpost_dp_full = post_fn.jacobian("p")(0.0, entries, p_stacked).toarray()
        # Slice/reorder dpost_dp columns to the sensitivity-parameter order.
        input_names = list(post_fn.input_names)
        sens_idx = [input_names.index(param) for param in sens_names]
        dpost_dp = dpost_dp_full[:, sens_idx]

        # dpost_dy @ s_integral + dpost_dp, mirroring postfix_sensitivities exactly.
        return dpost_dy @ s_integral + dpost_dp


class NativeInterpolatingObservation(NativeObservation):
    """Native observation of a solve that stored state derivatives.

    Its leaves can be evaluated at arbitrary times, so variables are built as
    lazily-evaluated :class:`pybamm.ProcessedVariable`s that cubic-Hermite
    reconstruct the state off the solution's own grid. Hermite is a no-op when
    a particular solution carries no ``yps``, which matches the CasADi path.
    """

    def build_variable(self, solution, name):
        """A lazily-evaluated ProcessedVariable whose leaves are compiled tapes."""
        vars_pybamm = []
        leaves = []
        time_integral = None
        nstates = solution.all_ys[0].shape[0]
        for model, rust_model in zip(solution.all_models, self._models, strict=True):
            var_pybamm, ti, leaf = self._segment_leaf(
                solution, name, model, rust_model, nstates
            )
            leaves.append(leaf)
            vars_pybamm.append(var_pybamm)
            time_integral = ti if ti is not None else time_integral
        placeholder_states = (
            [model.len_rhs_and_alg for model in solution.all_models]
            if solution.variables_returned
            else None
        )
        return pybamm.process_variable(
            name,
            vars_pybamm,
            NativeObserver(leaves, self, placeholder_states),
            solution,
            time_integral=time_integral,
        )


class NativeComputedObservation(NativeObservation):
    """Native observation of a solve read only on its own time points.

    Variables are evaluated eagerly into grid-aligned
    :class:`pybamm.ProcessedVariableComputed`s, which interpolate in time
    themselves rather than re-entering the compiled leaves.
    """

    def build_variable(self, solution, name):
        """An eager ProcessedVariableComputed evaluated on the solution's own grid."""
        base_variables = []
        base_variables_data = []
        # time integrals: accumulate integrands, postfix once after the loop
        integrand_segments = []
        first_time_integral = None
        first_var_pybamm = None
        # per-segment (compiled fn, ts, ys, inputs) for the sensitivity chain rule
        sens_segments = []
        for model, ts, ys, inputs, rust_model in zip(
            solution.all_models,
            solution.all_ts,
            solution.all_ys,
            solution.all_inputs,
            self._models,
            strict=True,
        ):
            if solution.variables_returned:
                # No states stored; the variable is state-free, so evaluate on a
                # shaped placeholder trajectory.
                ys = np.zeros((model.len_rhs_and_alg, ts.size))
            var_pybamm, time_integral, leaf = self._segment_leaf(
                solution, name, model, rust_model, ys.shape[0]
            )
            sens_segments.append((leaf, ts, ys, inputs))
            # returns (output, n_times) F-contiguous; .T is a zero-copy time-major view
            data = np.asarray(leaf.eval_trajectory(ts, ys, inputs)).T
            if time_integral is not None:
                integrand_segments.append(data.reshape(-1))
                if first_time_integral is None:
                    first_time_integral = time_integral
                    first_var_pybamm = var_pybamm
            else:
                base_variables.append(var_pybamm)
                base_variables_data.append(data)

        the_integral = None
        if first_time_integral is not None:
            # integrate once over the full trajectory (integrand lines up 1:1 with t)
            full_integrand = np.concatenate(integrand_segments)
            the_integral = first_time_integral.postfix(
                full_integrand, solution.t, solution.all_inputs[0]
            )
            base_variables = [first_var_pybamm]
            base_variables_data = [the_integral]

        var = pybamm.ProcessedVariableComputed(
            base_variables,
            [None] * len(base_variables),
            base_variables_data,
            solution,
            time_indep=first_time_integral is not None,
        )
        var._sensitivities = self._computed_sensitivities(
            solution, name, sens_segments, first_time_integral, the_integral
        )
        return var

    def _computed_sensitivities(
        self, solution, name, sens_segments, time_integral, postfix_value
    ):
        """Forward sensitivities for the eager path, or ``{}`` when there are none."""
        if not solution.has_sensitivities():
            return {}
        return native_sensitivities(
            sens_segments,
            solution._all_sensitivities["all"],
            solution.sensitivity_names,
            time_integral=time_integral,
            postfix=lambda inner, sens_names: self.postfix_sensitivities(
                solution.t,
                name,
                time_integral,
                postfix_value,
                inner,
                solution.all_inputs[0],
                sens_names,
            ),
        )


class OutputAssembly:
    """Attaching a solver's concatenated outputs-only payload to a Solution.

    A solve run with ``output_variables`` propagates no state trajectory: the
    solver evaluates the requested variables itself and returns them as one row
    per *flattened output component*, in variable order, so a vector variable
    spans ``lens[i]`` consecutive rows. Slicing that payload by variable ordinal
    instead would drop a vector variable's tail and shift every variable after
    it. Every solver that can produce such a payload assembles it through this
    one object, so the layout — and the time-integral postfix riding on it —
    lives in one place.

    Parameters
    ----------
    names : list of str
        The output variables, in row order.
    lens : list of int
        Flattened component count per variable, so ``names[i]`` owns rows
        ``sum(lens[:i])`` up to ``sum(lens[: i + 1])``.
    time_integrals : dict, optional
        Map of name to :class:`pybamm.ProcessedVariableTimeIntegral` for the
        outputs whose rows carry an integrand rather than the variable itself.
        Their postfix sum runs here, once the trajectory is in hand.
    casadi_fns : dict, optional
        Map of name to the CasADi function the rows were evaluated by, on the
        CasADi path only. It carries the sparsity a sparse variable is unrolled
        through; native rows are dense and pass nothing.

    Raises
    ------
    :class:`pybamm.SolverError`
        If ``names`` and ``lens`` are not 1:1.
    """

    def __init__(self, names, lens, *, time_integrals=None, casadi_fns=None):
        self._names = tuple(names)
        self._lens = tuple(int(length) for length in lens)
        if len(self._names) != len(self._lens):
            raise pybamm.SolverError(
                f"Output layout mismatch: {len(self._names)} output variables but "
                f"{len(self._lens)} row lengths."
            )
        self._time_integrals = dict(time_integrals or {})
        self._casadi_fns = dict(casadi_fns or {})
        offsets = list(accumulate(self._lens, initial=0))
        self._n_rows = offsets[-1]
        # The layout itself: one slice of payload rows per output variable.
        self._rows = tuple(slice(start, end) for start, end in pairwise(offsets))

    @classmethod
    def from_casadi(cls, names, casadi_fns, *, time_integrals=None):
        """The layout of a CasADi-evaluated payload, whose rows are its non-zeros.

        Parameters
        ----------
        names : list of str
            The output variables, in row order.
        casadi_fns : dict
            Map of name to the CasADi function evaluating it.
        time_integrals : dict, optional
            As for the constructor.

        Returns
        -------
        OutputAssembly
        """
        lens = [casadi_fns[name](0.0, 0.0, 0.0).sparsity().nnz() for name in names]
        return cls(names, lens, time_integrals=time_integrals, casadi_fns=casadi_fns)

    @property
    def names(self) -> tuple[str, ...]:
        """The output variables, in row order."""
        return self._names

    @property
    def lens(self) -> tuple[int, ...]:
        """Flattened component count per output variable."""
        return self._lens

    @property
    def n_rows(self) -> int:
        """Rows in one time point of the payload."""
        return self._n_rows

    def attach(self, solution, data, *, sensitivities=None, sensitivity_names=()):
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
            rather than lazily recomputed — an outputs-only solve keeps no state
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
                [self._casadi_fns.get(name)],
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

    def stack_parameter_blocks(self, blocks, n_timesteps, sensitivity_names):
        """``(n_t, n_rows, n_p)`` sensitivities from one flat block per parameter.

        Parameters
        ----------
        blocks : sequence of array-like
            One block per sensitivity parameter, each ``n_t * n_rows`` values in
            time-outer/output-inner order.
        n_timesteps : int
            Number of solution time points.
        sensitivity_names : list of str
            Sensitivity-parameter names, in ``blocks`` order.

        Returns
        -------
        :class:`numpy.ndarray`
            Sensitivities laid out for :meth:`attach`.

        Raises
        ------
        :class:`pybamm.SolverError`
            If there is not one block per named parameter.
        """
        if len(blocks) != len(sensitivity_names):
            raise pybamm.SolverError(
                f"Sensitivity block count mismatch: expected "
                f"{len(sensitivity_names)} parameter blocks (from "
                f"model.calculate_sensitivities) but the solver returned "
                f"{len(blocks)}."
            )
        return np.stack(
            [np.asarray(block).reshape(n_timesteps, self.n_rows) for block in blocks],
            axis=-1,
        )

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
        self._reject_sparse(name)
        n_timesteps, var_len, n_params = block.shape
        all_sens = block.reshape(n_timesteps * var_len, n_params)
        time_integral = self._time_integrals.get(name)
        if time_integral is not None:
            all_sens = time_integral.postfix_sensitivities(
                name, values, solution.t, solution.all_inputs[0], all_sens
            )
        sensitivities = {"all": all_sens}
        for i, param in enumerate(sensitivity_names):
            sensitivities[param] = all_sens[:, i : i + 1].reshape(-1)
        return sensitivities

    def _reject_sparse(self, name):
        """A CasADi-sparse variable's rows unroll, so its sensitivities cannot."""
        casadi_fn = self._casadi_fns.get(name)
        if casadi_fn is None:
            return
        evaluated = casadi_fn(0.0, 0.0, 0.0)
        sparsity = evaluated.sparsity()
        if sparsity.nnz() == sparsity.numel():
            return
        raise pybamm.SolverError(
            f"Sensitivity of sparse variables not supported. {name} is a sparse "
            f"variable with number of non-zeros {sparsity.nnz()} and shape "
            f"{evaluated.shape}"
        )


def join_observations(runs) -> ObservationBackend:
    """One backend covering ``runs``' segments, concatenated in order.

    Parameters
    ----------
    runs : list[tuple[ObservationBackend, int]]
        One ``(backend, n_segments)`` pair per Solution being joined, in order.
        The count comes from the caller because only a native backend holds
        per-segment state; the default CasADi one covers any run.

    Returns
    -------
    ObservationBackend
        :data:`CASADI_OBSERVATION` when no run is native, else a native backend
        spanning every segment: interpolating if any native run was, so a join
        never narrows how a solution can be read. A native run wins over a
        CasADi one, whose segments are observed by the first native model --
        the behaviour from before this seam existed, and sound only because the
        runs an experiment stitches share one discretised model. Nothing checks
        that, so do not rely on it for a genuine mix.
    """
    natives = [backend for backend, _ in runs if isinstance(backend, NativeObservation)]
    if not natives:
        return CASADI_OBSERVATION
    stand_in = natives[0].primary_model
    models = []
    for backend, n_segments in runs:
        if isinstance(backend, NativeObservation):
            models.extend(backend.segment_models)
        else:
            models.extend([stand_in] * n_segments)
    joined = (
        NativeInterpolatingObservation
        if any(isinstance(b, NativeInterpolatingObservation) for b in natives)
        else NativeComputedObservation
    )
    return joined._adopting(
        models, cache=_join_caches([backend.compile_cache for backend in natives])
    )


def _join_caches(caches):
    """The compile cache for a joined backend, earlier runs winning on collision.

    Identity is preserved when every run already shares one dict, so a merged
    solution keeps compiling into the solver's cache instead of into a copy.
    """
    if all(cache is caches[0] for cache in caches):
        return caches[0]
    merged = {}
    for cache in reversed(caches):
        merged.update(cache)
    return merged


#: Shared by every Solution that has not been given a native backend.
CASADI_OBSERVATION = CasadiObservation()
