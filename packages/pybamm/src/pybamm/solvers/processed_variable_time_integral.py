from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import casadi
import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid

import pybamm

# Tolerances for matching a discrete time to a solution time
_TIME_RTOL = 1e-9
_TIME_ATOL = 1e-10


@dataclass
class ProcessedVariableTimeIntegral:
    """
    How a variable holding a time integral or discrete time sum is evaluated.

    A continuous integral applies the trapezoidal rule over the solution times; a
    discrete sum adds its summand at its discrete times within the solution's span.
    A discrete sum leaves out its discrete times after the solution ends, such as
    at an early event, and :meth:`warn_times_after_end` logs how many.

    Parameters
    ----------
    method : str
        ``"continuous"`` for a :class:`pybamm.ExplicitTimeIntegral`, or
        ``"discrete"`` for a :class:`pybamm.DiscreteTimeSum`.
    sum_node : :class:`pybamm.Symbol`
        The time integral or discrete sum node.
    initial_condition : :class:`pybamm.Symbol`, numpy.ndarray or float
        The value of the integral at the first solution time. A symbol is
        evaluated with each solution's inputs.
    discrete_times : numpy.ndarray or None
        The times a discrete sum adds up its summand at.
    post_sum_node : :class:`pybamm.Symbol`, optional
        The variable as an expression of the integral, which it reads as a state
        vector. None if the variable is the integral itself.
    post_sum : :class:`casadi.Function`, optional
        ``post_sum_node`` converted to CasADi, taking the stacked inputs.
    """

    method: Literal["discrete", "continuous"]
    sum_node: pybamm.Symbol
    initial_condition: pybamm.Symbol | npt.NDArray[np.float64] | float
    discrete_times: npt.NDArray[np.float64] | None
    post_sum_node: pybamm.Symbol | None = None
    post_sum: casadi.Function | None = None

    def _sum_over_time(self, entries, t_pts) -> np.ndarray:
        """Sum or integrate time-major ``entries`` over the time points."""
        if self.method == "discrete":
            return np.sum(entries, axis=0)
        return trapezoid(entries, t_pts, axis=0)

    def _time_major(self, array, n_times, trailing=()) -> np.ndarray:
        """``array`` as ``(n_times, n_entries, *trailing)``."""
        array = np.asarray(array, dtype=float)
        if array.size == 0:
            # No samples, so the entry count comes from the integrand
            return np.zeros((n_times, int(self.sum_node.size), *trailing))
        return array.reshape(n_times, -1, *trailing)

    def sample_times(self, t_pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        The discrete times a discrete sum samples its summand at.

        Parameters
        ----------
        t_pts : numpy.ndarray
            The solution times, ascending.

        Returns
        -------
        numpy.ndarray
            The discrete times within ``[t_pts[0], t_pts[-1]]``.
        """
        times = np.asarray(self.discrete_times, dtype=float)
        start, end = t_pts[0], t_pts[-1]
        in_span = (
            ((times >= start) & (times <= end))
            | np.isclose(times, start, rtol=_TIME_RTOL, atol=_TIME_ATOL)
            | np.isclose(times, end, rtol=_TIME_RTOL, atol=_TIME_ATOL)
        )
        return np.clip(times[in_span], start, end)

    def sample_indices(
        self, t_pts: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.intp] | None:
        """
        The indices of the solution times the integrand is sampled at.

        Parameters
        ----------
        t_pts : numpy.ndarray
            The solution times, ascending.

        Returns
        -------
        numpy.ndarray or None
            Every index for a continuous integral. For a discrete sum, one index
            per :meth:`sample_times`, or None if one of them is not in ``t_pts``.
        """
        if self.method == "continuous":
            return np.arange(len(t_pts))
        nearest, matched = _nearest_times(t_pts, self.sample_times(t_pts))
        return nearest if np.all(matched) else None

    def checked_sample_indices(
        self, var_name: str, t_pts: npt.NDArray[np.float64], reason: str
    ) -> npt.NDArray[np.intp]:
        """
        As :meth:`sample_indices`, but raise if a discrete time is not in ``t_pts``.

        Parameters
        ----------
        var_name : str
            The name of the variable.
        t_pts : numpy.ndarray
            The solution (or output) times, ascending.
        reason : str
            Why the summand cannot be interpolated to discrete times not in
            ``t_pts``.

        Returns
        -------
        numpy.ndarray
            One index per sample time.

        Raises
        ------
        :class:`pybamm.SolverError`
            If a discrete time within the span of ``t_pts`` is not one of them.
        """
        indices = self.sample_indices(t_pts)
        if indices is None:
            times = self.sample_times(t_pts)
            missing = times[~_nearest_times(t_pts, times)[1]]
            raise pybamm.SolverError(
                f'Discrete-time-sum variable "{var_name}": {len(missing)} of its '
                f"discrete times (first: {missing[:3]}) are not solution times, and "
                f"{reason}. Include the discrete times in `t_interp` (or `t_eval`)."
            )
        return indices

    def warn_times_after_end(self, var_name: str, t_end: float) -> None:
        """
        Log a warning if a discrete sum has discrete times after ``t_end``.

        The sum leaves these times out, so a solution that ends early, such as at
        an event, sums fewer terms.

        Parameters
        ----------
        var_name : str
            The name of the variable.
        t_end : float
            The time the solution ends at.
        """
        if self.method == "continuous":
            return
        times = np.asarray(self.discrete_times, dtype=float)
        after_end = (times > t_end) & ~np.isclose(
            times, t_end, rtol=_TIME_RTOL, atol=_TIME_ATOL
        )
        n_after_end = np.count_nonzero(after_end)
        if n_after_end:
            pybamm.logger.warning(
                f'Discrete-time-sum variable "{var_name}": {n_after_end} of its '
                f"{times.size} discrete times are after the solution ends at "
                f"t = {t_end:g}, so the sum leaves them out."
            )

    def check_output_times(
        self,
        var_name: str,
        t_eval: npt.NDArray[np.float64],
        t_interp: npt.NDArray[np.float64],
    ) -> None:
        """
        Check an ``output_variables`` solve outputs at every discrete time in its span.

        Parameters
        ----------
        var_name : str
            The name of the variable.
        t_eval : numpy.ndarray
            The times the solve stops at.
        t_interp : numpy.ndarray
            The times the solve interpolates at, possibly empty.

        Raises
        ------
        :class:`pybamm.SolverError`
            If a discrete time within ``t_eval``'s span is in neither array.
        """
        if self.method == "continuous":
            return
        t_eval = np.asarray(t_eval, dtype=float)
        t_interp = np.asarray(t_interp, dtype=float)
        t_interp = t_interp[(t_interp >= t_eval[0]) & (t_interp <= t_eval[-1])]
        self.checked_sample_indices(
            var_name,
            np.union1d(t_eval, t_interp),
            "an output_variables solve keeps no states to interpolate",
        )

    def initial_condition_value(self, inputs: dict) -> npt.NDArray[np.float64]:
        """
        The initial condition for one input set, flattened.

        Parameters
        ----------
        inputs : dict
            The input parameters.

        Returns
        -------
        numpy.ndarray
        """
        initial_condition = self.initial_condition
        if isinstance(initial_condition, pybamm.Symbol):
            initial_condition = initial_condition.evaluate(inputs=inputs)
        return np.asarray(initial_condition, dtype=float).reshape(-1)

    def initial_condition_sensitivities(
        self, inputs: dict, sensitivity_names: list[str]
    ) -> npt.NDArray[np.float64]:
        """
        The derivatives of the initial condition with respect to the inputs.

        Parameters
        ----------
        inputs : dict
            The input parameters.
        sensitivity_names : list of str
            The inputs to differentiate with respect to, in column order.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_initial_condition, n_p)``; zero for a constant.
        """
        initial_condition = self.initial_condition
        if not (
            isinstance(initial_condition, pybamm.Symbol)
            and initial_condition.has_symbol_of_classes(pybamm.InputParameter)
        ):
            size = self.initial_condition_value(inputs).size
            n_params = sum(_input_size(inputs[name]) for name in sensitivity_names)
            return np.zeros((size, n_params))
        p_casadi, p_stacked, p_values = _symbolic_inputs(inputs, sensitivity_names)
        initial_condition_casadi = initial_condition.to_casadi(
            casadi.MX.sym("t"), casadi.MX.sym("y", 0), inputs={**inputs, **p_casadi}
        )
        jacobian = casadi.Function(
            "dinitial_condition_dp",
            [p_stacked],
            [casadi.jacobian(initial_condition_casadi, p_stacked)],
        )
        return jacobian(p_values).full()

    def postfix_sum(self, entries, t_pts, inputs) -> np.ndarray:
        """
        The integral or sum of the integrand, plus the initial condition.

        Parameters
        ----------
        entries : array-like
            The integrand at each of ``t_pts``, time-major.
        t_pts : numpy.ndarray
            The sample times.
        inputs : dict
            The input parameters.

        Returns
        -------
        numpy.ndarray
            One value per integrand entry.
        """
        entries = self._time_major(entries, len(t_pts))
        return self._sum_over_time(entries, t_pts) + self.initial_condition_value(
            inputs
        )

    def postfix(self, entries, t_pts, inputs) -> np.ndarray:
        """
        The variable's value from its integrand at the sample times.

        Parameters
        ----------
        entries : array-like
            The integrand at each of ``t_pts``, time-major.
        t_pts : numpy.ndarray
            The sample times.
        inputs : dict
            The input parameters.

        Returns
        -------
        numpy.ndarray
            The value, flattened.
        """
        the_integral = self.postfix_sum(entries, t_pts, inputs)
        if self.post_sum_node is None:
            return the_integral
        if self.post_sum is None:
            return self.post_sum_node.evaluate(
                0.0, the_integral[:, np.newaxis], None, inputs
            ).reshape(-1)
        inputs_stacked = pybamm.solvers.base_solver.flatten_inputs(inputs)
        return self.post_sum(0.0, the_integral, inputs_stacked).full().reshape(-1)

    def postfix_sensitivities(
        self,
        var_name,
        entries,
        t_pts,
        inputs,
        sensitivity_names,
        sensitivities,
        boundary_sensitivities=None,
    ) -> np.ndarray:
        """
        Compute the sensitivities of the postfix sum or integral.

        Parameters
        ----------
        var_name : str
            The name of the variable, for error messages.
        entries : np.ndarray
            The integrand at each sample time, time-major, as passed to
            ``postfix``.
        t_pts : np.ndarray
            The sample times.
        inputs : dict
            The input parameters.
        sensitivity_names : list of str
            The inputs the sensitivities are with respect to, in column order.
        sensitivities : np.ndarray
            The sensitivities of the integrand at each sample time, of shape
            ``(n_t * n_entries, n_p)``, time-major.
        boundary_sensitivities : np.ndarray, optional
            The continuous integral's sensitivity to its moving segment first and
            last times, of shape ``(n_entries, n_p)``, added to the integral of
            ``sensitivities``. A discrete sum has no such term.

        Returns
        -------
        np.ndarray
            The sensitivities of the postfix value, of shape ``(n_entries, n_p)``.
        """
        n_params = np.shape(sensitivities)[-1]
        sensitivities = self._time_major(sensitivities, len(t_pts), (n_params,))
        integral_sensitivities = self._sum_over_time(
            sensitivities, t_pts
        ) + self.initial_condition_sensitivities(inputs, sensitivity_names)
        if (
            boundary_sensitivities is not None
            and self.method == "continuous"
            and np.size(boundary_sensitivities) == integral_sensitivities.size
        ):
            integral_sensitivities = integral_sensitivities + np.reshape(
                boundary_sensitivities, integral_sensitivities.shape
            )
        if self.post_sum_node is None:
            return integral_sensitivities

        # The chain rule needs the post-sum expression's derivative at the integral
        integral = self.postfix_sum(entries, t_pts, inputs)
        p_casadi, p_casadi_stacked, inputs_stacked = _symbolic_inputs(
            inputs, sensitivity_names
        )
        y_casadi = casadi.MX.sym("y", integral.shape[0])
        sens_casadi = casadi.MX.sym("s_var", integral_sensitivities.shape)
        t_casadi = casadi.MX.sym("t")
        post_sum_casadi = self.post_sum_node.to_casadi(
            t_casadi, y_casadi, inputs={**inputs, **p_casadi}
        )

        dpost_dy = casadi.jacobian(post_sum_casadi, y_casadi)
        dpost_dp = casadi.jacobian(post_sum_casadi, p_casadi_stacked)
        sens = dpost_dy @ sens_casadi + dpost_dp
        sens_fun = casadi.Function(
            "sens_fun",
            [t_casadi, y_casadi, p_casadi_stacked, sens_casadi],
            [sens],
        )
        sens_values = sens_fun(0.0, integral, inputs_stacked, integral_sensitivities)
        return sens_values.full()

    @staticmethod
    def to_post_sum_expr(
        var: pybamm.Symbol,
        integral_node: pybamm.ExplicitTimeIntegral | pybamm.DiscreteTimeSum,
        nstates: int,
    ) -> pybamm.Symbol:
        if var == integral_node:
            # If the variable is the integral node itself, return a state vector
            return pybamm.StateVector(slice(0, nstates))

        # Should not be any time or state vector nodes in the children
        if any(isinstance(c, pybamm.Time | pybamm.StateVector) for c in var.children):
            raise ValueError(
                "For expressions containing time integrals, "
                "time or state vector nodes should only appear within the "
                "time integral node. "
            )
        new_children = [
            ProcessedVariableTimeIntegral.to_post_sum_expr(c, integral_node, nstates)
            for c in var.children
        ]
        return var.create_copy(new_children)

    @staticmethod
    def _check_supported(var: pybamm.Symbol, sum_node: pybamm.Symbol) -> None:
        """Raise unless ``sum_node``'s integrand is 0D or 1D and ``var`` keeps its shape."""
        integrand = sum_node.child
        shape = integrand.shape
        mesh = integrand.mesh
        is_0d = len(shape) == 0 or shape[0] == 1
        is_1d = isinstance(mesh, pybamm.SubMesh1D) and integrand.size in (
            mesh.npts,
            mesh.npts + 1,
        )
        if not (is_0d or is_1d):
            domains = {level: d for level, d in integrand.domains.items() if d}
            raise NotImplementedError(
                "Time integrals are only supported for integrands that are 0D or "
                "vary in one spatial dimension, not one of shape "
                f"{shape} on domains {domains}."
            )
        if var.size != sum_node.size:
            raise NotImplementedError(
                "An expression of a time integral must keep the integral's shape "
                f"{sum_node.shape}, not change it to {var.shape}; apply the "
                "reduction to the integrand instead."
            )

    @staticmethod
    def from_pybamm_var(
        var: pybamm.Symbol,
        nstates: int,
    ) -> ProcessedVariableTimeIntegral | None:
        sum_node = None
        for symbol in var.pre_order():
            if isinstance(symbol, pybamm.ExplicitTimeIntegral | pybamm.DiscreteTimeSum):
                if sum_node is None:
                    sum_node = symbol
                else:
                    raise ValueError(
                        "More than one time integral node found in the variable"
                        f" {var.name}. Only one time integral node is allowed."
                    )
        if sum_node is None:
            return None
        if sum_node == var:
            post_sum_node = None
        else:
            sum_y_len = sum_node.evaluate_for_shape().shape[0]
            post_sum_node = ProcessedVariableTimeIntegral.to_post_sum_expr(
                var, sum_node, sum_y_len
            )
        ProcessedVariableTimeIntegral._check_supported(var, sum_node)
        if isinstance(sum_node, pybamm.DiscreteTimeSum):
            return ProcessedVariableTimeIntegral(
                method="discrete",
                post_sum_node=post_sum_node,
                sum_node=sum_node,
                initial_condition=0.0,
                discrete_times=sum_node.sum_times,
            )
        elif isinstance(sum_node, pybamm.ExplicitTimeIntegral):
            # Kept symbolic: it may depend on inputs, which change between solves
            return ProcessedVariableTimeIntegral(
                method="continuous",
                post_sum_node=post_sum_node,
                sum_node=sum_node,
                initial_condition=sum_node.initial_condition,
                discrete_times=None,
            )
        else:
            raise ValueError("Unsupported variable type")  # pragma: no cover


def _nearest_times(t_pts, times):
    """The index of the nearest of ``t_pts`` to each of ``times``, and if it matches."""
    upper = np.clip(np.searchsorted(t_pts, times), 0, len(t_pts) - 1)
    lower = np.clip(upper - 1, 0, len(t_pts) - 1)
    nearest = np.where(
        np.abs(t_pts[lower] - times) < np.abs(t_pts[upper] - times), lower, upper
    )
    matched = np.isclose(t_pts[nearest], times, rtol=_TIME_RTOL, atol=_TIME_ATOL)
    return nearest, matched


def _symbolic_inputs(inputs, sensitivity_names):
    """CasADi symbols for the sensitivity inputs, stacked, and their values."""
    p_casadi = {
        name: casadi.MX.sym(name, _input_size(inputs[name]))
        for name in sensitivity_names
    }
    p_stacked = casadi.vertcat(*p_casadi.values())
    values_stacked = casadi.vertcat(*[inputs[name] for name in sensitivity_names])
    return p_casadi, p_stacked, values_stacked


def _input_size(value):
    """The number of sensitivity parameters an input value holds."""
    return value.shape[0] if isinstance(value, np.ndarray) else 1
