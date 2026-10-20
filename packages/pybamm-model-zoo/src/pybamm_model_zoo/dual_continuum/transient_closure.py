"""Transient closure variable of the dual-continuum model as exponential modes.

DC1t writes :math:`\\langle s^t \\rangle_A(t) = s_0 \\sum_m w_m (1 - e^{-t/\\theta_m})`
with :math:`\\sum_m w_m = 1`, so it starts at 0 and tends to :math:`s_0`, and
each mode turns the time convolution into one ODE. This module fits the modes
to closure data; the model itself does not use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

import pybamm
from pybamm_model_zoo.dual_continuum.isolated_sphere import (
    FARADAY,
    SHORT_TIME_FACTOR,
    mode_weights,
)
from pybamm_model_zoo.dual_continuum.particle import parameter_name, time_constant_name


@dataclass(frozen=True)
class TransientClosureFit:
    """Exponential modes fitted to a transient closure variable.

    Attributes
    ----------
    steady_value : float
        The steady closure variable :math:`s_0`, the limit of :math:`s(t)`.
    weights : numpy.ndarray
        Weights :math:`w_m` of the modes, non-negative and summing to 1.
    time_constants : numpy.ndarray
        Time constants :math:`\\theta_m` of the modes, in increasing order.
    dimensionless : bool
        Whether time and closure variable are dimensionless
        (:math:`t D / L^2` and :math:`s D F / L`).
    max_error : float
        Largest absolute difference between the fit and the fitted data.
    warnings : tuple of str
        Problems found in the data.
    data : tuple of numpy.ndarray
        Times and closure variable as given.
    fitted_data : tuple of numpy.ndarray
        Times and closure variable used in the fit (they differ from ``data``
        when ``fix_early_times`` replaced the early times).
    diffusivity : float or None
        Diffusivity used for the short-time limit, if any.
    early_time_fix : dict or None
        What ``fix_early_times`` changed, if used.
    """

    steady_value: float
    weights: npt.NDArray[np.float64]
    time_constants: npt.NDArray[np.float64]
    dimensionless: bool
    max_error: float
    warnings: tuple[str, ...]
    data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    fitted_data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    diffusivity: float | None = None
    early_time_fix: dict | None = field(default=None)

    @property
    def n_modes(self) -> int:
        """Number of modes, the value of the "transient closure modes" option."""
        return len(self.weights)

    def evaluate(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The fitted closure variable at times ``t``.

        Parameters
        ----------
        t : array_like
            Times, in the units of the data.

        Returns
        -------
        numpy.ndarray
            The fitted closure variable.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        decay = np.exp(-np.outer(t, 1 / self.time_constants))
        return self.steady_value * ((1 - decay) @ self.weights)

    def parameter_values(
        self, domain: str = "Positive", phase: str | None = None
    ) -> dict[str, float]:
        """Parameters of the DC1t model, ready for ``ParameterValues.update``.

        Parameters
        ----------
        domain : str, optional
            "Positive" or "Negative".
        phase : str, optional
            "primary" or "secondary", only for an electrode with several
            particle phases.

        Returns
        -------
        dict
            The steady closure variable, and the weight and time constant of
            each mode, dimensional or dimensionless as fitted.
        """
        s0 = "s0 surface average" + (" dimensionless" if self.dimensionless else "")
        values = {parameter_name(domain, s0, phase): self.steady_value}
        for m, (weight, theta) in enumerate(
            zip(self.weights, self.time_constants, strict=True), start=1
        ):
            values[parameter_name(domain, f"transient closure weight {m}", phase)] = (
                float(weight)
            )
            values[
                parameter_name(domain, time_constant_name(m, self.dimensionless), phase)
            ] = float(theta)
        return values

    def plot(self, show_plot: bool = True):
        """Plot the data and the fit against time, with the fit error.

        Parameters
        ----------
        show_plot : bool, optional
            Whether to show the figure.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        """
        plt = pybamm.import_optional_dependency("matplotlib.pyplot")
        t_data, s_data = self.data
        positive = t_data > 0
        t_plot = np.geomspace(
            min(t_data[positive].min(), self.time_constants[0]) / 10,
            t_data.max(),
            400,
        )
        fig, (ax, ax_error) = plt.subplots(
            2, 1, figsize=(6.5, 6), sharex=True, height_ratios=(3, 1)
        )
        ax.semilogx(
            t_data[positive], s_data[positive], "o", ms=3, color="k", label="data"
        )
        if self.early_time_fix is not None:
            t_fit, s_fit = self.fitted_data
            ax.semilogx(t_fit, s_fit, "x", ms=4, color="0.5", label="fitted points")
        ax.semilogx(
            t_plot, self.evaluate(t_plot), lw=2, label=f"fit, {self.n_modes} modes"
        )
        short_time = self._short_time_limit(t_plot)
        if short_time is not None:
            ax.semilogx(t_plot, short_time, ":", color="0.4", label="short-time limit")
            # The limit grows without bound; keep the axis on the data
            margin = 0.05 * abs(self.steady_value)
            ax.set_ylim(
                min(s_data.min(), self.steady_value, 0) - margin,
                max(s_data.max(), self.steady_value, 0) + margin,
            )
        ax.axhline(self.steady_value, ls="--", lw=0.8, color="0.6")
        ax.set_ylabel("closure variable")
        ax.legend(frameon=False)
        ax_error.semilogx(
            t_data[positive], self.evaluate(t_data[positive]) - s_data[positive], "."
        )
        ax_error.axhline(0, color="0.6", lw=0.8)
        ax_error.set_xlabel("time")
        ax_error.set_ylabel("fit - data")
        fig.tight_layout()
        if show_plot:  # pragma: no cover
            plt.show()
        return fig

    def _short_time_limit(self, t):
        if self.diffusivity is None and not self.dimensionless:
            return None
        return short_time_limit(
            t, self.steady_value, self.diffusivity, self.dimensionless
        )

    def __str__(self) -> str:
        units = "dimensionless" if self.dimensionless else "s"
        relative_error = 100 * self.max_error / abs(self.steady_value)
        lines = [
            (
                f"Transient closure fit: {self.n_modes} modes, max error "
                f"{self.max_error:.3g} ({relative_error:.2f}% of s0 = "
                f"{self.steady_value:.6g})"
            ),
            f"  mode   time constant [{units}]   weight",
        ]
        for m, (weight, theta) in enumerate(
            zip(self.weights, self.time_constants, strict=True), start=1
        ):
            lines.append(f"  {m:4d}   {theta:18.4g}   {weight:.4f}")
        if self.early_time_fix is not None:
            fix = self.early_time_fix
            lines.append(
                f"Early times replaced: data before {fix['resolved_after']:.3g} dropped, "
                f"short-time limit used up to {fix['short_time_until']:.3g}"
            )
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)


def short_time_limit(t, steady_value, diffusivity=None, dimensionless=False):
    """Short-time limit of the closure variable, valid for any geometry.

    While the diffusion length :math:`\\sqrt{Dt}` is much smaller than the
    particles, every surface responds as a flat wall with a uniform flux, so
    :math:`s \\approx -(2/\\sqrt{\\pi})\\sqrt{t/D}/F` (dimensionless:
    :math:`-(2/\\sqrt{\\pi})\\sqrt{t}`), with the sign of the steady value.

    Parameters
    ----------
    t : array_like
        Times.
    steady_value : float
        The steady closure variable, whose sign the limit takes.
    diffusivity : float, optional
        Diffusivity [m2.s-1], required unless ``dimensionless``.
    dimensionless : bool, optional
        Whether ``t`` is dimensionless.

    Returns
    -------
    numpy.ndarray
        The short-time limit at ``t``.
    """
    t = np.asarray(t, dtype=float)
    magnitude = SHORT_TIME_FACTOR * np.sqrt(t)
    if not dimensionless:
        magnitude = magnitude / (np.sqrt(diffusivity) * FARADAY)
    return np.sign(steady_value) * magnitude


def fit_transient_closure(
    t: npt.ArrayLike,
    s,
    steady_value: float | None = None,
    n_modes: int | None = None,
    tolerance: float = 5e-3,
    time_constant_range: tuple[float, float] | None = None,
    diffusivity: float | None = None,
    dimensionless: bool = False,
    fix_early_times: bool = False,
    resolved_after: float | None = None,
    max_modes: int = 30,
) -> TransientClosureFit:
    """Fit exponential modes to a transient closure variable :math:`s(t)`.

    Time constants are spaced logarithmically between the time scales of the
    data; the weights are a non-negative least-squares fit with
    :math:`\\sum_m w_m = 1`. With ``n_modes=None`` modes are added until the
    largest error meets ``tolerance``.

    Parameters
    ----------
    t : array_like
        Times at which :math:`s` is known [s], or dimensionless (:math:`t D/L^2`).
        Values at ``t <= 0`` are ignored.
    s : array_like or callable
        The surface-averaged closure variable at ``t`` [mol.m-1.A-1], or
        dimensionless (:math:`s D F / L`), or a function returning it.
    steady_value : float, optional
        The steady closure variable. Default: the last value of ``s``.
    n_modes : int, optional
        Number of modes. Default: the smallest number meeting ``tolerance``.
    tolerance : float, optional
        Target largest error, relative to the steady value. Default 0.5%.
    time_constant_range : tuple of float, optional
        Shortest and longest time constants. Default: a tenth of the first
        positive time, and a third of the time at which :math:`s` reaches 99%
        of the steady value.
    diffusivity : float, optional
        Active-material diffusivity [m2.s-1]. Enables the check of the early
        data against the short-time limit (always checked when dimensionless).
    dimensionless : bool, optional
        Whether ``t`` and ``s`` are dimensionless.
    fix_early_times : bool, optional
        Replace early data that the closure solve does not resolve by the
        short-time limit before fitting. Off by default; see Notes.
    resolved_after : float, optional
        With ``fix_early_times``, the time from which the data is trusted.
        Default: :math:`10 h^2/D`, with :math:`h` the cell size inferred from
        the early offset.
    max_modes : int, optional
        Largest number of modes tried when ``n_modes`` is not given.

    Returns
    -------
    TransientClosureFit
        The fitted modes, with diagnostics.

    Notes
    -----
    Early data far beyond the short-time limit
    :math:`-(2/\\sqrt{\\pi})\\sqrt{t/D}/F` usually means the closure solve
    does not resolve early times; ``fix_early_times`` is a workaround.
    """
    t = np.asarray(t, dtype=float).ravel()
    s = np.asarray(s(t) if callable(s) else s, dtype=float).ravel()
    if t.shape != s.shape:
        raise ValueError("t and s must have the same length")
    order = np.argsort(t)
    t, s = t[order], s[order]
    data = (t.copy(), s.copy())
    keep = t > 0
    t, s = t[keep], s[keep]
    if t.size < 3:
        raise ValueError("At least three positive times are needed")
    steady_value = float(s[-1] if steady_value is None else steady_value)
    if steady_value == 0:
        raise ValueError("The steady closure variable must be non-zero")
    if diffusivity is None and dimensionless:
        diffusivity = 1.0
    if fix_early_times and diffusivity is None:
        raise ValueError(
            "fix_early_times needs the diffusivity (or dimensionless data)"
        )

    warnings = _check_data(t, s, steady_value, diffusivity, dimensionless, tolerance)
    early_time_fix = None
    if fix_early_times:
        t, s, early_time_fix = _replace_early_times(
            t, s, steady_value, diffusivity, dimensionless, resolved_after
        )

    if time_constant_range is None:
        reached = np.nonzero(s / steady_value >= 0.99)[0]
        t_settle = t[reached[0]] if reached.size else t[-1]
        time_constant_range = (t[0] / 10, max(t_settle / 3, 10 * t[0]))
    low, high = time_constant_range
    if not 0 < low < high:
        raise ValueError("time_constant_range must satisfy 0 < shortest < longest")

    target = tolerance * abs(steady_value)
    candidates = [n_modes] if n_modes is not None else range(2, max_modes + 1)
    best = None
    for n in candidates:
        time_constants = np.geomspace(low, high, n) if n > 1 else np.array([high])
        weights, error = mode_weights(t, s, steady_value, time_constants)
        if best is None or error < best[2] * (1 - 1e-6):
            best = (weights, time_constants, error)
        if error <= target:
            break
    weights, time_constants, error = best
    if n_modes is None and error > target:
        warnings.append(
            f"The tolerance ({100 * tolerance:.2g}% of s0) was not met with up to "
            f"{max_modes} modes; the best fit, {len(weights)} modes, is returned"
        )
    used = weights > 0
    weights, time_constants = weights[used], time_constants[used]

    fit = TransientClosureFit(
        steady_value=steady_value,
        weights=weights,
        time_constants=time_constants,
        dimensionless=dimensionless,
        max_error=float(error),
        warnings=tuple(warnings),
        data=data,
        fitted_data=(t, s),
        diffusivity=diffusivity,
        early_time_fix=early_time_fix,
    )
    for warning in fit.warnings:
        pybamm.logger.warning(f"fit_transient_closure: {warning}")
    return fit


def _check_data(t, s, steady_value, diffusivity, dimensionless, tolerance):
    """Problems in the closure data that the fit cannot represent."""
    warnings = []
    overshoot = np.max(s / steady_value) - 1
    if overshoot > tolerance / 2:
        warnings.append(
            f"s(t) overshoots its final value by {overshoot * abs(steady_value):.3g} "
            f"({100 * overshoot:.2g}% of s0); the modes cannot represent an "
            "overshoot, which often means the closure solve is not converged"
        )
    if diffusivity is not None:
        limit = short_time_limit(t[0], steady_value, diffusivity, dimensionless)
        ratio = s[0] / limit
        if ratio > 2:
            warnings.append(
                f"The first data point (s = {s[0]:.4g} at t = {t[0]:.3g}) is "
                f"{ratio:.3g} times the short-time limit -(2/sqrt(pi)) sqrt(t/D)/F. "
                "The closure solve probably does not resolve early times (a surface "
                f"value extrapolated from cells of size ~{_cell_size(s[0] - limit, diffusivity, dimensionless):.3g} "
                "would explain it); the model would then jump too far when the "
                "current changes. See fix_early_times"
            )
    else:
        fraction = s[0] / steady_value
        if fraction > 0.05:
            warnings.append(
                f"{100 * fraction:.2g}% of s0 is reached by the first data point "
                f"(t = {t[0]:.3g}); check that the closure solve resolves early "
                "times (give the diffusivity to compare with the short-time limit)"
            )
    return warnings


def _cell_size(offset, diffusivity, dimensionless):
    """Cell size h whose centre-to-surface extrapolation offset is h / (2 D F)."""
    if dimensionless:
        return 2 * abs(offset)
    return 2 * abs(offset) * diffusivity * FARADAY


def _replace_early_times(
    t, s, steady_value, diffusivity, dimensionless, resolved_after
):
    """Drop unresolved early data and add points of the short-time limit."""
    offset = s[0] - short_time_limit(t[0], steady_value, diffusivity, dimensionless)
    cell_size = _cell_size(offset, diffusivity, dimensionless)
    if resolved_after is None:
        resolved_after = 10 * cell_size**2 / diffusivity
    # Equivalent sphere radius from s0 = -R / (5 D F), the scale of the particles
    radius = 5 * abs(steady_value) * (1.0 if dimensionless else diffusivity * FARADAY)
    short_time_until = min(resolved_after / 10, (0.03 * radius) ** 2 / diffusivity)
    keep = t >= resolved_after
    if keep.sum() < 3:
        raise ValueError(
            f"Fewer than three data points after resolved_after = {resolved_after:.3g}"
        )
    t_short = np.geomspace(min(t[0], short_time_until / 1e3), short_time_until, 20)
    s_short = short_time_limit(t_short, steady_value, diffusivity, dimensionless)
    info = {
        "cell_size": cell_size,
        "resolved_after": resolved_after,
        "short_time_until": short_time_until,
        "points_dropped": int((~keep).sum()),
    }
    return (
        np.concatenate([t_short, t[keep]]),
        np.concatenate([s_short, s[keep]]),
        info,
    )
