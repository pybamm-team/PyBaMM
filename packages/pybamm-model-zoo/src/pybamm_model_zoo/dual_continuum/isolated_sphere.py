"""Transient closure variable of an isolated sphere, and its exponential modes.

The model uses :func:`isolated_sphere_modes` for ``"closure variable":
"isolated sphere"``; :mod:`~pybamm_model_zoo.dual_continuum.transient_closure`
fits modes to closure data of other geometries.
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, nnls

#: Faraday constant [C.mol-1].
FARADAY = 96485.33212
#: Factor of the short-time limit, s ~ -(2 / sqrt(pi)) sqrt(t / D) / F.
SHORT_TIME_FACTOR = 2 / np.sqrt(np.pi)
#: Number of eigenvalues of tan(x) = x used for the isolated sphere.
_N_EIGENVALUES = 20000


@functools.cache
def sphere_eigenvalues(n: int = _N_EIGENVALUES) -> npt.NDArray[np.float64]:
    """Positive roots of :math:`\\tan\\lambda = \\lambda`, in increasing order.

    Parameters
    ----------
    n : int, optional
        Number of roots.

    Returns
    -------
    numpy.ndarray
        The first ``n`` roots.
    """
    n_exact = min(n, 200)
    exact = [
        brentq(lambda x: np.tan(x) - x, k * np.pi + 1e-9, (k + 0.5) * np.pi - 1e-9)
        for k in range(1, n_exact + 1)
    ]
    # Asymptotic expansion, accurate to round-off beyond the first 200 roots
    q = (np.arange(n_exact + 1, n + 1) + 0.5) * np.pi
    asymptotic = q - 1 / q - 2 / (3 * q**3)
    return np.concatenate([exact, asymptotic])


def isolated_sphere_closure(
    t: npt.ArrayLike, radius: float | None = None, diffusivity: float | None = None
) -> npt.NDArray[np.float64]:
    """Surface-averaged transient closure variable of an isolated sphere.

    The solution of the closure problem of Paten et al. (2026c, Eq. 13),

    .. math::
        \\langle s^t \\rangle_A(t) = -\\frac{R}{DF}\\left(\\frac{1}{5}
        - \\sum_m \\frac{2}{\\lambda_m^2} e^{-\\lambda_m^2 D t / R^2}\\right),

    where :math:`\\lambda_m` are the positive roots of :math:`\\tan\\lambda = \\lambda`.
    Without ``radius`` and ``diffusivity`` it is the dimensionless closure
    variable :math:`s D F / R` as a function of :math:`t D / R^2`.

    Parameters
    ----------
    t : array_like
        Times [s], or dimensionless times.
    radius : float, optional
        Sphere radius [m].
    diffusivity : float, optional
        Active-material diffusivity [m2.s-1].

    Returns
    -------
    numpy.ndarray
        The closure variable at ``t`` [mol.m-1.A-1], or dimensionless.
    """
    if (radius is None) != (diffusivity is None):
        raise ValueError("Give both radius and diffusivity, or neither")
    t_star = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t_star < 0):
        raise ValueError("Times must be non-negative")
    if radius is not None:
        t_star = t_star * diffusivity / radius**2
    lam2 = sphere_eigenvalues() ** 2
    transient = np.zeros_like(t_star)
    for start in range(0, lam2.size, 2000):
        chunk = lam2[start : start + 2000]
        transient += (2 / chunk * np.exp(-np.outer(t_star, chunk))).sum(axis=1)
    s_star = -(0.2 - transient)
    # The series converges slowly at very early times: use the short-time limit
    early = t_star < 1e-8
    s_star[early] = -SHORT_TIME_FACTOR * np.sqrt(t_star[early])
    if radius is None:
        return s_star
    return s_star * radius / (diffusivity * FARADAY)


def mode_weights(t, s, steady_value, time_constants):
    """Non-negative weights, summing to 1, of modes with given time constants.

    Parameters
    ----------
    t : numpy.ndarray
        Positive times.
    s : numpy.ndarray
        The closure variable at ``t``.
    steady_value : float
        The steady closure variable.
    time_constants : numpy.ndarray
        The time constants of the modes, in the units of ``t``.

    Returns
    -------
    tuple
        The weights, and the largest error of the fitted closure variable.
    """
    basis = 1 - np.exp(-np.outer(t, 1 / time_constants))
    # A heavily weighted extra row imposes sum(w) = 1
    penalty = 1e3 * np.sqrt(len(t))
    matrix = np.vstack([basis, penalty * np.ones(len(time_constants))])
    rhs = np.concatenate([s / steady_value, [penalty]])
    weights, _ = nnls(matrix, rhs)
    weights = weights / weights.sum()
    return weights, np.abs(steady_value * (basis @ weights) - s).max()


@functools.cache
def isolated_sphere_modes(n_modes: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Weights and dimensionless time constants for an isolated sphere.

    Time constants are in units of :math:`R^2/D`, fitted to
    :func:`isolated_sphere_closure` between :math:`10^{-6}` and the slowest
    eigenmode, :math:`1/\\lambda_1^2`.

    Parameters
    ----------
    n_modes : int
        Number of modes.

    Returns
    -------
    tuple
        The weights and the time constants.
    """
    t_star = np.geomspace(1e-8, 2, 600)
    slowest = 1 / sphere_eigenvalues(1)[0] ** 2
    time_constants = (
        np.geomspace(1e-6, slowest, n_modes) if n_modes > 1 else np.array([slowest])
    )
    weights, _ = mode_weights(
        t_star, isolated_sphere_closure(t_star), -0.2, time_constants
    )
    return tuple(float(w) for w in weights), tuple(float(x) for x in time_constants)
