"""Combined Nyquist and Bode plots for one or more impedance spectra."""

import pybamm
from pybamm.util import import_optional_dependency


def eis_plot(solutions, labels=None, show_plot=True, **kwargs):
    """Plot Nyquist and Bode plots side by side, overlaying several spectra.

    Parameters
    ----------
    solutions : :class:`pybamm.EISSolution` or :class:`pybamm.EISSimulation` or list
        The spectra to plot. Simulations must have been solved.
    labels : list of str, optional
        A legend label for each spectrum.
    show_plot : bool, optional
        Whether to call ``plt.show()`` (default True).
    **kwargs
        Additional keyword arguments passed to ``ax.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of matplotlib.axes.Axes
        The Nyquist, magnitude and phase axes.
    """
    plt = import_optional_dependency("matplotlib.pyplot")

    if not isinstance(solutions, list):
        solutions = [solutions]
    solutions = [
        s.solution if isinstance(s, pybamm.EISSimulation) else s for s in solutions
    ]
    if any(s is None for s in solutions):
        raise ValueError(
            "EIS simulation has not been solved yet. Call solve() before plotting."
        )
    if labels is None:
        labels = [None] * len(solutions)
    elif len(labels) != len(solutions):
        raise ValueError("labels must have the same length as solutions")

    fig = plt.figure(figsize=(12, 5))
    grid = fig.add_gridspec(2, 2)
    ax_nyquist = fig.add_subplot(grid[:, 0])
    ax_magnitude = fig.add_subplot(grid[0, 1])
    ax_phase = fig.add_subplot(grid[1, 1], sharex=ax_magnitude)

    for solution, label in zip(solutions, labels, strict=True):
        pybamm.nyquist_plot(solution.impedance, ax=ax_nyquist, label=label, **kwargs)
        pybamm.bode_plot(
            solution.frequencies,
            solution.impedance,
            ax=(ax_magnitude, ax_phase),
            label=label,
            **kwargs,
        )

    if any(label is not None for label in labels):
        ax_nyquist.legend()
    fig.tight_layout()
    if show_plot:  # pragma: no cover
        plt.show()

    return fig, (ax_nyquist, ax_magnitude, ax_phase)
