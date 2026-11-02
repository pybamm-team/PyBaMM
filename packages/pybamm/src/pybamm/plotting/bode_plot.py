"""Bode plot for electrochemical impedance spectroscopy data."""

import numpy as np

from pybamm.util import import_optional_dependency


def bode_plot(
    frequencies,
    impedance,
    ax=None,
    show_plot=True,
    marker="o",
    linestyle="None",
    **kwargs,
):
    """Generate a Bode plot (impedance magnitude and phase against frequency).

    Parameters
    ----------
    frequencies : array-like
        Frequencies in Hz.
    impedance : array-like
        Complex impedance values.
    ax : sequence of two matplotlib.axes.Axes, optional
        Axes for the magnitude and phase. If None, a new figure is created.
    show_plot : bool, optional
        Whether to call ``plt.show()`` (default True). Automatically set to
        False when *ax* is provided.
    marker : str, optional
        Marker style (default ``"o"``).
    linestyle : str, optional
        Line style (default ``"None"``).
    **kwargs
        Additional keyword arguments passed to ``ax.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure (None when *ax* was provided by the caller).
    ax : tuple of matplotlib.axes.Axes
        The magnitude and phase axes.
    """
    plt = import_optional_dependency("matplotlib.pyplot")

    frequencies = np.asarray(frequencies)
    impedance = np.asarray(impedance, dtype=complex)

    if ax is not None:
        fig = None
        show_plot = False
    else:
        fig, ax = plt.subplots(2, 1, sharex=True)
    ax_magnitude, ax_phase = ax

    ax_magnitude.loglog(
        frequencies, np.abs(impedance), marker=marker, linestyle=linestyle, **kwargs
    )
    ax_phase.semilogx(
        frequencies,
        -np.angle(impedance, deg=True),
        marker=marker,
        linestyle=linestyle,
        **kwargs,
    )
    ax_magnitude.set_ylabel(r"$|Z|$ [Ohm]")
    ax_phase.set_ylabel(r"$-\arg(Z)$ [deg]")
    ax_phase.set_xlabel("Frequency [Hz]")
    if show_plot:  # pragma: no cover
        plt.show()

    return fig, (ax_magnitude, ax_phase)
