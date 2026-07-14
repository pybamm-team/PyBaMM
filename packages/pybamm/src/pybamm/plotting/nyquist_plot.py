"""Nyquist plot for electrochemical impedance spectroscopy data."""

import numpy as np

from pybamm.util import import_optional_dependency


def nyquist_plot(data, ax=None, show_plot=True, marker="o", linestyle="None", **kwargs):
    """Generate a Nyquist plot from complex impedance data.

    Parameters
    ----------
    data : array-like
        Complex impedance values.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
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
    ax : matplotlib.axes.Axes
    """
    plt = import_optional_dependency("matplotlib.pyplot")

    if isinstance(data, list):
        data = np.array(data)

    if ax is not None:
        fig = None
        show_plot = False
    else:
        fig, ax = plt.subplots()

    ax.plot(data.real, -data.imag, marker=marker, linestyle=linestyle, **kwargs)
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    axmax = max(xmax, ymax)
    ax.set_xlim(0, axmax)
    ax.set_ylim(0, axmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$Z_\mathrm{Re}$ [Ohm]")
    ax.set_ylabel(r"$-Z_\mathrm{Im}$ [Ohm]")
    if show_plot:  # pragma: no cover
        plt.show()

    return fig, ax
