import numpy as np
import matplotlib.pyplot as plt


def plot_yz_averaged_electrolyte_concentration(
    times, spmecc=None, reduced=None, full=None
):

    color = [None, None, None]

    num_of_models = 0

    if reduced:
        num_of_models += 1
    if full:
        num_of_models += 1

    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(11)

    fig.tight_layout()

    if spmecc:
        model_name = "SPMeCC"
        plot_model(fig, ax, times, reduced, model_name, "blue")

    if reduced:
        model_name = "2+1 SPMe"
        plot_model(fig, ax, times, reduced, model_name, "red")

    if full:
        model_name = "2+1 DFN"
        plot_model(fig, ax, times, reduced, model_name, "green")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$c_{e}$")


def plot_model(fig, ax, times, model, model_name, color):
    x = np.linspace(0, 1, 100)

    for t in times:
        ax.plot(x, model["YZ-averaged electrolyte concentration"])


def make_plot(fig, ax, y, z, var, name, color_lim):

    if color_lim is not None:
        im = ax.pcolormesh(
            y, z, var, vmin=color_lim[0], vmax=color_lim[1], shading="gouraud"
        )
    else:
        im = ax.pcolormesh(y, z, var, shading="gouraud")
    ax.set_title(name)

    fig.colorbar(im, ax=ax)

