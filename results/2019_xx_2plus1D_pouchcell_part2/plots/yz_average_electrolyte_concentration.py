import numpy as np
import matplotlib.pyplot as plt


def plot_yz_averaged_electrolyte_concentration(
    times, spmecc=None, reduced=None, full=None
):

    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(11)
    fig.tight_layout()

    # if spmecc:
    #     model_name = "SPMeCC"
    #     plot_model(fig, ax, times, spmecc, model_name, "coral", "-")

    if reduced:
        model_name = "2+1 SPMe"
        plot_model(fig, ax, times, reduced, model_name, "royalblue", "-")

    if full:
        model_name = "2+1 DFN"
        plot_model(fig, ax, times, full, model_name, "darkgreen", "--")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$c_{e}$")


def plot_model(fig, ax, times, model, model_name, color, linestyle):
    x = np.linspace(0, 1, 100)

    linestyles = [":", "-.", "--", "-"]
    colors = ["royalblue", "coral", "forestgreen", "orchid"]

    for i, t in enumerate(times):
        ax.plot(
            x,
            model["YZ-averaged electrolyte concentration"](t, x),
            linestyle=linestyle,
            color=colors[i],
        )
