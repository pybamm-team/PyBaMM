# average current densisty plots


def plot_av_cc_current(
    ax, spm=None, spmecc=None, reduced=None, full=None, x_axis="Time [h]", color=None
):

    if spm:
        x = spm[x_axis]
        current = spm["Average local current density [A.m-2]"]
        ax.plot(x, current, label="SPM", color=color, linestyle=":")

    if spmecc:
        x = spmecc[x_axis]
        current = spmecc["Average local current density [A.m-2]"]
        ax.plot(x, current, label="SPMeCC", color=color, linestyle="-.")

    if reduced:
        x = reduced[x_axis]
        current = reduced["Average local current density [A.m-2]"]
        ax.plot(x, current, label="Reduced 2+1D", color=color, linestyle="--")

    if full:
        x = full[x_axis]
        current = full["Average local current density [A.m-2]"]
        ax.plot(x, current, label="Full 2+1D", color=color, linestyle="-")

    ax.set_xlabel(x_axis)
    ax.set_ylabel("Average local current density [A.m-2]")
    ax.legend()

