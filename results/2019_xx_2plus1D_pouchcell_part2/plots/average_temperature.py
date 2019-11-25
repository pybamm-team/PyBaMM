# cell temperature


def plot_average_temperature(
    ax, spm=None, spmecc=None, reduced=None, full=None, x_axis="Time [h]", color=None
):

    if spm:
        x = spm[x_axis]
        T_vol_av = spm["Volume-averaged cell temperature [K]"]
        ax.plot(x, T_vol_av, label="SPM", color=color, linestyle=":")

    if spmecc:
        x = spmecc[x_axis]
        T_vol_av = spmecc["Volume-averaged cell temperature [K]"]
        ax.plot(x, T_vol_av, label="SPMeCC", color=color, linestyle="-.")

    if reduced:
        x = reduced[x_axis]
        T_vol_av = reduced["Volume-averaged cell temperature [K]"]
        ax.plot(x, T_vol_av, label="Reduced 2+1D", color=color, linestyle="--")

    if full:
        x = full[x_axis]
        T_vol_av = full["Volume-averaged cell temperature [K]"]
        ax.plot(x, T_vol_av, label="Full 2+1D", color=color, linestyle="-")

    ax.set_xlabel(x_axis)
    ax.set_ylabel("Volume-averaged cell temperature [K]")
    ax.legend()

