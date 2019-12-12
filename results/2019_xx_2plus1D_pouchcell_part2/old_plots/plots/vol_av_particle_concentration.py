def plot_vol_av_particle_concentration(
    ax, var_name, spmecc=None, reduced=None, full=None, x_axis="Time [h]", color=None,
):

    if spmecc:
        x = spmecc[x_axis]
        voltage = spmecc[var_name]
        ax.plot(x, voltage, label="SPMeCC", color=color, linestyle="-.")

    if reduced:
        x = reduced[x_axis]
        voltage = reduced[var_name]
        ax.plot(x, voltage, label="Reduced 2+1D", color=color, linestyle="--")

    if full:
        x = full[x_axis]
        voltage = full[var_name]
        ax.plot(x, voltage, label="Full 2+1D", color=color, linestyle="-")

    ax.set_xlabel(x_axis)
    ax.set_ylabel(var_name)
    ax.legend()

