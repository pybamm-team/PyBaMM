

def plot_voltage(
    ax, spm=None, spmecc=None, reduced=None, full=None, x_axis="Time [h]", color=None
):

    if spm:
        x = spm[x_axis]
        voltage = spm["Terminal voltage [V]"]
        ax.plot(x, voltage, label="SPM", color=color, linestyle=":")

    if spmecc:
        x = spmecc[x_axis]
        voltage = spmecc["Terminal voltage [V]"]
        ax.plot(x, voltage, label="SPMeCC", color=color, linestyle="-.")

    if reduced:
        x = reduced[x_axis]
        voltage = reduced["Terminal voltage [V]"]
        ax.plot(x, voltage, label="Reduced 2+1D", color=color, linestyle="--")

    if full:
        x = full[x_axis]
        voltage = full["Terminal voltage [V]"]
        ax.plot(x, voltage, label="Full 2+1D", color=color, linestyle="-")

    ax.set_xlabel(x_axis)
    ax.set_ylabel("Terminal voltage [V]")
    ax.legend()

