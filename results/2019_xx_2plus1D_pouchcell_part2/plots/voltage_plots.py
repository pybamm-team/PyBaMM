import matplotlib.pyplot as plt


def plot_voltage(
    t, spm=None, spmecc=None, reduced=None, full=None, x_axis="Time [h]", color=None
):

    if spm:
        x = spm[x_axis](t)
        voltage = spm["Terminal voltage [V]"](t)
        plt.plot(x, voltage, label="SPM", color=color, linestyle=":")

    if spmecc:
        x = spmecc[x_axis]
        voltage = spmecc["Terminal voltage [V]"]
        plt.plot(x, voltage, label="SPMeCC", color=color, linestyle="-.")

    if reduced:
        x = reduced[x_axis](t)
        voltage = reduced["Terminal voltage [V]"](t)
        plt.plot(x, voltage, label="Reduced 2+1D", color=color, linestyle="--")

    if full:
        x = full[x_axis](t)
        voltage = full["Terminal voltage [V]"](t)
        plt.plot(x, voltage, label="Full 2+1D", color=color, linestyle="-")

    plt.xlabel(x_axis)
    plt.ylabel("Terminal voltage [V]")
    plt.legend()

