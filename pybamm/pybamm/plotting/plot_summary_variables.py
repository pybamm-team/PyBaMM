#
# Method for plotting/comparing summary variables
#
import numpy as np
import pybamm
from pybamm.util import have_optional_dependency


def plot_summary_variables(
    solutions, output_variables=None, labels=None, testing=False, **kwargs_fig
):
    """
    Generate a plot showing/comparing the summary variables.

    Parameters
    ----------
    solutions : (iter of) :class:`pybamm.Solution`
        The solution(s) for the model(s) from which to extract summary variables.
    output_variables: list (optional)
        A list of variables to plot automatically. If None, the default ones are used.
    labels: list (optional)
        A list of labels to be added to the legend. No labels are added by default.
    testing : bool (optional)
        Whether to actually make the plot (turned off for unit tests).
    kwargs_fig
        Keyword arguments, passed to plt.subplots.

    """
    plt = have_optional_dependency("matplotlib.pyplot")

    if isinstance(solutions, pybamm.Solution):
        solutions = [solutions]

    # setting a default value for figsize
    kwargs_fig = {"figsize": (15, 8), **kwargs_fig}

    if output_variables is None:
        output_variables = [
            "Capacity [A.h]",
            "Loss of lithium inventory [%]",
            "Total capacity lost to side reactions [A.h]",
            "Loss of active material in negative electrode [%]",
            "Loss of active material in positive electrode [%]",
            "x_100",
            "x_0",
            "y_100",
            "y_0",
        ]

    # find the number of subplots to be created
    length = len(output_variables)
    n = int(length // np.sqrt(length))
    m = int(np.ceil(length / n))

    # create subplots
    fig, axes = plt.subplots(n, m, **kwargs_fig)

    # loop through the subplots and plot the output_variables
    for var, ax in zip(output_variables, axes.flat):
        # loop through the solutions to compare output_variables
        for solution in solutions:
            # plot summary variable v/s cycle number
            ax.plot(
                solution.summary_variables["Cycle number"],
                solution.summary_variables[var],
            )
        # label the axes
        ax.set_xlabel("Cycle number")
        ax.set_ylabel(var)
        ax.set_xlim([1, solution.summary_variables["Cycle number"][-1]])

    fig.tight_layout()

    # add labels in legend
    if labels is not None:  # pragma: no cover
        fig.legend(labels, loc="lower right")
    if not testing:  # pragma: no cover
        plt.show()

    return axes
