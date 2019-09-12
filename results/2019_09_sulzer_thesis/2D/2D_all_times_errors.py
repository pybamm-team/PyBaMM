#
# Times and errors for discharge of a lead-acid battery in 2D
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
import shared_plotting_2D
from collections import defaultdict
from shared_solutions_2D import error_comparison, time_comparison

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None


def fill_nan(array):
    ok = ~np.isnan(array)
    xp = ok.ravel().nonzero()[0]
    fp = array[~np.isnan(array)]
    x = np.isnan(array).ravel().nonzero()[0]
    array[np.isnan(array)] = np.interp(x, xp, fp)
    return array


def plot_errors(model_voltages):
    t_eval = np.linspace(0, 1)
    models = list(model_voltages.keys())
    Crates = list(model_voltages[models[0]].keys())
    sigmas = list(model_voltages[models[0]][Crates[0]].keys())
    errors = np.zeros((len(Crates), len(sigmas)))
    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(6.4, 4.5))
    models = [
        "1D LOQS",
        "1D Composite",
        "1D Full",
        "1+1D LOQS",
        "1+1D Composite Averaged",
        "1+1D Composite",
    ]
    for i, model in enumerate(models):
        Crates_variables = model_voltages[model]
        for j, (Crate, sigmas_voltages) in enumerate(Crates_variables.items()):
            for k, (sigma, reduced_voltage) in enumerate(sigmas_voltages.items()):
                full_voltage = model_voltages["1+1D Full"][Crate][sigma]
                errors[k, j] = pybamm.rmse(full_voltage, reduced_voltage)
        # errors = fill_nan(errors)
        ax = axes.flat[i]
        ax.set_xlabel("C-rate, $\\mathcal{C}$")
        if i % 3 == 0:
            ax.set_ylabel("$\\hat{{\\sigma}}_p$ [S/m]")
        CS = ax.contourf(
            Crates, sigmas, np.log(errors), vmin=-5, vmax=1, levels=100, cmap="jet"
        )
        for c in CS.collections:
            c.set_edgecolor("face")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model)
    cb_ax = fig.add_axes([0.89, 0.11, 0.02, 0.77])
    cbar = fig.colorbar(CS, cax=cb_ax, ticks=[-10, -8, -6, -4, -2, 0, 2])
    cbar.set_label("log(RMSE) [V]", rotation=270, labelpad=15)
    file_name = "2d_asymptotics_rmse.eps"
    plt.subplots_adjust(hspace=0.5, wspace=0.1, left=0.1, right=0.87)
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def plot_times(model_times):
    "Plot solver times for both 1D and 2D"
    shared_plotting_2D.plot_times(model_times, dimensions=1)
    file_name = "1d_discharge_asymptotics_solver_times.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)

    shared_plotting_2D.plot_times(model_times, dimensions=2)
    file_name = "2d_discharge_asymptotics_solver_times.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def discharge_errors(compute):
    savefile = "nocutoff_8by8_2d_discharge_asymptotics_errors.pickle"
    if compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(
                {"surface form": "algebraic"}, name="1D Full"
            ),
            pybamm.lead_acid.LOQS(name="1D LOQS"),
            pybamm.lead_acid.Composite(name="1D Composite"),
            pybamm.lead_acid.Composite(
                {"current collector": "potential pair quite conductive averaged"},
                name="1+1D Composite Averaged",
            ),
            pybamm.lead_acid.NewmanTiedemann(
                {
                    "surface form": "algebraic",
                    "dimensionality": 1,
                    "current collector": "potential pair",
                },
                name="1+1D Full",
            ),
            pybamm.lead_acid.LOQS(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1+1D LOQS",
            ),
            pybamm.lead_acid.Composite(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1+1D Composite",
            ),
        ]
        Crates = np.logspace(np.log10(0.01), np.log10(10), 8)
        sigmas = np.logspace(np.log10(8000), np.log10(1000 * 8000), 8)
        t_eval = np.linspace(0, 1, 100)
        model_voltages = error_comparison(models, Crates, sigmas, t_eval)
        with open(savefile, "wb") as f:
            pickle.dump(model_voltages, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile, "rb") as f:
                model_voltages = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_errors(model_voltages)


def discharge_times(compute):
    savefile = "2d_discharge_asymptotics_times.pickle"
    if compute:
        models = [
            pybamm.lead_acid.NewmanTiedemann(
                {"surface form": "algebraic"}, name="1D Full"
            ),
            pybamm.lead_acid.LOQS(name="1D LOQS"),
            pybamm.lead_acid.Composite(name="1D Composite"),
            pybamm.lead_acid.Composite(
                {"current collector": "potential pair quite conductive averaged"},
                name="1+1D Composite Averaged",
            ),
            pybamm.lead_acid.NewmanTiedemann(
                {
                    "surface form": "algebraic",
                    "dimensionality": 1,
                    "current collector": "potential pair",
                },
                name="1+1D Full",
            ),
            pybamm.lead_acid.LOQS(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1+1D LOQS",
            ),
            pybamm.lead_acid.Composite(
                {"dimensionality": 1, "current collector": "potential pair"},
                name="1+1D Composite",
            ),
        ]
        Crate = 1
        sigma = 10 * 8000
        all_npts = np.logspace(0.5, 3, 10)
        t_eval = np.linspace(0, 0.6, 100)
        model_times = time_comparison(models, Crate, sigma, all_npts, t_eval)
        with open(savefile, "wb") as f:
            pickle.dump(model_times, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(savefile, "rb") as f:
                model_times = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_times(model_times)


if __name__ == "__main__":
    pybamm.set_logging_level("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    discharge_errors(args.compute)
    discharge_times(args.compute)
    plt.show()
