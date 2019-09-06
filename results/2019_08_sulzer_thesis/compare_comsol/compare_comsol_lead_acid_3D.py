import argparse
import pybamm
import numpy as np
import os
import pickle
import scipy.interpolate as interp
import matplotlib.pyplot as plt

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = None

parser = argparse.ArgumentParser()
parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
args = parser.parse_args()

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

"-----------------------------------------------------------------------------"
"Pick Crate and load comsol data"

# Crate
# NOTE: the results in pybamm stop when a voltage cutoff is reached, so
# for higher C-rate the pybamm solution may stop before the comsol solution
Crates = [0.1, 1, 2]
sigmas = [8000, 5 * 8000, 10 * 8000, 100 * 8000]

# load the comsol results
comsol_voltages = pickle.load(
    open("results/2019_08_sulzer_thesis/compare_comsol/comsol_voltages.pickle", "rb")
)
savefile = "comsol_voltage_comparison_data.pickle"

if args.compute:
    "-----------------------------------------------------------------------------"
    "Create and solve pybamm model"

    # load model and geometry
    pybamm.set_logging_level("INFO")
    pybamm_models = [
        pybamm.lead_acid.NewmanTiedemann(
            {"surface form": "algebraic"}, name="1D PyBaMM"
        ),
        pybamm.lead_acid.NewmanTiedemann(
            {
                "surface form": "algebraic",
                "current collector": "potential pair",
                "dimensionality": 1,
            },
            name="1+1D PyBaMM",
        ),
    ]

    # load parameters and process model and geometry
    param = pybamm_models[0].default_parameter_values
    # Change the t_plus function to agree with Comsol
    param["Darken thermodynamic factor"] = np.ones_like
    param["MacInnes t_plus function"] = lambda x: 1 - 2 * x

    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 8, var.x_s: 8, var.x_p: 8, var.z: 8}
    discs = {}
    for model in pybamm_models:
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)

        # create mesh
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        discs[model] = disc

    all_voltages = comsol_voltages
    for Crate in Crates:
        current = Crate * 17
        for sigma in sigmas:
            # Comsol
            # discharge timescale
            pybamm.logger.info(
                """Setting typical current to {} A
            and positive electrode condutivity to {} S/m""".format(
                    current, sigma
                )
            )
            param.update(
                {
                    "Typical current [A]": current,
                    "Positive electrode conductivity [S.m-1]": sigma,
                }
            )
            tau = param.process_symbol(
                pybamm.standard_parameters_lead_acid.tau_discharge
            )

            # solve model at comsol times
            comsol_t_1d = comsol_voltages[Crate][sigma]["1D COMSOL"][0]
            comsol_t_2d = comsol_voltages[Crate][sigma]["2D COMSOL"][0]

            # PyBaMM
            for model in pybamm_models:
                if model == "1+1D PyBaMM":
                    time = comsol_t_2d
                else:
                    time = comsol_t_1d
                t_eval = time / tau.evaluate(0)
                param.update_model(model, discs[model])
                solution = model.default_solver.solve(model, t_eval)
                voltage = pybamm.ProcessedVariable(
                    model.variables["Battery voltage [V]"], solution.t, solution.y, mesh
                )
                all_voltages[Crate][sigma][model.name] = (time, voltage(t_eval))

    with open(savefile, "wb") as f:
        data = all_voltages
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

else:
    try:
        with open(savefile, "rb") as f:
            all_voltages = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Run with --compute to generate results")

fig, axes = plt.subplots(len(sigmas), len(Crates), sharey=True, figsize=(6.4, 5))
linestyles = {
    "1D COMSOL": "k:",
    "1D PyBaMM": "r:",
    "2D COMSOL": "k-",
    "1+1D PyBaMM": "r-",
}
for i, Crate in enumerate(Crates):
    for j, sigma in enumerate(sigmas):
        all_voltages[Crate][sigma].update(comsol_voltages[Crate][sigma])
        variables = all_voltages[Crate][sigma]
        ax = axes[j, i]
        for model, (time, voltage) in variables.items():
            time = time / 3600
            if i == 0 and j == 0:
                label = model
            else:
                label = None
            ax.plot(time, voltage, linestyles[model], label=label)
            ax.set_xlim([0, np.max(time)])
            ax.set_ylim([10.5, 13.5])
            if i == 0:
                sigma_exponent = int(np.floor(np.log10(sigma)))
                sigma_dash = 0.05 * sigma / 8000
                ax.set_ylabel(
                    (
                        "$\\hat{{\\sigma}}_p = {}\\times 10^{{{}}}$ S/m"
                        + "\n$(\\sigma'_p={}/\\mathcal{{C}})$"
                    ).format(sigma / 10 ** sigma_exponent, sigma_exponent, sigma_dash),
                    rotation=0,
                    labelpad=50,
                )
                ax.yaxis.get_label().set_verticalalignment("center")
            if j == 0:
                ax.set_title(
                    "\\textbf{{({})}} {}C ($\\mathcal{{C}}_e={}$)".format(
                        chr(97 + i), abs(Crate), abs(Crate) * 0.6
                    )
                )
            if j == len(sigmas) - 1:
                ax.set_xlabel("Time [h]")
            else:
                ax.set_xticklabels([])
            if i == 1 and j == len(sigmas) - 1:
                ax.set_xticks([0, 0.4, 0.8])

leg = fig.legend(loc="lower center", ncol=4, columnspacing=1.1)
plt.subplots_adjust(
    bottom=0.19, top=0.92, left=0.28, right=0.97, hspace=0.08, wspace=0.05
)
if OUTPUT_DIR is not None:
    plt.savefig(OUTPUT_DIR + "comsol_voltages_comparison.eps", format="eps", dpi=1000)
plt.show()
