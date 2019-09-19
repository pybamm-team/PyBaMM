#
# Simulations: self-discharge
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybamm
import shared_plotting
from config import OUTPUT_DIR
from shared_solutions import model_comparison


def plot_voltages(all_variables, t_eval):
    shared_plotting.plot_voltages(all_variables, t_eval)
    file_name = "sefl_discharge_voltage_comparison.eps"
    if OUTPUT_DIR is not None:
        plt.savefig(OUTPUT_DIR + file_name, format="eps", dpi=1000)


def self_discharge_states(compute):
    save_file = "self_discharge_data.pickle"
    if compute:
        models = [
            pybamm.lead_acid.Full(name="Full, without oxygen"),
            pybamm.lead_acid.Full(
                {"side reactions": ["oxygen"]}, name="Full, with oxygen"
            ),
            pybamm.lead_acid.LOQS(
                {"surface form": "algebraic", "side reactions": ["oxygen"]},
                name="LOQS, with oxygen",
            ),
        ]
        extra_parameter_values = {
            "Current function": pybamm.GetConstantCurrent(current=0)
        }
        t_eval = np.linspace(0, 1000, 100)
        all_variables, t_eval = model_comparison(
            models, [1], t_eval, extra_parameter_values=extra_parameter_values
        )
        with open(save_file, "wb") as f:
            data = (all_variables, t_eval)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(save_file, "rb") as f:
                (all_variables, t_eval) = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Run script with '--compute' first to generate results"
            )
    plot_voltages(all_variables, t_eval)


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="(Re)-compute results.")
    args = parser.parse_args()
    self_discharge_states(args.compute)
    plt.show()
