import pybamm
import matplotlib.pyplot as plt


parameters = [
    "Marquis2019",
    "NCA_Kim2011",
    "Ramadass2004",
    "Chen2020",
    "Chen2020_plating",
    "Ecker2015",
]
models = ["SPM", "DFN"]
npts = [4, 8, 16, 32, 64]
for model_ in models:
    if model_ == "SPM":
        x = 1
    else:
        x = 2

    for params in parameters:

        solutions = []
        ns = []

        for N in npts:

            solver = pybamm.CasadiSolver(mode="fast")
            if model_ == "SPM":
                model = pybamm.lithium_ion.SPM()
            else:
                model = pybamm.lithium_ion.DFN()
            parameter_values = pybamm.ParameterValues(params)
            var_pts = {
                "x_n": N,  # negative electrode
                "x_s": N,  # separator
                "x_p": N,  # positive electrode
                "r_n": N,  # negative particle
                "r_p": N,  # positive particle
            }
            sim = pybamm.Simulation(
                model, solver=solver, parameter_values=parameter_values, var_pts=var_pts
            )
            time = 0
            for k in range(0, 5):

                solution = sim.solve([0, 3500])
                time += solution.solve_time.value
            time = time / 5

            state = sim.built_model.concatenated_initial_conditions.shape[0]
            ns.append(state)

            solutions.append(time)

            plt.subplot(1, 2, x)

            plt.plot(ns, solutions)
            plt.title(f"Work Precision Sets for {model_}")
            plt.xlabel("number of states")
            # plt.xticks(ns)
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel("time(s)")

plt.gca().legend(parameters, loc="upper right", fancybox=True, framealpha=0.5)
plt.tight_layout()
# plt.show()

plt.savefig(
    f"benchmarks/benchmark_images/time_vs_no_of_states_{pybamm.__version__}.png"
)


content = f"## Solve Time vs Number of states\n<img src='./benchmark_images/time_vs_no_of_states_{pybamm.__version__}.png'>\n"  # noqa

with open("./benchmarks/release_work_precision_sets.md", "r") as original:
    data = original.read()
with open("./benchmarks/release_work_precision_sets.md", "w") as modified:
    modified.write(f"{content}\n{data}")
