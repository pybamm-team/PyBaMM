import numpy as np
import pytest

import pybamm


class TestPlotSummaryVariables:
    @pytest.fixture(scope="class")
    def summary_solutions(self):
        return [
            pybamm.Simulation(
                pybamm.lithium_ion.SPM(),
                experiment=pybamm.Experiment(
                    ["Discharge at C/10 for 10 seconds"] * number_of_cycles
                ),
            ).solve(calc_esoh=False)
            for number_of_cycles in [2, 3]
        ]

    @pytest.mark.parametrize("number_of_variables", [1, 2])
    @pytest.mark.parametrize("number_of_solutions", [1, 2])
    def test_requested_variables(
        self, summary_solutions, number_of_variables, number_of_solutions
    ):
        plt = pybamm.import_optional_dependency("matplotlib.pyplot")
        output_variables = [
            "Throughput capacity [A.h]",
            "Loss of lithium inventory [%]",
        ][:number_of_variables]
        solutions = summary_solutions[:number_of_solutions]
        try:
            axes = pybamm.plot_summary_variables(
                solutions[0] if number_of_solutions == 1 else solutions,
                output_variables=output_variables,
                show_plot=False,
            )
            if number_of_variables == 1:
                assert axes is plt.gcf().axes[0]
                assert len(axes.figure.axes) == 1
            else:
                assert axes.shape == (2,)

            for variable, axis in zip(
                output_variables, np.atleast_1d(axes).flat, strict=True
            ):
                assert axis.get_xlabel() == "Cycle number"
                assert axis.get_ylabel() == variable
                assert len(axis.get_lines()) == number_of_solutions
                for line, solution in zip(axis.get_lines(), solutions, strict=True):
                    np.testing.assert_array_equal(
                        line.get_xdata(), solution.summary_variables.cycle_number
                    )
                    np.testing.assert_allclose(
                        line.get_ydata(), solution.summary_variables[variable]
                    )
        finally:
            plt.close("all")

    def test_plot(self):
        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/10 for 10 hours or until 3.3 V",
                    "Rest for 1 hour",
                    "Charge at 1 A until 4.1 V",
                    "Hold at 4.1 V until 50 mA",
                    "Rest for 1 hour",
                )
            ]
            * 3,
        )
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
        sim = pybamm.Simulation(
            model, experiment=experiment, parameter_values=parameter_values
        )
        sol = sim.solve(initial_soc=1)

        axes = pybamm.plot_summary_variables(sol, show_plot=False)

        axes = axes.flatten()
        assert len(axes) == 9

        for output_var, ax in zip(output_variables, axes, strict=False):
            assert ax.get_xlabel() == "Cycle number"
            assert ax.get_ylabel() == output_var

            cycle_number, var = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(
                cycle_number, sol.summary_variables.cycle_number
            )
            np.testing.assert_allclose(sol.summary_variables[output_var], var)

        axes = pybamm.plot_summary_variables(
            [sol, sol], labels=["SPM", "SPM"], show_plot=False
        )

        axes = axes.flatten()
        assert len(axes) == 9

        for output_var, ax in zip(output_variables, axes, strict=False):
            assert ax.get_xlabel() == "Cycle number"
            assert ax.get_ylabel() == output_var

            cycle_number, var = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(
                cycle_number, sol.summary_variables.cycle_number
            )
            np.testing.assert_allclose(sol.summary_variables[output_var], var)

            cycle_number, var = ax.get_lines()[1].get_data()
            np.testing.assert_array_equal(
                cycle_number, sol.summary_variables.cycle_number
            )
            np.testing.assert_allclose(sol.summary_variables[output_var], var)
