#
# Compare basic models with full models
#
import numpy as np

import pybamm


class TestCompareBasicModels:
    def test_compare_dfns(self):
        parameter_values = pybamm.ParameterValues("Ecker2015")
        basic_dfn = pybamm.lithium_ion.BasicDFN()
        dfn = pybamm.lithium_ion.DFN()

        # Solve basic DFN
        solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
        basic_sim = pybamm.Simulation(
            basic_dfn, parameter_values=parameter_values, solver=solver
        )
        t_eval = [0, 3600]
        t_interp = np.linspace(0, 3600)
        basic_sim.solve(t_eval=t_eval, t_interp=t_interp)
        basic_sol = basic_sim.solution

        # Solve main DFN
        sim = pybamm.Simulation(dfn, parameter_values=parameter_values, solver=solver)
        sim.solve(t_eval=t_eval, t_interp=t_interp)
        sol = sim.solution

        # Compare solution data
        np.testing.assert_allclose(basic_sol.t, sol.t)
        # Compare variables
        for name in basic_dfn.variables:
            np.testing.assert_allclose(
                basic_sol[name].entries, sol[name].entries, rtol=2e-3
            )

    def test_compare_dfns_composite(self):
        basic_dfn = pybamm.lithium_ion.BasicDFNComposite()
        dfn = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "open-circuit potential": (("single", "current sigmoid"), "single"),
            }
        )
        parameter_values = pybamm.ParameterValues("Chen2020_composite")

        # Solve basic DFN
        solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
        basic_sim = pybamm.Simulation(
            basic_dfn, parameter_values=parameter_values, solver=solver
        )
        t_eval = [0, 3600]
        t_interp = np.linspace(0, 3600)
        basic_sim.solve(t_eval=t_eval, t_interp=t_interp)
        basic_sol = basic_sim.solution

        # Solve main DFN
        sim = pybamm.Simulation(dfn, parameter_values=parameter_values)
        t_eval = np.linspace(0, 3600)
        sim.solve(t_eval=t_eval, t_interp=t_interp)
        sol = sim.solution

        # Compare solution data
        np.testing.assert_allclose(basic_sol.t, sol.t)
        # Compare variables
        for name in basic_dfn.variables:
            np.testing.assert_allclose(
                basic_sol[name].entries, sol[name].entries, rtol=1e-2, atol=1e-6
            )

    def test_compare_spms(self):
        parameter_values = pybamm.ParameterValues("Ecker2015")
        basic_spm = pybamm.lithium_ion.BasicSPM()
        spm = pybamm.lithium_ion.SPM()

        # Solve basic SPM
        solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
        basic_sim = pybamm.Simulation(
            basic_spm, parameter_values=parameter_values, solver=solver
        )
        t_eval = [0, 3600]
        t_interp = np.linspace(0, 3600)
        basic_sim.solve(t_eval=t_eval, t_interp=t_interp)
        basic_sol = basic_sim.solution

        # Solve main SPM
        sim = pybamm.Simulation(spm, parameter_values=parameter_values, solver=solver)
        sim.solve(t_eval=t_eval, t_interp=t_interp)
        sol = sim.solution

        # Compare solution data
        np.testing.assert_allclose(basic_sol.t, sol.t)
        # Compare variables
        for name in basic_spm.variables:
            np.testing.assert_allclose(
                basic_sol[name].entries, sol[name].entries, rtol=5e-4
            )
