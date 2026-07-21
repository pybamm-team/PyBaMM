import numpy as np
import pytest

import pybamm


class TestThermal:
    def test_consistent_cooling(self):
        "Test the cooling is consistent between the 1D, 1+1D and 2+1D SPMe models"

        # Load models
        options = {"thermal": "lumped"}
        spme_1D = pybamm.lithium_ion.SPMe(options=options)
        options = {
            "thermal": "x-lumped",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        spme_1p1D = pybamm.lithium_ion.SPMe(options=options)
        options = {
            "thermal": "x-lumped",
            "current collector": "potential pair",
            "dimensionality": 2,
        }
        spme_2p1D = pybamm.lithium_ion.SPMe(options=options)
        models = {"SPMe 1D": spme_1D, "SPMe 1+1D": spme_1p1D, "SPMe 2+1D": spme_2p1D}

        # Set up parameter values
        parameter_values = pybamm.ParameterValues("NCA_Kim2011")
        C_rate = 5
        h_cn = 10
        h_cp = 5
        h_tab_n = 250
        h_tab_p = 250
        h_edge = 100
        # for the lumped model, the total heat transfer coefficient is the area-weighted
        # average of the heat transfer coefficients
        param = spme_1D.param
        L = param.L
        L_y = param.L_y
        L_z = param.L_z
        L_tab_n = param.n.L_tab
        L_tab_p = param.p.L_tab
        L_cn = param.n.L_cc
        L_cp = param.p.L_cc

        h_total = (
            h_cn * L_y * L_z
            + h_cp * L_y * L_z
            + h_tab_n * L_tab_n * L_cn
            + h_tab_p * L_tab_p * L_cp
            + h_edge * (2 * L_y * L + 2 * L_z * L - L_tab_n * L_cn - L_tab_p * L_cp)
        ) / (2 * L_y * L_z + 2 * L_y * L + 2 * L_z * L)

        parameter_values.update(
            {
                "Negative current collector"
                + " surface heat transfer coefficient [W.m-2.K-1]": h_cn,
                "Positive current collector"
                + " surface heat transfer coefficient [W.m-2.K-1]": h_cp,
                "Negative tab heat transfer coefficient [W.m-2.K-1]": h_tab_n,
                "Positive tab heat transfer coefficient [W.m-2.K-1]": h_tab_p,
                "Edge heat transfer coefficient [W.m-2.K-1]": h_edge,
                "Total heat transfer coefficient [W.m-2.K-1]": h_total,
                # Set high thermal and electrical conductivity in current collectors
                "Negative current collector"
                + " thermal conductivity [W.m-1.K-1]": 267.467 * 100000,
                "Positive current collector"
                + " thermal conductivity [W.m-1.K-1]": 158.079 * 100000,
                "Negative current collector conductivity [S.m-1]": 1e10,
                "Positive current collector conductivity [S.m-1]": 1e10,
            }
        )

        # Solve models
        solutions = {}
        var_pts = {"x_n": 3, "x_s": 3, "x_p": 3, "r_n": 3, "r_p": 3, "y": 5, "z": 5}
        t_eval = [0, 3500 / 6]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 100)
        for model_name, model in models.items():
            solver = pybamm.IDAKLUSolver()
            sim = pybamm.Simulation(
                model,
                var_pts=var_pts,
                solver=solver,
                parameter_values=parameter_values,
                C_rate=C_rate,
            )
            sim.solve(t_eval)

            solutions[model_name] = sim.solution[
                "Volume-averaged cell temperature [K]"
            ](t_interp)

        # check volume-averaged cell temperature is within
        # 1e-5 relative error

        def err(a, b):
            return np.max(np.abs(a - b)) / np.max(np.abs(a))

        assert 1e-5 > err(solutions["SPMe 1+1D"], solutions["SPMe 2+1D"])

    def test_surface_temperature_models(self):
        models = {
            option: pybamm.lithium_ion.SPM(
                {"thermal": "lumped", "surface temperature": option}
            )
            for option in ["lumped", "ambient"]
        }

        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update(
            {
                "Casing heat capacity [J.K-1]": 30,
                "Environment thermal resistance [K.W-1]": 10,
            }
        )

        sols = {}
        t_eval = [0, 3600]
        t_interp = np.linspace(0, 3600, 100)
        for name, model in models.items():
            sim = pybamm.Simulation(model, parameter_values=parameter_values)
            sol = sim.solve(t_eval=t_eval, t_interp=t_interp)
            sols[name] = sol

        for var in ["Volume-averaged cell temperature [K]", "Surface temperature [K]"]:
            # ignore first entry as it is the initial condition
            T_ambient_model = sols["ambient"][var].entries[1:]
            T_lumped_model = sols["lumped"][var].entries[1:]
            np.testing.assert_array_less(T_ambient_model, T_lumped_model)

    def test_lumped_contact_resistance(self):
        # Test that the heating with contact resistance is greater than without

        # load models
        model_no_contact_resistance = pybamm.lithium_ion.SPMe(
            {
                "cell geometry": "arbitrary",
                "thermal": "lumped",
                "contact resistance": "false",
            }
        )
        model_contact_resistance = pybamm.lithium_ion.SPMe(
            {
                "cell geometry": "arbitrary",
                "thermal": "lumped",
                "contact resistance": "true",
            }
        )
        models = [model_no_contact_resistance, model_contact_resistance]

        # parameters
        parameter_values = pybamm.ParameterValues("Marquis2019")
        lumped_params = parameter_values.copy()
        lumped_params_contact_resistance = parameter_values.copy()

        lumped_params_contact_resistance.update(
            {
                "Contact resistance [Ohm]": 0.05,
            }
        )

        # solve the models
        params = [lumped_params, lumped_params_contact_resistance]
        sols = []
        for model, param in zip(models, params, strict=False):
            sim = pybamm.Simulation(model, parameter_values=param)
            sim.solve([0, 3600])
            sols.append(sim.solution)

        # get the average temperature from each model
        avg_cell_temp = sols[0]["X-averaged cell temperature [K]"].entries
        avg_cell_temp_cr = sols[1]["X-averaged cell temperature [K]"].entries

        # Lumped thermal with contact resistance > without; skip first entry (same IC)
        np.testing.assert_array_less(avg_cell_temp[1:], avg_cell_temp_cr[1:])

    @pytest.mark.parametrize("thermal", ["isothermal", "lumped", "x-full"])
    def test_temperature_dependent_contact_resistance(self, thermal):
        # A temperature-dependent "Contact resistance [Ohm]" should work for all
        # thermal options and match the constant value at the reference temperature.

        R_ref = 0.05

        def R_contact(T):
            # equal to R_ref at T_ref = 298.15 K, increasing with temperature
            return R_ref * (1 + 0.01 * (T - 298.15))

        parameter_values = pybamm.ParameterValues("Marquis2019")
        options = {"thermal": thermal, "contact resistance": "true"}

        # constant value
        const_params = parameter_values.copy()
        const_params.update({"Contact resistance [Ohm]": R_ref})
        sim_const = pybamm.Simulation(
            pybamm.lithium_ion.SPMe(options), parameter_values=const_params
        )
        sol_const = sim_const.solve([0, 3600])

        # temperature-dependent value
        fn_params = parameter_values.copy()
        fn_params.update({"Contact resistance [Ohm]": R_contact})
        sim_fn = pybamm.Simulation(
            pybamm.lithium_ion.SPMe(options), parameter_values=fn_params
        )
        sol_fn = sim_fn.solve([0, 3600])

        dphi_const = sol_const["Contact overpotential [V]"].entries
        dphi_fn = sol_fn["Contact overpotential [V]"].entries

        if thermal == "isothermal":
            # temperature is fixed at the reference, so the two agree
            np.testing.assert_allclose(dphi_const, dphi_fn, rtol=1e-6)
        else:
            # cell heats up above the reference, so R (and hence the contact
            # overpotential magnitude) is larger than the constant case
            np.testing.assert_array_less(np.abs(dphi_const[1:]), np.abs(dphi_fn[1:]))
