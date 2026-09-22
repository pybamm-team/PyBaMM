#
# Test basic model classes
#
import numpy as np
import pytest

import pybamm


def lithium_per_area(model, electrolyte_domains, particles):
    """Lithium inventory per unit electrode area [mol.m-2], from the model's own
    concentration variables. ``particles`` is a list of
    (particle concentration variable name, active material volume fraction
    parameter name, electrode thickness parameter name)."""
    n = pybamm.Scalar(0)
    for domain, porosity, thickness in electrolyte_domains:
        c_e = model.variables[f"{domain} electrolyte concentration [mol.m-3]"]
        n += pybamm.x_average(c_e * pybamm.Parameter(porosity)) * pybamm.Parameter(
            thickness
        )
    for c_s_name, eps_s, thickness in particles:
        c_s = model.variables[c_s_name]
        n += pybamm.x_average(
            pybamm.Parameter(eps_s) * pybamm.r_average(c_s)
        ) * pybamm.Parameter(thickness)
    return n


def coarse_unstructured_var_pts(model):
    """Coarse mesh for the unstructured DFN models: 5 through-cell points per
    domain, 10 radial points and 3 points in each transverse direction."""
    var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_p": 10, "r_n": 10}
    for var in model.default_var_pts:
        if not isinstance(var, str):
            var_pts[var] = 3
    return var_pts


class TestElectrolyteConservation:
    """The electrolyte balance must conserve lithium when the transference number
    depends on concentration (issue #5745). The check is on the total lithium
    inventory, which is constant to solver tolerance for a conservative
    formulation and drifts by ~1e-2 relative for the non-conservative one."""

    def test_basic_dfn(self):
        model = pybamm.lithium_ion.BasicDFN()
        model.variables["Lithium per area [mol.m-2]"] = lithium_per_area(
            model,
            [
                (
                    "Negative",
                    "Negative electrode porosity",
                    "Negative electrode thickness [m]",
                ),
                ("Separator", "Separator porosity", "Separator thickness [m]"),
                (
                    "Positive",
                    "Positive electrode porosity",
                    "Positive electrode thickness [m]",
                ),
            ],
            [
                (
                    "Negative particle concentration [mol.m-3]",
                    "Negative electrode active material volume fraction",
                    "Negative electrode thickness [m]",
                ),
                (
                    "Positive particle concentration [mol.m-3]",
                    "Positive electrode active material volume fraction",
                    "Positive electrode thickness [m]",
                ),
            ],
        )
        # ORegan2022 has a concentration-dependent transference number
        parameter_values = pybamm.ParameterValues("ORegan2022")
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
        )
        sol = sim.solve([0, 1200], initial_soc=0.5)
        li = sol["Lithium per area [mol.m-2]"].entries
        np.testing.assert_allclose(li, li[0], rtol=1e-8)

    def test_basic_dfn_composite(self):
        model = pybamm.lithium_ion.BasicDFNComposite()
        model.variables["Lithium per area [mol.m-2]"] = lithium_per_area(
            model,
            [
                (
                    "Negative",
                    "Negative electrode porosity",
                    "Negative electrode thickness [m]",
                ),
                ("Separator", "Separator porosity", "Separator thickness [m]"),
                (
                    "Positive",
                    "Positive electrode porosity",
                    "Positive electrode thickness [m]",
                ),
            ],
            [
                (
                    "Negative primary particle concentration [mol.m-3]",
                    "Primary: Negative electrode active material volume fraction",
                    "Negative electrode thickness [m]",
                ),
                (
                    "Negative secondary particle concentration [mol.m-3]",
                    "Secondary: Negative electrode active material volume fraction",
                    "Negative electrode thickness [m]",
                ),
                (
                    "Positive particle concentration [mol.m-3]",
                    "Positive electrode active material volume fraction",
                    "Positive electrode thickness [m]",
                ),
            ],
        )
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        parameter_values["Cation transference number"] = pybamm.ParameterValues(
            "ORegan2022"
        )["Cation transference number"]
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
        )
        sol = sim.solve([0, 1200])
        li = sol["Lithium per area [mol.m-2]"].entries
        np.testing.assert_allclose(li, li[0], rtol=1e-8)

    @pytest.mark.parametrize(
        "model",
        [
            pybamm.lithium_ion.BasicDFN2DUnstructured(),
            pybamm.lithium_ion.BasicDFN3DUnstructured(),
        ],
        ids=["2d_unstructured", "3d_unstructured"],
    )
    def test_basic_dfn_unstructured(self, model):
        model.variables["Total lithium inventory [mol]"] = (
            model.variables["Total lithium [mol]"]
            + model.variables["Total solid lithium [mol]"]
        )
        parameter_values = pybamm.ParameterValues("ORegan2022")
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            var_pts=coarse_unstructured_var_pts(model),
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
        )
        sol = sim.solve([0, 1200], initial_soc=0.5)
        li = sol["Total lithium inventory [mol]"].entries
        np.testing.assert_allclose(li, li[0], rtol=1e-8)


class BaseBasicModelTest:
    def test_with_experiment(self):
        model = self.model
        experiment = pybamm.Experiment(
            [
                "Discharge at C/3 until 3.5V",
                "Hold at 3.5V for 1 hour",
                "Rest for 10 min",
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve(calc_esoh=False)

        # Check the solve returned a solution
        assert sol is not None

        # Check that the solution contains the expected number of cycles
        assert len(sol.cycles) == 3

        # Check that the solution terminated because it reached final time
        assert sol.termination == "final time"


class TestBasicSPM(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.BasicSPM()


class TestBasicDFN(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.BasicDFN()


class TestBasicDFNComposite(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.BasicDFNComposite()


class TestBasicDFNHalfCell(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        options = {"working electrode": "positive"}
        self.model = pybamm.lithium_ion.BasicDFNHalfCell(options)


class TestBasicDFN2DUnstructured(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.BasicDFN2DUnstructured(element_type="quad")

    def test_matches_structured(self):
        t_eval = np.linspace(0, 3600, 20)
        var_pts = coarse_unstructured_var_pts(self.model)

        model_s = pybamm.lithium_ion.BasicDFN2D()
        sol_s = pybamm.Simulation(model_s, var_pts=var_pts).solve(t_eval)
        sol_u = pybamm.Simulation(self.model, var_pts=var_pts).solve(t_eval)

        V_s = sol_s["Voltage [V]"](t=t_eval)
        V_u = sol_u["Voltage [V]"](t=t_eval)
        np.testing.assert_allclose(V_u, V_s, atol=5e-3)

        # The electrolyte current density is a two-component vector field
        i_e_u = sol_u["Electrolyte current density [A.m-2]"]
        assert len(i_e_u.entries) == 2
        assert all(np.all(np.isfinite(component)) for component in i_e_u.entries)


class TestBasicDFN3DUnstructured(BaseBasicModelTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.BasicDFN3DUnstructured()

    def test_matches_2d(self):
        t_eval = np.linspace(0, 3600, 20)

        model_2d = pybamm.lithium_ion.BasicDFN2DUnstructured(element_type="quad")
        sim_2d = pybamm.Simulation(
            model_2d, var_pts=coarse_unstructured_var_pts(model_2d)
        )
        sol_2d = sim_2d.solve(t_eval)

        sim_3d = pybamm.Simulation(
            self.model, var_pts=coarse_unstructured_var_pts(self.model)
        )
        sol_3d = sim_3d.solve(t_eval)

        V_2d = sol_2d["Voltage [V]"](t=t_eval)
        V_3d = sol_3d["Voltage [V]"](t=t_eval)
        # The 3D model is uniform in y, so it reproduces the 2D solution
        np.testing.assert_allclose(V_3d, V_2d, atol=1e-4)

        # Both models report the lithium of the whole cell, in mol
        for name in ["Total lithium [mol]", "Total solid lithium [mol]"]:
            np.testing.assert_allclose(
                sol_3d[name](t=t_eval), sol_2d[name](t=t_eval), rtol=1e-6
            )

        N_e = sol_3d["Electrolyte flux [mol.m-2.s-1]"]
        assert len(N_e.entries) == 3
