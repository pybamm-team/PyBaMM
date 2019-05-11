#
# Tests comparing model outputs for standard variables
#
import pybamm
import numpy as np


class StandardOutputComparison(object):
    "Calls all the tests comparing standard output variables."

    def __init__(self, models, parameter_values, discs, solvers):
        self.models = models
        self.parameter_values = parameter_values
        self.discs = discs
        self.solvers = solvers

        if isinstance(self.models[0], pybamm.LithiumIonBaseModel):
            self.chemistry = "Lithium-ion"
        elif isinstance(self.models[0], pybamm.LeadAcidBaseModel):
            self.chemistry = "Lead acid"

    def run_test_class(self, ClassName):
        "Run all tests from a class 'ClassName'"
        tests = ClassName(self.models, self.parameter_values, self.discs, self.solvers)
        # tests.test_all()

    def test_all(self):
        self.run_test_class(BaseOutputComparison)
        # self.run_test_class(VoltageComparison)
        # self.run_test_class(ElectrolyteConcentrationComparison)
        # self.run_test_class(PotentialComparison)
        # self.run_test_class(CurrentComparison)
        #
        # if self.chemistry == "Lithium-ion":
        #     self.run_test_class(ParticleConcentrationComparison)


class BaseOutputComparison(object):
    def __init__(self, models, param, discs, solvers):
        self.models = models
        self.discs = discs
        self.solvers = solvers

        self.set_output_times()
        self.set_meshes()

    def set_output_times(self):
        # Get max time allowed from the simulation, i.e. the smallest end time common
        # to all the solvers
        max_t = min([solver.t[-1] for solver in self.solvers.values()])

        # Assign common time
        solver0 = self.solvers[self.models[0]]
        max_index = np.where(solver0.t == max_t)[0][0]
        self.t = solver0.t[:max_index]

        # Check times
        for model in self.models:
            np.testing.assert_array_equal(self.t, self.solvers[model].t[:max_index])

    def set_meshes(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        disc0 = self.discs[self.models[0]]

        # Assign
        self.x_n = disc0.mesh["negative electrode"][0].nodes
        self.x_s = disc0.mesh["separator"][0].nodes
        self.x_p = disc0.mesh["positive electrode"][0].nodes
        self.x = disc0.mesh.combine_submeshes(*whole_cell)[0].nodes
        self.x_n_edge = disc0.mesh["negative electrode"][0].edges
        self.x_s_edge = disc0.mesh["separator"][0].edges
        self.x_p_edge = disc0.mesh["positive electrode"][0].edges
        self.x_edge = disc0.mesh.combine_submeshes(*whole_cell)[0].edges

        if isinstance(self.models[0], pybamm.LithiumIonBaseModel):
            self.r_n = disc0.mesh["negative particle"][0].nodes
            self.r_p = disc0.mesh["positive particle"][0].nodes
            self.r_n_edge = disc0.mesh["negative particle"][0].edges
            self.r_p_edge = disc0.mesh["positive particle"][0].edges

        # Check meshes from other models
        check_mesh = np.testing.assert_array_equal
        for model in self.models[1:]:
            disc = self.discs[model]
            check_mesh(self.x_n, disc.mesh["negative electrode"][0].nodes)
            check_mesh(self.x_s, disc.mesh["separator"][0].nodes)
            check_mesh(self.x_p, disc.mesh["positive electrode"][0].nodes)
            check_mesh(self.x, disc.mesh.combine_submeshes(*whole_cell)[0].nodes)
            check_mesh(self.x_n_edge, disc.mesh["negative electrode"][0].edges)
            check_mesh(self.x_s_edge, disc.mesh["separator"][0].edges)
            check_mesh(self.x_p_edge, disc.mesh["positive electrode"][0].edges)
            check_mesh(self.x_edge, disc.mesh.combine_submeshes(*whole_cell)[0].edges)

            if isinstance(self.models[0], pybamm.LithiumIonBaseModel):
                check_mesh(self.r_n, disc.mesh["negative particle"][0].nodes)
                check_mesh(self.r_p, disc.mesh["positive particle"][0].nodes)
                check_mesh(self.r_n_edge, disc.mesh["negative particle"][0].edges)
                check_mesh(self.r_p_edge, disc.mesh["positive particle"][0].edges)

    def get_vars(self, var):
        "Helper function to reduce repeated code."
        return [
            pybamm.ProcessedVariable(
                model.variables[var],
                self.t,
                self.solvers[model].y,
                mesh=self.discs[model].mesh,
            )
            for model in self.models
        ]


class VoltageComparison(BaseOutputComparison):
    def __init__(self, model, param, disc, solver, operating_condition):
        super().__init__(model, param, disc, solver, operating_condition)

        self.eta_n = self.get_var("Negative reaction overpotential [V]")
        self.eta_p = self.get_var("Positive reaction overpotential [V]")
        self.eta_r_av = self.get_var("Average reaction overpotential [V]")

        self.eta_e_av = self.get_var("Average electrolyte overpotential [V]")
        self.Delta_Phi_s_av = self.get_var("Average solid phase ohmic losses [V]")

        self.ocp_n_av = self.get_var(
            "Average negative electrode open circuit potential [V]"
        )
        self.ocp_p_av = self.get_var(
            "Average positive electrode open circuit potential [V]"
        )
        self.ocv_av = self.get_var("Average open circuit voltage [V]")
        self.voltage = self.get_var("Terminal voltage [V]")

    def test_each_reaction_overpotential(self):
        """Testing that:
            - discharge: eta_n > 0, eta_p < 0
            - charge: eta_n < 0, eta_p > 0
            - off: eta_n == 0, eta_p == 0
            """
        tol = 0.001
        t, x_n, x_p = self.t, self.x_n, self.x_p
        if self.operating_condition == "discharge":
            np.testing.assert_array_less(-self.eta_n(t, x_n), tol)
            np.testing.assert_array_less(self.eta_p(t, x_p), tol)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(self.eta_n(t, x_n), tol)
            np.testing.assert_array_less(-self.eta_p(t, x_p), tol)
        elif self.operating_condition == "off":
            np.testing.assert_array_equal(self.eta_n(t, x_n), 0)
            np.testing.assert_array_equal(-self.eta_p(t, x_p), 0)

    def test_overpotentials(self):
        """Testing that all are:
            - discharge: . < 0
            - charge: . > 0
            - off: . == 0
        """
        tol = 0.001
        if self.operating_condition == "discharge":
            np.testing.assert_array_less(self.eta_r_av(self.t), tol)
            np.testing.assert_array_less(self.eta_e_av(self.t), tol)
            np.testing.assert_array_less(self.Delta_Phi_s_av(self.t), tol)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(-self.eta_r_av(self.t), tol)
            np.testing.assert_array_less(-self.eta_e_av(self.t), tol)
            np.testing.assert_array_less(-self.Delta_Phi_s_av(self.t), tol)

        elif self.operating_condition == "off":
            np.testing.assert_array_equal(self.eta_r_av(self.t), 0)
            np.testing.assert_array_equal(self.eta_e_av(self.t), 0)
            np.testing.assert_array_equal(self.Delta_Phi_s_av(self.t), 0)

    def test_ocps(self):
        """ Testing that:
            - discharge: ocp_n increases, ocp_p decreases
            - charge: ocp_n decreases, ocp_p increases
            - off: ocp_n, ocp_p constant
        """
        neg_end_vs_start = self.ocp_n_av(self.t[-1]) - self.ocp_n_av(self.t[1])
        pos_end_vs_start = self.ocp_p_av(self.t[-1]) - self.ocp_p_av(self.t[1])
        if self.operating_condition == "discharge":
            np.testing.assert_array_less(-neg_end_vs_start, 0)
            np.testing.assert_array_less(pos_end_vs_start, 0)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(neg_end_vs_start, 0)
            np.testing.assert_array_less(-pos_end_vs_start, 0)
        elif self.operating_condition == "off":
            np.testing.assert_array_almost_equal(neg_end_vs_start, 0)
            np.testing.assert_array_almost_equal(pos_end_vs_start, 0)

    def test_ocv(self):
        """Testing that:
            - discharge: ocv decreases
            - charge: ocv increases
            - off: ocv constant
        """

        end_vs_start = self.ocv_av(self.t[-1]) - self.ocv_av(self.t[1])

        if self.operating_condition == "discharge":
            np.testing.assert_array_less(end_vs_start, 0)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(-end_vs_start, 0)
        elif self.operating_condition == "off":
            np.testing.assert_array_almost_equal(end_vs_start, 0)

    def test_voltage(self):
        """Testing that:
            - discharge: voltage decreases
            - charge: voltage increases
            - off: voltage constant
        """
        end_vs_start = self.voltage(self.t[-1]) - self.voltage(self.t[1])

        if self.operating_condition == "discharge":
            np.testing.assert_array_less(end_vs_start, 0)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(-end_vs_start, 0)
        elif self.operating_condition == "off":
            np.testing.assert_array_almost_equal(end_vs_start, 0)

    def test_consistent(self):
        """Test voltage components are consistent with one another by ensuring they sum
        correctly"""

        np.testing.assert_array_almost_equal(
            self.ocv_av(self.t), self.ocp_p_av(self.t) - self.ocp_n_av(self.t)
        )

        np.testing.assert_array_almost_equal(
            self.voltage(self.t),
            self.ocv_av(self.t)
            + self.eta_r_av(self.t)
            + self.eta_e_av(self.t)
            + self.Delta_Phi_s_av(self.t),
            decimal=3,
        )

    def test_all(self):
        self.test_each_reaction_overpotential()
        self.test_overpotentials()
        self.test_ocps()
        self.test_ocv()
        self.test_voltage()
        self.test_consistent()


class ParticleConcentrationComparison(BaseOutputComparison):
    def __init__(self, model, param, disc, solver, operating_condition):
        super().__init__(model, param, disc, solver, operating_condition)

        self.c_s_n = self.get_var("Negative particle concentration")
        self.c_s_p = self.get_var("Positive particle concentration")

        self.c_s_n_surf = self.get_var("Negative particle surface concentration")
        self.c_s_p_surf = self.get_var("Positive particle surface concentration")

        self.N_s_n = self.get_var("Negative particle flux")
        self.N_s_p = self.get_var("Positive particle flux")

    def test_concentration_increase_decrease(self):
        """Test all concentrations in negative particles decrease and all
        concentrations in positive particles increase over a discharge."""

        t, x_n, x_p, r_n, r_p = self.t, self.x_n, self.x_p, self.r_n, self.r_p

        neg_end_vs_start = self.c_s_n(t[1:], x_n, r_n) - self.c_s_n(t[:-1], x_n, r_n)
        pos_end_vs_start = self.c_s_p(t[1:], x_p, r_p) - self.c_s_p(t[:-1], x_p, r_p)

        if self.operating_condition == "discharge":
            np.testing.assert_array_less(neg_end_vs_start, 0)
            np.testing.assert_array_less(-pos_end_vs_start, 0)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(-neg_end_vs_start, 0)
            np.testing.assert_array_less(pos_end_vs_start, 0)
        elif self.operating_condition == "off":
            np.testing.assert_array_almost_equal(neg_end_vs_start, 0)
            np.testing.assert_array_almost_equal(pos_end_vs_start, 0)

    def test_concentration_limits(self):
        "Test that concentrations do not go below 0 or exceed the maximum."
        t, x_n, x_p, r_n, r_p = self.t, self.x_n, self.x_p, self.r_n, self.r_p

        np.testing.assert_array_less(-self.c_s_n(t, x_n, r_n), 0)
        np.testing.assert_array_less(-self.c_s_p(t, x_p, r_p), 0)

        np.testing.assert_array_less(self.c_s_n(t, x_n, r_n), 1)
        np.testing.assert_array_less(self.c_s_p(t, x_p, r_p), 1)

    def test_conservation(self):
        "Test amount of lithium stored across all particles is constant."
        # TODO: add an output for total lithium in particles

    def test_concentration_profile(self):
        """Test that the concentration in the centre of the negative particles is
        greater than the average concentration in the particle and also that the
        concentration on the surface of the negative particle is less than the average
        concentration in the particle. Test opposite is true for the positive
        particle."""
        # TODO: add an output for average particle concentration

    def test_fluxes(self):
        """Test that no flux holds in the centre of the particle. Test that surface
        flux in the negative particles is greater than zero and that the flux in the
        positive particles is less than zero during a discharge."""
        t, x_n, x_p, r_n, r_p = self.t, self.x_n, self.x_p, self.r_n_edge, self.r_p_edge

        if self.operating_condition == "discharge":
            np.testing.assert_array_less(0, self.N_s_n(t[1:], x_n, r_n[1:]))
            np.testing.assert_array_less(self.N_s_p(t[1:], x_p, r_p[1:]), 0)
        if self.operating_condition == "charge":
            np.testing.assert_array_less(self.N_s_n(t[1:], x_n, r_n[1:]), 0)
            np.testing.assert_array_less(0, self.N_s_p(t[1:], x_p, r_p[1:]))
        if self.operating_condition == "off":
            np.testing.assert_array_almost_equal(self.N_s_n(t, x_n, r_n), 0)
            np.testing.assert_array_almost_equal(self.N_s_p(t, x_p, r_p), 0)

        np.testing.assert_array_equal(0, self.N_s_n(t, x_n, r_n[0]))
        np.testing.assert_array_equal(0, self.N_s_p(t, x_p, r_p[0]))

    def test_all(self):
        self.test_concentration_increase_decrease()
        self.test_concentration_limits()
        self.test_conservation()
        self.test_concentration_profile()
        self.test_fluxes()


class ElectrolyteConcentrationComparison(BaseOutputComparison):
    def __init__(self, model, param, disc, solver, operating_condition):
        super().__init__(model, param, disc, solver, operating_condition)

        self.c_e = self.get_var("Electrolyte concentration")

        self.c_e_n = self.get_var("Negative electrolyte concentration")
        self.c_e_s = self.get_var("Separator electrolyte concentration")
        self.c_e_p = self.get_var("Positive electrolyte concentration")

        # TODO: output average electrolyte concentration
        # self.c_e_av = self.get_var("Average electrolyte concentration")
        # self.c_e_n_av = self.get_var("Average negative electrolyte concentration")
        # self.c_e_s_av = self.get_var("Average separator electrolyte concentration")
        # self.c_e_p_av = self.get_var("Average positive electrolyte concentration")

        # self.N_e = self.get_var("Electrolyte flux")
        self.N_e_hat = self.get_var("Reduced cation flux")

    def test_concentration_limit(self):
        "Test that the electrolyte concentration is always greater than zero."
        np.testing.assert_array_less(-self.c_e(self.t, self.x), 0)

    def test_conservation(self):
        "Test conservation of species in the electrolyte."
        # sufficient to check average concentration is constant

        # diff = self.c_e_av.entries[:, 1:] - self.c_e_av.entries[:, :-1]
        # np.testing.assert_array_almost_equal(diff, 0)

    def test_concentration_profile(self):
        """Test continuity of the concentration profile. Test average concentration is
        as expected and that the concentration in the negative electrode is greater
        than the average and the concentration in the positive is less than the average
        during a discharge."""

        # TODO: uncomment when have average concentrations
        # # small number so that can use array less
        # epsilon = 0.001

        # if self.operating_condition == "discharge":
        #     np.testing.assert_array_less(
        #         -self.c_e_n_av.entries, self.c_e_av.entries + epsilon
        #     )
        #     np.testing.assert_array_less(
        #         self.c_e_p_av.entries, self.c_e_av.entries + epsilon
        #     )
        # elif self.operating_condition == "charge":
        #     np.testing.assert_array_less(
        #         -self.c_e_n_av.entries, self.c_e_av.entries + epsilon
        #     )
        #     np.testing.assert_array_less(
        #         self.c_e_p_av.entries, self.c_e_av.entries + epsilon
        #     )
        # elif self.operating_condition == "off":
        #     np.testing.assert_array_equal(self.c_e_n_av.entries, self.c_e_av.entries)
        #     np.testing.assert_array_equal(self.c_e_s_av.entries, self.c_e_av.entries)
        #     np.testing.assert_array_equal(self.c_e_p_av.entries, self.c_e_av.entries)

    def test_fluxes(self):
        """Test that the internal boundary fluxes are continuous. Test current
        collector fluxes are zero."""
        t, x = self.t, self.x_edge
        np.testing.assert_array_equal(self.N_e_hat(t, x[0]), 0)
        np.testing.assert_array_equal(self.N_e_hat(t, x[-1]), 0)

    def test_splitting(self):
        """Test that when splitting the concentrations and fluxes by negative electrode,
        separator, and positive electrode, we get the correct behaviour: continuous
        solution and recover combined through concatenation."""
        t, x_n, x_s, x_p, x = self.t, self.x_n, self.x_s, self.x_p, self.x
        c_e_combined = np.concatenate(
            (self.c_e_n(t, x_n), self.c_e_s(t, x_s), self.c_e_p(t, x_p)), axis=0
        )

        np.testing.assert_array_equal(self.c_e(t, x), c_e_combined)

    def test_all(self):
        self.test_concentration_limit()
        self.test_conservation()
        self.test_concentration_profile()
        self.test_fluxes()
        self.test_splitting()


class PotentialComparison(BaseOutputComparison):
    def __init__(self, model, param, disc, solver, operating_condition):
        super().__init__(model, param, disc, solver, operating_condition)

        self.phi_s_n = self.get_var("Negative electrode potential [V]")
        self.phi_s_p = self.get_var("Positive electrode potential [V]")

        self.phi_e = self.get_var("Electrolyte potential [V]")

        self.phi_e_n = self.get_var("Negative electrolyte potential [V]")
        self.phi_e_s = self.get_var("Separator electrolyte potential [V]")
        self.phi_e_p = self.get_var("Positive electrolyte potential [V]")

    def test_negative_electrode_potential_profile(self):
        """Test that negative electrode potential is zero on left boundary. Test
        average negative electrode potential is less than or equal to zero."""
        t, x, _ = self.phi_s_n.t_x_r_sol

        np.testing.assert_array_almost_equal(self.phi_s_n(t, x=0), 0, decimal=5)

    def test_positive_electrode_potential_profile(self):
        """Test average positive electrode potential is less than the positive electrode
        potential on the right current collector."""

        # TODO: add these when have averages

    def test_potential_differences(self):
        """Test electrolyte potential is less than the negative electrode potential.
        Test that the positive electrode potential is greater than the negative
        electrode potential."""

        # TODO: these tests with averages

        np.testing.assert_array_less(-self.phi_s_p(self.t, self.x_p), 0)

    def test_all(self):
        self.test_negative_electrode_potential_profile()
        self.test_positive_electrode_potential_profile()
        self.test_potential_differences()


class CurrentComparison(BaseOutputComparison):
    def __init__(self, model, param, disc, solver, operating_condition):
        super().__init__(model, param, disc, solver, operating_condition)

        self.j = self.get_var("Interfacial current density")
        self.j0 = self.get_var("Exchange-current density")

        self.j_n = self.get_var("Negative electrode interfacial current density")
        self.j_p = self.get_var("Positive electrode interfacial current density")
        self.j_n_av = self.get_var(
            "Average negative electrode interfacial current density"
        )
        self.j_p_av = self.get_var(
            "Average positive electrode interfacial current density"
        )

        self.j0_n = self.get_var("Negative electrode exchange-current density")
        self.j0_p = self.get_var("Positive electrode exchange-current density")

        self.i_s_n = self.get_var("Negative electrode current density")
        self.i_s_p = self.get_var("Positive electrode current density")

    def test_interfacial_current_average(self):
        """Test that average of the interfacial current density is equal to the true
        value."""
        np.testing.assert_array_almost_equal(
            self.j_n_av(self.t), self.i_cell / self.l_n
        )
        np.testing.assert_array_almost_equal(
            self.j_p_av(self.t), -self.i_cell / self.l_p, decimal=5
        )

    def test_conservation(self):
        """Test sum of electrode and electrolyte current densities give the applied
        current density"""

        # TODO: add a total function

    def test_current_density_boundaries(self):
        """Test the boundary values of the current densities"""
        t, x_n, x_p = self.t, self.x_n_edge, self.x_p_edge

        current_param = pybamm.electrical_parameters.current_with_time
        parameter_values = self.model.default_parameter_values
        i_cell = parameter_values.process_symbol(current_param).evaluate(t=t)
        np.testing.assert_array_almost_equal(self.i_s_n(t, x_n[0]), i_cell, decimal=4)
        np.testing.assert_array_almost_equal(self.i_s_n(t, x_n[-1]), 0, decimal=4)
        np.testing.assert_array_almost_equal(self.i_s_p(t, x_p[-1]), i_cell, decimal=4)
        np.testing.assert_array_almost_equal(self.i_s_p(t, x_p[0]), 0, decimal=4)

    def test_all(self):
        self.test_interfacial_current_average()
        self.test_conservation()
        self.test_current_density_boundaries()
