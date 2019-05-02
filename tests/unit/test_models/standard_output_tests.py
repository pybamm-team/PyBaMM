#
# Standard tests on the standard set of model outputs
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pybamm
import numpy as np


class StandardOutputTests(object):
    "Calls all the tests on the standard output variables."

    def __init__(self, model, disc, solver, parameter_values):
        self.model = model
        self.disc = disc
        self.solver = solver

        if isinstance(self.model, pybamm.LithiumIonBaseModel):
            self.chemistry = "Lithium-ion"
        elif isinstance(self.model, pybamm.LeadAcidBaseModel):
            self.chemistry = "Lead acid"

        current_sign = np.sign(parameter_values["Typical current density"])
        if current_sign == 1:
            self.operating_condition = "discharge"
        elif current_sign == -1:
            self.operating_condition = "charge"
        else:
            self.operating_condition = "off"

    def test_voltage(self):
        tests = VoltageTests(
            self.model, self.disc, self.solver, self.operating_condition
        )
        tests.test_all()

    def test_electrolyte(self):
        tests = ElectrolyteConcentrationTests(
            self.model, self.disc, self.solver, self.operating_condition
        )
        tests.test_all()

    def test_potentials(self):
        tests = PotentialTests(
            self.model, self.disc, self.solver, self.operating_condition
        )
        tests.test_all()

    def test_particles(self):
        tests = ParticleConcentrationTests(
            self.model, self.disc, self.solver, self.operating_condition
        )
        tests.test_all()

    def test_currents(self):
        tests = CurrentTests(
            self.model, self.disc, self.solver, self.operating_condition
        )
        tests.test_all()

    def test_all(self):
        self.test_voltage()
        self.test_electrolyte()
        self.test_potentials()
        self.test_currents()

        if self.chemistry == "Lithium-ion":
            self.test_particles()


class BaseOutputTest(object):
    def __init__(self, model, disc, solver, operating_condition):
        self.model = model
        self.disc = disc
        self.solver = solver
        self.operating_condition = operating_condition

    def get_var(self, var):
        "Helper function to reduce repeated code."
        return pybamm.ProcessedVariable(
            self.model.variables[var], self.solver.t, self.solver.y, mesh=self.disc.mesh
        )


class VoltageTests(BaseOutputTest):
    def __init__(self, model, disc, solver, operating_condition):
        super().__init__(model, disc, solver, operating_condition)

        self.eta_n = self.get_var("Negative reaction overpotential [V]")
        self.eta_p = self.get_var("Positive reaction overpotential [V]")
        self.eta_r_av = self.get_var("Average reaction overpotential [V]")
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
        if self.operating_condition == "discharge":
            np.testing.assert_array_less(-self.eta_n.entries, 0)
            np.testing.assert_array_less(self.eta_p.entries, 0)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(self.eta_n.entries, 0)
            np.testing.assert_array_less(-self.eta_p.entries, 0)
        elif self.operating_condition == "off":
            np.testing.assert_array_equal(self.eta_n.entries, 0)
            np.testing.assert_array_equal(-self.eta_p.entries, 0)

    def test_total_reaction_overpotential(self):
        """Testing that:
            - discharge: eta_r_av < 0
            - charge: eta_r_av > 0
            - off: eta_r_av == 0
        """
        if self.operating_condition == "discharge":
            np.testing.assert_array_less(self.eta_r_av.entries, 0)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(-self.eta_r_av.entries, 0)
        elif self.operating_condition == "off":
            np.testing.assert_array_equal(self.eta_r_av.entries, 0)

    def test_ocps(self):
        """ Testing that:
            - discharge: ocp_n increases, ocp_p decreases
            - charge: ocp_n decreases, ocp_p increases
            - off: ocp_n, ocp_p constant
        """

        neg_end_vs_start = self.ocp_n_av.entries[:, -1] - self.ocp_n_av.entries[:, 1]
        pos_end_vs_start = self.ocp_p_av.entries[:, -1] - self.ocp_p_av.entries[:, 1]
        if self.operating_condition == "discharge":
            np.testing.assert_array_less(neg_end_vs_start, 0)
            np.testing.assert_array_less(-pos_end_vs_start, 0)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(-neg_end_vs_start, 0)
            np.testing.assert_array_less(pos_end_vs_start, 0)
        elif self.operating_condition == "off":
            np.testing.assert_array_almost_equal(neg_end_vs_start, 0)
            np.testing.assert_array_almost_equal(pos_end_vs_start, 0)

    def test_ocv(self):
        "Test open-circuit-voltage decreases during discharge"

    def test_voltage(self):
        "Test terminal voltage is less than open-circuit-voltage during a discharge"

    def test_consistent(self):
        """Test voltage components are consistent with one another by ensuring they sum 
        correctly"""

        # ocv = ocp_p - ocp_n
        # v = ocv + eta_r

    def test_all(self):
        self.test_each_reaction_overpotential()
        self.test_total_reaction_overpotential()
        self.test_ocps()
        self.test_ocv()
        self.test_voltage()
        self.test_consistent()


class ParticleConcentrationTests(object):
    def __init__(self, model, disc, solver, parameter_values):
        self.model = model
        self.disc = disc
        self.solver = solver

        # create all the required terms here

    def test_concentration_increase_decrease(self):
        """Test all concentrations in negative particles decrease and all 
        concentrations in positive particles increase during discharge."""

    def test_concentration_limits(self):
        "Test that concentrations do not go below 0 or exceed the maximum."

    def test_conservation(self):
        "Test amount of lithium stored across all particles is constant."

    def test_concentration_profile(self):
        """Test that the concentration in the centre of the negative particles is 
        greater than the average concentration in the particle and also that the 
        concentration on the surface of the negative particle is less than the average 
        concentration in the particle. Test opposite is true for the positive 
        particle."""

    def test_fluxes(self):
        """Test that no flux holds in the centre of the particle. Test that surface 
        flux in the negative particles is less than zero and that the flux on the 
        surface of the positive particles is greater than zeros during a discharge."""

    def test_all(self):
        self.test_concentration_increase_decrease()
        self.test_concentration_limits()
        self.test_conservation()
        self.test_concentration_profile()
        self.test_fluxes()


class ElectrolyteConcentrationTests(object):
    def __init__(self, model, disc, solver, parameter_values):
        self.model = model
        self.disc = disc
        self.solver = solver

        # create all the required terms here

    def test_concentration_limit(self):
        "Test that the electrolyte concentration is always greater than zero."

    def test_conservation(self):
        "Test conservation of species in the electrolyte."

    def test_concentration_profile(self):
        """Test continuity of the concentration profile. Test average concentration is 
        as expected and that the concentration in the negative electrode is greater 
        than the average and the concentration in the positive is less than the average 
        during a discharge."""

    def test_fluxes(self):
        """Test that the internal boundary fluxes are continuous. Test current 
        collector fluxes are zero."""

    def test_splitting(self):
        """Test that when splitting the concentrations and fluxes by negative electrode,
        separator, and positive electrode, we get the correct behaviour: continuous 
        solution and recover combined through concatenation."""

    def test_all(self):
        self.test_concentration_limit()
        self.test_conservation()
        self.test_concentration_profile()
        self.test_fluxes()
        self.test_splitting()


class PotentialTests(object):
    def __init__(self, model, disc, solver, parameter_values):
        self.model = model
        self.disc = disc
        self.solver = solver

        # create all the required terms here

    def test_negative_electrode_potential_profile(self):
        """Test that negative electrode potential is zero on left boundary. Test
        average negative electrode potential is less than or equal to zero."""

    def test_positive_electrode_potential_profile(self):
        """Test average positive electrode potential is less than the positive electrode
        potential on the right current collector."""

    def test_potential_differences(self):
        """Test electrolyte potential is less than the negative electrode potential.
        Test that the positive electrode potential is greater than the negative 
        electrode potential."""

    def test_potential_profile(self):
        """Test that negative electrode potential is zero on left boundary. Test
        average negative electrode potential is less than or equal to zero. Test
        average positive electrode potential is greater than average negative electrode 
        potential. Test average positive electrode potential is less than the positive 
        electrode potential on the right current collector."""

    def test_all(self):
        self.test_negative_electrode_potential_profile()
        self.test_positive_electrode_potential_profile()
        self.test_potential_differences()
        self.test_potential_profile()


class CurrentTests(object):
    def __init__(self, model, disc, solver, parameter_values):
        self.model = model
        self.disc = disc
        self.solver = solver

    def test_interfacial_current_average(self):
        """Test that average of the interfacial current density is equal to the true 
        value."""

    def test_conservation(self):
        """Test sum of electrode and electrolyte current densities give the applied 
        current density"""

    def test_all(self):
        self.test_interfacial_current_average()
        self.test_conservation()

