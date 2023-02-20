#
# Standard tests on the standard set of model outputs
#
import pybamm
import numpy as np


class StandardOutputTests(object):
    """Calls all the tests on the standard output variables."""

    def __init__(self, model, parameter_values, disc, solution):
        # Assign attributes
        self.model = model
        self.parameter_values = parameter_values
        self.disc = disc
        self.solution = solution

        if isinstance(self.model, pybamm.lithium_ion.BaseModel):
            self.chemistry = "Lithium-ion"
        elif isinstance(self.model, pybamm.lead_acid.BaseModel):
            self.chemistry = "Lead acid"

        # Only for constant current
        current_sign = np.sign(parameter_values["Current function [A]"])

        if current_sign == 1:
            self.operating_condition = "discharge"
        elif current_sign == -1:
            self.operating_condition = "charge"
        else:
            self.operating_condition = "off"

    def process_variables(self):
        return

    def run_test_class(self, ClassName):
        """Run all tests from a class 'ClassName'"""
        tests = ClassName(
            self.model,
            self.parameter_values,
            self.disc,
            self.solution,
            self.operating_condition,
        )
        tests.test_all()

    def test_all(self, skip_first_timestep=False):
        self.run_test_class(VoltageTests)
        self.run_test_class(ElectrolyteConcentrationTests)
        self.run_test_class(PotentialTests)
        self.run_test_class(CurrentTests)

        if self.chemistry == "Lithium-ion":
            self.run_test_class(ParticleConcentrationTests)
            self.run_test_class(DegradationTests)

        if self.model.options["convection"] != "none":
            self.run_test_class(VelocityTests)


class BaseOutputTest(object):
    def __init__(self, model, param, disc, solution, operating_condition):
        self.model = model
        self.param = param
        self.disc = disc
        self.solution = solution
        self.operating_condition = operating_condition

        # Get phase names
        self.phase_name_n = (
            "" if self.model.options.negative["particle phases"] == "1" else "primary "
        )
        self.phase_name_p = (
            "" if self.model.options.positive["particle phases"] == "1" else "primary "
        )

        # Use dimensional time and space
        self.t = solution.t * model.timescale_eval
        geo = pybamm.geometric_parameters

        L_x = param.evaluate(geo.L_x)
        self.x_n = disc.mesh["negative electrode"].nodes * L_x
        self.x_s = disc.mesh["separator"].nodes * L_x
        self.x_p = disc.mesh["positive electrode"].nodes * L_x
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        self.x = disc.mesh[whole_cell].nodes * L_x
        self.x_n_edge = disc.mesh["negative electrode"].edges * L_x
        self.x_s_edge = disc.mesh["separator"].edges * L_x
        self.x_p_edge = disc.mesh["positive electrode"].edges * L_x
        self.x_edge = disc.mesh[whole_cell].edges * L_x

        if isinstance(self.model, pybamm.lithium_ion.BaseModel):
            R_n_typ = model.length_scales["negative particle"].evaluate()
            R_p_typ = model.length_scales["positive particle"].evaluate()
            self.r_n = disc.mesh["negative particle"].nodes * R_n_typ
            self.r_p = disc.mesh["positive particle"].nodes * R_p_typ
            self.r_n_edge = disc.mesh["negative particle"].edges * R_n_typ
            self.r_p_edge = disc.mesh["positive particle"].edges * R_p_typ
            if self.model.options["particle size"] == "distribution":
                self.R_n = disc.mesh["negative particle size"].nodes * R_n_typ
                self.R_p = disc.mesh["positive particle size"].nodes * R_p_typ

        # Useful parameters
        self.l_n = param.evaluate(geo.n.l)
        self.l_p = param.evaluate(geo.p.l)

        current_param = self.model.param.current_with_time

        self.i_cell = param.process_symbol(current_param).evaluate(solution.t)


class VoltageTests(BaseOutputTest):
    def __init__(self, model, param, disc, solution, operating_condition):
        super().__init__(model, param, disc, solution, operating_condition)

        self.eta_r_n = solution[
            f"Negative electrode {self.phase_name_n}reaction overpotential [V]"
        ]
        self.eta_r_p = solution[
            f"Positive electrode {self.phase_name_p}reaction overpotential [V]"
        ]
        self.eta_r_n_av = solution[
            f"X-averaged negative electrode {self.phase_name_n}"
            "reaction overpotential [V]"
        ]
        self.eta_r_p_av = solution[
            f"X-averaged positive electrode {self.phase_name_p}"
            "reaction overpotential [V]"
        ]
        self.eta_r_av = solution["X-averaged reaction overpotential [V]"]

        self.eta_sei_av = solution["X-averaged SEI film overpotential [V]"]

        self.eta_e_av = solution["X-averaged electrolyte overpotential [V]"]
        self.delta_phi_s_av = solution["X-averaged solid phase ohmic losses [V]"]

        self.ocp_n_av = solution[
            f"X-averaged negative electrode {self.phase_name_n}"
            "open circuit potential [V]"
        ]
        self.ocp_p_av = solution[
            f"X-averaged positive electrode {self.phase_name_p}"
            "open circuit potential [V]"
        ]
        self.ocv_av = solution["X-averaged open circuit voltage [V]"]
        self.voltage = solution["Terminal voltage [V]"]

    def test_each_reaction_overpotential(self):
        """Testing that:
        - discharge: eta_r_n > 0, eta_r_p < 0
        - charge: eta_r_n < 0, eta_r_p > 0
        - off: eta_r_n == 0, eta_r_p == 0
        """
        tol = 0.01
        t, x_n, x_p = self.t, self.x_n, self.x_p
        if self.operating_condition == "discharge":
            np.testing.assert_array_less(-self.eta_r_n(t, x_n), tol)
            np.testing.assert_array_less(self.eta_r_p(t, x_p), tol)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(self.eta_r_n(t, x_n), tol)
            np.testing.assert_array_less(-self.eta_r_p(t, x_p), tol)
        elif self.operating_condition == "off":
            np.testing.assert_array_almost_equal(self.eta_r_n(t, x_n), 0)
            np.testing.assert_array_almost_equal(-self.eta_r_p(t, x_p), 0)

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
            np.testing.assert_array_less(self.delta_phi_s_av(self.t), tol)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(-self.eta_r_av(self.t), tol)
            np.testing.assert_array_less(-self.eta_e_av(self.t), tol)
            np.testing.assert_array_less(-self.delta_phi_s_av(self.t), tol)

        elif self.operating_condition == "off":
            np.testing.assert_array_almost_equal(self.eta_r_av(self.t), 0)
            np.testing.assert_array_almost_equal(self.eta_e_av(self.t), 0, decimal=16)
            # For some reason SPM gives delta_phi_s_av ~ 1e-17
            np.testing.assert_array_almost_equal(
                self.delta_phi_s_av(self.t), 0, decimal=16
            )

    def test_ocps(self):
        """Testing that:
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
            self.eta_r_av(self.t), self.eta_r_p_av(self.t) - self.eta_r_n_av(self.t)
        )

        np.testing.assert_array_almost_equal(
            self.voltage(self.t),
            self.ocv_av(self.t)
            + self.eta_r_av(self.t)
            + self.eta_e_av(self.t)
            + self.delta_phi_s_av(self.t)
            + self.eta_sei_av(self.t),
            decimal=2,
        )

    def test_all(self):
        self.test_each_reaction_overpotential()
        self.test_overpotentials()
        self.test_ocps()
        self.test_ocv()
        self.test_voltage()
        self.test_consistent()


class ParticleConcentrationTests(BaseOutputTest):
    def __init__(self, model, param, disc, solution, operating_condition):
        super().__init__(model, param, disc, solution, operating_condition)

        self.c_s_n = solution[f"Negative {self.phase_name_n}particle concentration"]
        self.c_s_p = solution[f"Positive {self.phase_name_p}particle concentration"]

        self.c_s_n_rav = solution[
            f"R-averaged negative {self.phase_name_n}particle concentration"
        ]
        self.c_s_p_rav = solution[
            f"R-averaged positive {self.phase_name_p}particle concentration"
        ]

        self.c_s_n_surf = solution[
            f"Negative {self.phase_name_n}particle surface concentration"
        ]
        self.c_s_p_surf = solution[
            f"Positive {self.phase_name_p}particle surface concentration"
        ]

        self.c_s_n_tot = solution["Total lithium in negative electrode [mol]"]
        self.c_s_p_tot = solution["Total lithium in positive electrode [mol]"]

        self.N_s_n = solution[f"Negative {self.phase_name_n}particle flux"]
        self.N_s_p = solution[f"Positive {self.phase_name_p}particle flux"]

        self.n_Li_side = solution["Total lithium lost to side reactions [mol]"]
        self.n_Li_LAM_n = solution[
            "Loss of lithium due to loss of active material in negative electrode [mol]"
        ]
        self.n_Li_LAM_p = solution[
            "Loss of lithium due to loss of active material in positive electrode [mol]"
        ]

        if model.options["particle size"] == "distribution":
            # These concentration variables are only present for distribution models.

            # Take only the x-averaged of these for now, since variables cannot have
            # 4 domains yet
            self.c_s_n_dist = solution[
                "X-averaged negative particle concentration distribution"
            ]
            self.c_s_p_dist = solution[
                "X-averaged positive particle concentration distribution"
            ]

            self.c_s_n_surf_dist = solution[
                "Negative particle surface concentration distribution"
            ]
            self.c_s_p_surf_dist = solution[
                "Positive particle surface concentration distribution"
            ]

    def test_concentration_increase_decrease(self):
        """Test all concentrations in negative particles decrease and all
        concentrations in positive particles increase over a discharge."""

        t, x_n, x_p, r_n, r_p = self.t, self.x_n, self.x_p, self.r_n, self.r_p

        tol = 1e-16

        if self.model.options["particle"] in ["quadratic profile", "quartic profile"]:
            # For the assumed polynomial concentration profiles the values
            # can increase/decrease within the particle as the polynomial shifts,
            # so we just check the average instead
            neg_diff = self.c_s_n_rav(t[1:], x_n) - self.c_s_n_rav(t[:-1], x_n)
            pos_diff = self.c_s_p_rav(t[1:], x_p) - self.c_s_p_rav(t[:-1], x_p)
            neg_end_vs_start = self.c_s_n_rav(t[-1], x_n) - self.c_s_n_rav(t[0], x_n)
            pos_end_vs_start = self.c_s_p_rav(t[-1], x_p) - self.c_s_p_rav(t[0], x_p)
        elif self.model.options["particle size"] == "distribution":
            R_n, R_p = self.R_n, self.R_p
            # Test the concentration variables that depend on x-R (surface values only,
            # as 3D vars not implemented)
            neg_diff = self.c_s_n_surf_dist(t[1:], x=x_n, R=R_n) - self.c_s_n_surf_dist(
                t[:-1], x=x_n, R=R_n
            )
            pos_diff = self.c_s_p_surf_dist(t[1:], x=x_p, R=R_p) - self.c_s_p_surf_dist(
                t[:-1], x=x_p, R=R_p
            )
            neg_end_vs_start = self.c_s_n_surf_dist(
                t[-1], x=x_n, R=R_n
            ) - self.c_s_n_surf_dist(t[0], x=x_n, R=R_n)
            pos_end_vs_start = self.c_s_p_surf_dist(
                t[-1], x=x_p, R=R_p
            ) - self.c_s_p_surf_dist(t[0], x=x_p, R=R_p)
            tol = 1e-15
        else:
            neg_diff = self.c_s_n(t[1:], x_n, r_n) - self.c_s_n(t[:-1], x_n, r_n)
            pos_diff = self.c_s_p(t[1:], x_p, r_p) - self.c_s_p(t[:-1], x_p, r_p)
            neg_end_vs_start = self.c_s_n(t[-1], x_n, r_n) - self.c_s_n(t[0], x_n, r_n)
            pos_end_vs_start = self.c_s_p(t[-1], x_p, r_p) - self.c_s_p(t[0], x_p, r_p)

        if self.operating_condition == "discharge":
            np.testing.assert_array_less(neg_diff, tol)
            np.testing.assert_array_less(-tol, pos_diff)
            np.testing.assert_array_less(neg_end_vs_start, 0)
            np.testing.assert_array_less(0, pos_end_vs_start)
        elif self.operating_condition == "charge":
            np.testing.assert_array_less(-tol, neg_diff)
            np.testing.assert_array_less(pos_diff, tol)
            np.testing.assert_array_less(0, neg_end_vs_start)
            np.testing.assert_array_less(pos_end_vs_start, 0)
        elif self.operating_condition == "off":
            np.testing.assert_array_almost_equal(neg_diff, 0)
            np.testing.assert_array_almost_equal(pos_diff, 0)
            np.testing.assert_array_almost_equal(neg_end_vs_start, 0)
            np.testing.assert_array_almost_equal(pos_end_vs_start, 0)

    def test_concentration_limits(self):
        """Test that concentrations do not go below 0 or exceed the maximum."""
        t, x_n, x_p, r_n, r_p = self.t, self.x_n, self.x_p, self.r_n, self.r_p

        np.testing.assert_array_less(-self.c_s_n(t, x_n, r_n), 0)
        np.testing.assert_array_less(-self.c_s_p(t, x_p, r_p), 0)

        np.testing.assert_array_less(self.c_s_n(t, x_n, r_n), 1)
        np.testing.assert_array_less(self.c_s_p(t, x_p, r_p), 1)
        if self.model.options["particle size"] == "distribution":
            R_n, R_p = self.R_n, self.R_p
            # Cannot have 3D processed variables, so test concs that depend on
            # r-R and x-R

            # r-R (x-averaged)
            np.testing.assert_array_less(-self.c_s_n_dist(t, r=r_n, R=R_n), 0)
            np.testing.assert_array_less(-self.c_s_p_dist(t, r=r_p, R=R_p), 0)

            np.testing.assert_array_less(self.c_s_n_dist(t, r=r_n, R=R_n), 1)
            np.testing.assert_array_less(self.c_s_p_dist(t, r=r_p, R=R_p), 1)

            # x-R (surface concentrations)
            np.testing.assert_array_less(-self.c_s_n_surf_dist(t, x=x_n, R=R_n), 0)
            np.testing.assert_array_less(-self.c_s_p_surf_dist(t, x=x_p, R=R_p), 0)

            np.testing.assert_array_less(self.c_s_n_surf_dist(t, x=x_n, R=R_n), 1)
            np.testing.assert_array_less(self.c_s_p_surf_dist(t, x=x_p, R=R_p), 1)

    def test_conservation(self):
        """Test amount of lithium stored across all particles and in SEI layers is
        constant."""
        c_s_tot = (
            self.c_s_n_tot(self.solution.t)
            + self.c_s_p_tot(self.solution.t)
            + self.n_Li_side(self.solution.t)
            + self.n_Li_LAM_n(self.solution.t)
            + self.n_Li_LAM_p(self.solution.t)
        )
        diff = (c_s_tot[1:] - c_s_tot[:-1]) / c_s_tot[:-1]
        if self.model.options["particle"] == "quartic profile":
            np.testing.assert_array_almost_equal(diff, 0, decimal=10)
        # elif self.model.options["particle size"] == "distribution":
        #     np.testing.assert_array_almost_equal(diff, 0, decimal=10)
        elif self.model.options["surface form"] == "differential":
            np.testing.assert_array_almost_equal(diff, 0, decimal=10)
        else:
            np.testing.assert_array_almost_equal(diff, 0, decimal=15)

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

        t, x_n, x_p, r_n, r_p = (
            self.t,
            self.x_n,
            self.x_p,
            self.r_n_edge,
            self.r_p_edge,
        )
        if self.model.options["particle"] == "uniform profile":
            # Fluxes are zero everywhere since the concentration is uniform
            np.testing.assert_array_almost_equal(self.N_s_n(t, x_n, r_n), 0)
            np.testing.assert_array_almost_equal(self.N_s_p(t, x_p, r_p), 0)
        else:
            if self.operating_condition == "discharge":
                if self.model.options["particle"] == "quartic profile":
                    # quartic profile has a transient at the beginning where
                    # the concentration "rearranges" giving flux of the opposite
                    # sign, so ignore first three times
                    np.testing.assert_array_less(0, self.N_s_n(t[3:], x_n, r_n[1:]))
                    np.testing.assert_array_less(self.N_s_p(t[3:], x_p, r_p[1:]), 0)
                else:
                    np.testing.assert_array_less(
                        -1e-16, self.N_s_n(t[1:], x_n, r_n[1:])
                    )
                    np.testing.assert_array_less(self.N_s_p(t[1:], x_p, r_p[1:]), 1e-16)
            if self.operating_condition == "charge":
                np.testing.assert_array_less(self.N_s_n(t[1:], x_n, r_n[1:]), 1e-16)
                np.testing.assert_array_less(-1e-16, self.N_s_p(t[1:], x_p, r_p[1:]))
            if self.operating_condition == "off":
                np.testing.assert_array_almost_equal(self.N_s_n(t, x_n, r_n), 0)
                np.testing.assert_array_almost_equal(self.N_s_p(t, x_p, r_p), 0)

        np.testing.assert_array_almost_equal(0, self.N_s_n(t, x_n, r_n[0]), decimal=4)
        np.testing.assert_array_almost_equal(0, self.N_s_p(t, x_p, r_p[0]), decimal=4)

    def test_all(self):
        self.test_concentration_increase_decrease()
        self.test_concentration_limits()
        self.test_conservation()
        self.test_concentration_profile()
        self.test_fluxes()


class ElectrolyteConcentrationTests(BaseOutputTest):
    def __init__(self, model, param, disc, solution, operating_condition):
        super().__init__(model, param, disc, solution, operating_condition)

        self.c_e = solution["Electrolyte concentration"]

        self.c_e_n = solution["Negative electrolyte concentration"]
        self.c_e_s = solution["Separator electrolyte concentration"]
        self.c_e_p = solution["Positive electrolyte concentration"]

        self.c_e_av = solution["X-averaged electrolyte concentration"]
        self.c_e_n_av = solution["X-averaged negative electrolyte concentration"]
        self.c_e_s_av = solution["X-averaged separator electrolyte concentration"]
        self.c_e_p_av = solution["X-averaged positive electrolyte concentration"]
        self.c_e_tot = solution["Total lithium in electrolyte [mol]"]

        self.N_e_hat = solution["Electrolyte flux"]
        # self.N_e_hat = solution["Reduced cation flux"]

    def test_concentration_limit(self):
        """Test that the electrolyte concentration is always greater than zero."""
        np.testing.assert_array_less(-self.c_e(self.t, self.x), 0)

    def test_conservation(self):
        """Test conservation of species in the electrolyte."""
        # sufficient to check average concentration is constant

        diff = (
            self.c_e_tot(self.solution.t[1:]) - self.c_e_tot(self.solution.t[:-1])
        ) / self.c_e_tot(self.solution.t[:-1])
        if self.model.options["surface form"] == "differential":
            np.testing.assert_array_almost_equal(diff, 0, decimal=9)
        else:
            np.testing.assert_array_almost_equal(diff, 0, decimal=14)

    def test_concentration_profile(self):
        """Test continuity of the concentration profile. Test average concentration is
        as expected and that the concentration in the negative electrode is greater
        than the average and the concentration in the positive is less than the average
        during a discharge."""

        # TODO: uncomment when have average concentrations
        # small number so that can use array less
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
        """Test current collector fluxes are zero. Tolerance reduced for surface form
        models (bug in implementation of boundary conditions?)"""

        t, x = self.t, self.x_edge
        np.testing.assert_array_almost_equal(self.N_e_hat(t, x[0]), 0, decimal=3)
        np.testing.assert_array_almost_equal(self.N_e_hat(t, x[-1]), 0, decimal=3)

    def test_splitting(self):
        """Test that when splitting the concentrations and fluxes by negative electrode,
        separator, and positive electrode, we get the correct behaviour: continuous
        solution and recover combined through concatenation."""
        t, x_n, x_s, x_p, x = self.t, self.x_n, self.x_s, self.x_p, self.x
        c_e_combined = np.concatenate(
            (self.c_e_n(t, x_n), self.c_e_s(t, x_s), self.c_e_p(t, x_p)), axis=0
        )

        np.testing.assert_array_almost_equal(self.c_e(t, x), c_e_combined, decimal=14)

    def test_all(self):
        self.test_concentration_limit()
        self.test_concentration_profile()
        self.test_fluxes()
        self.test_splitting()

        if isinstance(self.model, pybamm.lithium_ion.BaseModel):
            # electrolyte is not conserved in lead-acid models
            self.test_conservation()


class PotentialTests(BaseOutputTest):
    def __init__(self, model, param, disc, solution, operating_condition):
        super().__init__(model, param, disc, solution, operating_condition)

        self.phi_s_n = solution["Negative electrode potential [V]"]
        self.phi_s_p = solution["Positive electrode potential [V]"]
        self.phi_s_n_av = solution["X-averaged negative electrode potential [V]"]
        self.phi_s_p_av = solution["X-averaged positive electrode potential [V]"]

        self.phi_e = solution["Electrolyte potential [V]"]
        self.phi_e_n = solution["Negative electrolyte potential [V]"]
        self.phi_e_s = solution["Separator electrolyte potential [V]"]
        self.phi_e_p = solution["Positive electrolyte potential [V]"]
        self.phi_e_n_av = solution["X-averaged negative electrolyte potential [V]"]
        self.phi_e_p_av = solution["X-averaged positive electrolyte potential [V]"]

        self.delta_phi_n = solution[
            "Negative electrode surface potential difference [V]"
        ]
        self.delta_phi_p = solution[
            "Positive electrode surface potential difference [V]"
        ]
        self.delta_phi_n_av = solution[
            "X-averaged negative electrode surface potential difference [V]"
        ]
        self.delta_phi_p_av = solution[
            "X-averaged positive electrode surface potential difference [V]"
        ]

        self.grad_phi_e = solution["Gradient of electrolyte potential"]
        self.grad_phi_e_n = solution["Gradient of negative electrolyte potential"]
        self.grad_phi_e_s = solution["Gradient of separator electrolyte potential"]
        self.grad_phi_e_p = solution["Gradient of positive electrolyte potential"]

    def test_negative_electrode_potential_profile(self):
        """Test that negative electrode potential is zero on left boundary. Test
        average negative electrode potential is less than or equal to zero."""
        np.testing.assert_array_almost_equal(self.phi_s_n(self.t, x=0), 0, decimal=5)

    def test_positive_electrode_potential_profile(self):
        """Test average positive electrode potential is less than the positive electrode
        potential on the right current collector."""

        # TODO: add these when have averages

    def test_potential_differences(self):
        """Test that potential differences are the difference between electrode
        potential and electrolyte potential"""
        t, x_n, x_p = self.t, self.x_n, self.x_p

        np.testing.assert_array_almost_equal(
            self.phi_s_n(t, x_n) - self.phi_e_n(t, x_n), self.delta_phi_n(t, x_n)
        )
        np.testing.assert_array_almost_equal(
            self.phi_s_p(t, x_p) - self.phi_e_p(t, x_p),
            self.delta_phi_p(t, x_p),
            decimal=5,
        )

    def test_average_potential_differences(self):
        """Test that average potential differences are the difference between electrode
        potential and electrolyte potential"""
        t = self.t

        np.testing.assert_array_almost_equal(
            self.phi_s_n_av(t) - self.phi_e_n_av(t), self.delta_phi_n_av(t), decimal=4
        )
        np.testing.assert_array_almost_equal(
            self.phi_s_p_av(t) - self.phi_e_p_av(t), self.delta_phi_p_av(t), decimal=4
        )

    def test_gradient_splitting(self):
        t, x_n, x_s, x_p, x = self.t, self.x_n, self.x_s, self.x_p, self.x
        grad_phi_e_combined = np.concatenate(
            (
                self.grad_phi_e_n(t, x_n),
                self.grad_phi_e_s(t, x_s),
                self.grad_phi_e_p(t, x_p),
            ),
            axis=0,
        )

        np.testing.assert_array_equal(self.grad_phi_e(t, x), grad_phi_e_combined)

    def test_all(self):
        self.test_negative_electrode_potential_profile()
        self.test_positive_electrode_potential_profile()
        self.test_potential_differences()
        self.test_average_potential_differences()


class CurrentTests(BaseOutputTest):
    def __init__(self, model, param, disc, solution, operating_condition):
        super().__init__(model, param, disc, solution, operating_condition)

        self.a_j_n = solution[
            "Negative electrode volumetric interfacial current density"
        ]
        self.a_j_p = solution[
            "Positive electrode volumetric interfacial current density"
        ]
        self.a_j_n_av = solution[
            "X-averaged negative electrode volumetric interfacial current density"
        ]
        self.a_j_p_av = solution[
            "X-averaged positive electrode volumetric interfacial current density"
        ]
        self.a_j_n_sei = solution[
            "Negative electrode SEI volumetric interfacial current density"
        ]
        self.a_j_n_sei_av = solution[
            "X-averaged negative electrode SEI volumetric interfacial current density"
        ]
        self.a_j_n_pl = solution[
            "Negative electrode lithium plating volumetric interfacial current density"
        ]
        self.a_j_n_pl_av = solution[
            "X-averaged negative electrode lithium plating "
            "volumetric interfacial current density"
        ]

        self.i_s_n = solution["Negative electrode current density"]
        self.i_s_p = solution["Positive electrode current density"]
        self.i_s = solution["Electrode current density"]
        self.i_e = solution["Electrolyte current density"]

    def test_interfacial_current_average(self):
        """Test that average of the surface area density distribution (in x)
        multiplied by the interfacial current density is equal to the true
        value."""

        np.testing.assert_array_almost_equal(
            np.mean(
                self.a_j_n(self.t, self.x_n)
                + self.a_j_n_sei(self.t, self.x_n)
                + self.a_j_n_pl(self.t, self.x_n),
                axis=0,
            ),
            self.i_cell / self.l_n,
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            np.mean(self.a_j_p(self.t, self.x_p), axis=0),
            -self.i_cell / self.l_p,
            decimal=4,
        )

    def test_conservation(self):
        """Test sum of electrode and electrolyte current densities give the applied
        current density"""
        t, x_n, x_s, x_p = self.t, self.x_n, self.x_s, self.x_p

        current_param = self.model.param.current_with_time

        i_cell = self.param.process_symbol(current_param).evaluate(t=t)
        for x in [x_n, x_s, x_p]:
            np.testing.assert_array_almost_equal(
                self.i_s(t, x) + self.i_e(t, x), i_cell, decimal=2
            )
        np.testing.assert_array_almost_equal(
            self.i_s(t, x_n), self.i_s_n(t, x_n), decimal=3
        )
        np.testing.assert_array_almost_equal(
            self.i_s(t, x_p), self.i_s_p(t, x_p), decimal=3
        )

    def test_current_density_boundaries(self):
        """Test the boundary values of the current densities"""
        t, x_n, x_p = self.t, self.x_n_edge, self.x_p_edge

        current_param = self.model.param.current_with_time

        i_cell = self.param.process_symbol(current_param).evaluate(t=t)
        np.testing.assert_array_almost_equal(self.i_s_n(t, x_n[0]), i_cell, decimal=2)
        np.testing.assert_array_almost_equal(self.i_s_n(t, x_n[-1]), 0, decimal=4)
        np.testing.assert_array_almost_equal(self.i_s_p(t, x_p[-1]), i_cell, decimal=3)
        np.testing.assert_array_almost_equal(self.i_s_p(t, x_p[0]), 0, decimal=4)

    def test_all(self):
        self.test_conservation()
        self.test_current_density_boundaries()
        # Skip average current test if capacitance is used, since average interfacial
        # current density will be affected slightly by capacitance effects
        if self.model.options["surface form"] != "differential":
            self.test_interfacial_current_average()


class VelocityTests(BaseOutputTest):
    def __init__(self, model, param, disc, solution, operating_condition):
        super().__init__(model, param, disc, solution, operating_condition)

        self.v_box = solution["Volume-averaged velocity"]
        self.i_e = solution["Electrolyte current density"]
        self.dVbox_dz = solution["Transverse volume-averaged acceleration"]

    def test_velocity_boundaries(self):
        """Test the boundary values of the current densities"""
        L_x = self.x_edge[-1]
        np.testing.assert_array_almost_equal(self.v_box(self.t, 0), 0, decimal=4)
        np.testing.assert_array_almost_equal(self.v_box(self.t, L_x), 0, decimal=4)

    def test_vertical_velocity(self):
        """Test the boundary values of the current densities"""
        L_x = self.x_edge[-1]
        np.testing.assert_array_equal(self.dVbox_dz(self.t, 0), 0)
        np.testing.assert_array_less(self.dVbox_dz(self.t, 0.5 * L_x), 0)
        np.testing.assert_array_equal(self.dVbox_dz(self.t, L_x), 0)

    def test_velocity_vs_current(self):
        """Test the boundary values of the current densities"""
        t, x_n, x_p = self.t, self.x_n, self.x_p

        beta_n = self.model.param.n.beta
        beta_n = self.param.evaluate(beta_n)
        beta_p = self.model.param.p.beta
        beta_p = self.param.evaluate(beta_p)

        np.testing.assert_array_almost_equal(
            self.v_box(t, x_n), beta_n * self.i_e(t, x_n)
        )
        np.testing.assert_array_almost_equal(
            self.v_box(t, x_p), beta_p * self.i_e(t, x_p)
        )

    def test_all(self):
        self.test_velocity_boundaries()
        self.test_vertical_velocity()
        self.test_velocity_vs_current()


class DegradationTests(BaseOutputTest):
    def __init__(self, model, param, disc, solution, operating_condition):
        super().__init__(model, param, disc, solution, operating_condition)

        self.LAM_ne = solution["Loss of active material in negative electrode [%]"]
        self.LAM_pe = solution["Loss of active material in positive electrode [%]"]
        self.LLI = solution["Loss of lithium inventory [%]"]
        self.n_Li_lost = solution["Total lithium lost [mol]"]
        self.n_Li_lost_rxn = solution["Total lithium lost to side reactions [mol]"]
        self.n_Li_lost_LAM_n = solution[
            "Loss of lithium due to loss of active material in negative electrode [mol]"
        ]
        self.n_Li_lost_LAM_p = solution[
            "Loss of lithium due to loss of active material in positive electrode [mol]"
        ]

    def test_degradation_modes(self):
        """Test degradation modes are between 0 and 100%"""
        np.testing.assert_array_less(-3e-3, self.LLI(self.t))
        np.testing.assert_array_less(-1e-13, self.LAM_ne(self.t))
        np.testing.assert_array_less(-1e-13, self.LAM_pe(self.t))
        np.testing.assert_array_less(-1e-13, self.n_Li_lost_LAM_n(self.t))
        np.testing.assert_array_less(-1e-13, self.n_Li_lost_LAM_p(self.t))
        np.testing.assert_array_less(self.LLI(self.t), 100)
        np.testing.assert_array_less(self.LAM_ne(self.t), 100)
        np.testing.assert_array_less(self.LAM_pe(self.t), 100)

    def test_lithium_lost(self):
        """Test the two ways of measuring lithium lost give the same value"""
        np.testing.assert_array_almost_equal(
            self.n_Li_lost(self.t),
            self.n_Li_lost_rxn(self.t)
            + self.n_Li_lost_LAM_n(self.t)
            + self.n_Li_lost_LAM_p(self.t),
            decimal=5,
        )

    def test_all(self):
        self.test_degradation_modes()
        self.test_lithium_lost()
