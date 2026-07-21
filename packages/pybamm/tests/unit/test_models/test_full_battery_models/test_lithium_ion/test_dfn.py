import pytest

import pybamm
from tests import BaseUnitTestLithiumIon


class TestDFN(BaseUnitTestLithiumIon):
    def setup_method(self):
        self.model = pybamm.lithium_ion.DFN

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "integrated"}
        with pytest.raises(pybamm.OptionError, match=r"electrolyte conductivity"):
            pybamm.lithium_ion.DFN(options)

    def test_stoichiometry_dependent_conductivity(self):
        # the electrode conductivity submodels should feed the surface stoichiometry
        # into sigma, so a stoichiometry-dependent conductivity changes the solution
        values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(pybamm.lithium_ion.DFN(), parameter_values=values)
        sol_const = sim.solve([0, 600])
        v_const = sol_const["Voltage [V]"].entries[-1]

        sigma_n = values["Negative electrode conductivity [S.m-1]"]
        sigma_p = values["Positive electrode conductivity [S.m-1]"]
        values_sto = values.copy()
        values_sto.update(
            {
                "Negative electrode conductivity [S.m-1]": lambda sto, T: (
                    0.1 * sigma_n * (0.1 + sto)
                ),
                "Positive electrode conductivity [S.m-1]": lambda sto, T: (
                    0.1 * sigma_p * (0.1 + sto)
                ),
            }
        )
        sim_sto = pybamm.Simulation(
            pybamm.lithium_ion.DFN(), parameter_values=values_sto
        )
        sol_sto = sim_sto.solve([0, 600])
        v_sto = sol_sto["Voltage [V]"].entries[-1]

        assert abs(v_const - v_sto) > 1e-4

    def test_stoichiometry_dependent_conductivity_composite(self):
        # primary-phase surface stoichiometry should feed sigma in a two-phase electrode
        options = {"particle phases": ("2", "1")}
        values = pybamm.ParameterValues("Chen2020_composite")
        values.update(
            {
                "Negative electrode conductivity [S.m-1]": lambda sto, T: (
                    0.5 * (0.1 + sto)
                ),
            }
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.DFN(options), parameter_values=values
        )
        sol = sim.solve([0, 600])
        assert sol["Voltage [V]"].entries[-1] > 0

    def test_well_posed_size_distribution(self):
        options = {"particle size": "distribution"}
        self.check_well_posedness(options)

    def test_well_posed_size_distribution_uniform_profile(self):
        options = {"particle size": "distribution", "particle": "uniform profile"}
        self.check_well_posedness(options)

    def test_well_posed_size_distribution_tuple(self):
        options = {"particle size": ("single", "distribution")}
        self.check_well_posedness(options)

    def test_well_posed_size_distribution_composite(self):
        options = {"particle size": "distribution", "particle phases": "2"}
        self.check_well_posedness(options)

    def test_well_posed_current_sigmoid_ocp_with_psd(self):
        options = {
            "open-circuit potential": "current sigmoid",
            "particle size": "distribution",
        }
        self.check_well_posedness(options)

    def test_well_posed_one_state_differential_capacity_hysteresis_ocp_with_psd(self):
        options = {
            "open-circuit potential": "one-state differential capacity hysteresis",
            "particle size": "distribution",
        }
        self.check_well_posedness(options)

    def test_well_posed_one_state_differential_capacity_hysteresis_ocp_with_composite(
        self,
    ):
        options = {
            "open-circuit potential": (
                ("one-state differential capacity hysteresis", "single"),
                "single",
            ),
            "particle phases": ("2", "1"),
        }
        self.check_well_posedness(options)

    def test_well_posed_one_state_differential_capacity_hysteresis_thermal(self):
        options = {
            "open-circuit potential": "one-state differential capacity hysteresis",
            "thermal": "lumped",
        }
        self.check_well_posedness(options)

    def test_well_posed_one_state_hysteresis_ocp_with_psd(self):
        options = {
            "open-circuit potential": "one-state hysteresis",
            "particle size": "distribution",
        }
        self.check_well_posedness(options)

    def test_well_posed_one_state_hysteresis_ocp_with_composite(self):
        options = {
            "open-circuit potential": (("one-state hysteresis", "single"), "single"),
            "particle phases": ("2", "1"),
        }
        self.check_well_posedness(options)

    def test_well_posed_one_state_hysteresis_thermal(self):
        options = {
            "open-circuit potential": "one-state hysteresis",
            "thermal": "lumped",
        }
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_explicit_power(self):
        options = {"operating mode": "explicit power"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_explicit_resistance(self):
        options = {"operating mode": "explicit resistance"}
        self.check_well_posedness(options)

    def test_well_posed_msmr_with_psd(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "particle size": "distribution",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        self.check_well_posedness(options)
