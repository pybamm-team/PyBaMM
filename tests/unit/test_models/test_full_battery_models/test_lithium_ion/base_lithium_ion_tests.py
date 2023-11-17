#
# Base unit tests for the lithium-ion models
#
import pybamm


class BaseUnitTestLithiumIon:
    def check_well_posedness(self, options):
        model = self.model(options)
        model.check_well_posedness()

    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        self.check_well_posedness(options)

    def test_well_posed_isothermal_heat_source(self):
        options = {
            "calculate heat source for isothermal models": "true",
            "thermal": "isothermal",
        }
        self.check_well_posedness(options)

    def test_well_posed_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        self.check_well_posedness(options)

        options = {"current collector": "potential pair", "dimensionality": 2}
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped"}
        self.check_well_posedness(options)

    def test_well_posed_x_full_thermal_model(self):
        options = {"thermal": "x-full"}
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        }
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
        }
        self.check_well_posedness(options)

    def test_well_posed_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
        }
        self.check_well_posedness(options)

    def test_well_posed_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        }
        self.check_well_posedness(options)

    def test_well_posed_contact_resistance(self):
        options = {"contact resistance": "true"}
        self.check_well_posedness(options)

    def test_well_posed_particle_uniform(self):
        options = {"particle": "uniform profile"}
        self.check_well_posedness(options)

    def test_well_posed_particle_quadratic(self):
        options = {"particle": "quadratic profile"}
        self.check_well_posedness(options)

    def test_well_posed_particle_quartic(self):
        options = {"particle": "quartic profile"}
        self.check_well_posedness(options)

    def test_well_posed_particle_mixed(self):
        options = {"particle": ("Fickian diffusion", "quartic profile")}
        self.check_well_posedness(options)

    def test_well_posed_constant_utilisation(self):
        options = {"interface utilisation": "constant"}
        self.check_well_posedness(options)

    def test_well_posed_current_driven_utilisation(self):
        options = {"interface utilisation": "current-driven"}
        self.check_well_posedness(options)

    def test_well_posed_mixed_utilisation(self):
        options = {"interface utilisation": ("current-driven", "constant")}
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("stress-driven", "none")}
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("none", "stress-driven")}
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_reaction(self):
        options = {"loss of active material": "reaction-driven"}
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_reaction(self):
        options = {"loss of active material": "stress and reaction-driven"}
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_current_negative(self):
        options = {"loss of active material": ("current-driven", "none")}
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_current_positive(self):
        options = {"loss of active material": ("none", "current-driven")}
        self.check_well_posedness(options)

    def test_well_posed_surface_form_differential(self):
        options = {"surface form": "differential"}
        self.check_well_posedness(options)

    def test_well_posed_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        self.check_well_posedness(options)

    def test_well_posed_kinetics_asymmetric_butler_volmer(self):
        options = {"intercalation kinetics": "asymmetric Butler-Volmer"}
        self.check_well_posedness(options)

    def test_well_posed_kinetics_linear(self):
        options = {"intercalation kinetics": "linear"}
        self.check_well_posedness(options)

    def test_well_posed_kinetics_marcus(self):
        options = {"intercalation kinetics": "Marcus"}
        self.check_well_posedness(options)

    def test_well_posed_kinetics_mhc(self):
        options = {"intercalation kinetics": "Marcus-Hush-Chidsey"}
        self.check_well_posedness(options)

    def test_well_posed_sei_constant(self):
        options = {"SEI": "constant"}
        self.check_well_posedness(options)

    def test_well_posed_sei_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        self.check_well_posedness(options)

    def test_well_posed_asymmetric_sei_reaction_limited(self):
        options = {"SEI": "reaction limited (asymmetric)"}
        self.check_well_posedness(options)

    def test_well_posed_sei_reaction_limited_average_film_resistance(self):
        options = {
            "SEI": "reaction limited",
            "SEI film resistance": "average",
        }
        self.check_well_posedness(options)

    def test_well_posed_asymmetric_sei_reaction_limited_average_film_resistance(self):
        options = {
            "SEI": "reaction limited (asymmetric)",
            "SEI film resistance": "average",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        self.check_well_posedness(options)

    def test_well_posed_sei_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        self.check_well_posedness(options)

    def test_well_posed_sei_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        self.check_well_posedness(options)

    def test_well_posed_sei_ec_reaction_limited(self):
        options = {
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_asymmetric_ec_reaction_limited(self):
        options = {
            "SEI": "ec reaction limited (asymmetric)",
            "SEI porosity change": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_mechanics_negative_cracking(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        self.check_well_posedness(options)

    def test_well_posed_mechanics_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        self.check_well_posedness(options)

    def test_well_posed_mechanics_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        self.check_well_posedness(options)

    def test_well_posed_mechanics_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        self.check_well_posedness(options)

    def test_well_posed_mechanics_stress_induced_diffusion(self):
        options = {
            "particle mechanics": "swelling only",
            "stress-induced diffusion": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_mechanics_stress_induced_diffusion_mixed(self):
        options = {
            "particle mechanics": "swelling only",
            "stress-induced diffusion": ("true", "false"),
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_reaction_limited_on_cracks(self):
        options = {
            "SEI": "reaction limited",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_solvent_diffusion_limited_on_cracks(self):
        options = {
            "SEI": "solvent-diffusion limited",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_electron_migration_limited_on_cracks(self):
        options = {
            "SEI": "electron-migration limited",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_interstitial_diffusion_limited_on_cracks(self):
        options = {
            "SEI": "interstitial-diffusion limited",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_ec_reaction_limited_on_cracks(self):
        options = {
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
        }
        self.check_well_posedness(options)

    def test_well_posed_reversible_plating(self):
        options = {"lithium plating": "reversible"}
        self.check_well_posedness(options)

    def test_well_posed_irreversible_plating(self):
        options = {"lithium plating": "irreversible"}
        self.check_well_posedness(options)

    def test_well_posed_partially_reversible_plating(self):
        options = {"lithium plating": "partially reversible"}
        self.check_well_posedness(options)

    def test_well_posed_reversible_plating_with_porosity(self):
        options = {
            "lithium plating": "reversible",
            "lithium plating porosity change": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_irreversible_plating_with_porosity(self):
        options = {
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_partially_reversible_plating_with_porosity(self):
        options = {
            "lithium plating": "partially reversible",
            "lithium plating porosity change": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_discharge_energy(self):
        options = {"calculate discharge energy": "true"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_voltage(self):
        options = {"operating mode": "voltage"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_power(self):
        options = {"operating mode": "power"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_differential_power(self):
        options = {"operating mode": "differential power"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_resistance(self):
        options = {"operating mode": "resistance"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_differential_resistance(self):
        options = {"operating mode": "differential resistance"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_cccv(self):
        options = {"operating mode": "CCCV"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_function(self):
        def external_circuit_function(variables):
            I = variables["Current [A]"]
            V = variables["Voltage [V]"]
            return (
                V
                + I
                - pybamm.FunctionParameter(
                    "Function", {"Time [s]": pybamm.t}, print_name="test_fun"
                )
            )

        options = {"operating mode": external_circuit_function}
        self.check_well_posedness(options)

    def test_well_posed_particle_phases(self):
        options = {"particle phases": "2"}
        self.check_well_posedness(options)

        options = {"particle phases": ("2", "1")}
        self.check_well_posedness(options)

        options = {"particle phases": ("1", "2")}
        self.check_well_posedness(options)

    def test_well_posed_particle_phases_sei(self):
        options = {"particle phases": "2", "SEI": "ec reaction limited"}
        self.check_well_posedness(options)

    def test_well_posed_current_sigmoid_ocp(self):
        options = {"open-circuit potential": "current sigmoid"}
        self.check_well_posedness(options)

    def test_well_posed_msmr(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
            "surface form": "differential",
        }
        self.check_well_posedness(options)

    def test_well_posed_current_sigmoid_exchange_current(self):
        options = {"exchange-current density": "current sigmoid"}
        self.check_well_posedness(options)

    def test_well_posed_current_sigmoid_diffusivity(self):
        options = {"diffusivity": "current sigmoid"}
        self.check_well_posedness(options)

    def test_well_posed_psd(self):
        options = {"particle size": "distribution", "surface form": "algebraic"}
        self.check_well_posedness(options)

    def test_well_posed_composite_kinetic_hysteresis(self):
        options = {
            "particle phases": ("2", "1"),
            "exchange-current density": (
                ("current sigmoid", "single"),
                "current sigmoid",
            ),
            "open-circuit potential": (("current sigmoid", "single"), "single"),
        }
        self.check_well_posedness(options)

    def test_well_posed_composite_diffusion_hysteresis(self):
        options = {
            "particle phases": ("2", "1"),
            "diffusivity": (("current sigmoid", "current sigmoid"), "current sigmoid"),
            "open-circuit potential": (("current sigmoid", "single"), "single"),
        }
        self.check_well_posedness(options)
