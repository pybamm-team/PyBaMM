#
# Base unit tests for the lithium-ion models
#
import pytest

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
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped"}
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_model_surface_temperature(self):
        options = {"thermal": "lumped", "surface temperature": "lumped"}
        self.check_well_posedness(options)

    def test_well_posed_x_full_thermal_model(self):
        options = {"thermal": "x-full", "cell geometry": "pouch"}
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_capacity_model(self):
        options = {"thermal": "lumped", "use lumped thermal capacity": "true"}
        self.check_well_posedness(options)

    def test_incompatible_lumped_thermal_capacity_option(self):
        options = {
            "thermal": "x-full",
            "use lumped thermal capacity": "true",
            "cell geometry": "pouch",
        }
        with pytest.raises(
            pybamm.OptionError,
            match=r"Lumped thermal capacity model only compatible with lumped thermal models",
        ):
            self.check_well_posedness(options)

    def test_well_posed_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_isothermal_heat_source_hom(self):
        options = {
            "calculate heat source for isothermal models": "true",
            "thermal": "isothermal",
            "heat of mixing": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_2plus1D_hom(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "heat of mixing": "true",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "heat of mixing": "true",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_model_1D_hom(self):
        options = {"thermal": "lumped", "heat of mixing": "true"}
        self.check_well_posedness(options)

    def test_well_posed_x_full_thermal_model_hom(self):
        options = {
            "thermal": "x-full",
            "heat of mixing": "true",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_1plus1D_hom(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
            "heat of mixing": "true",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_lumped_thermal_2plus1D_hom(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
            "heat of mixing": "true",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_thermal_1plus1D_hom(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
            "heat of mixing": "true",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_thermal_2plus1D_hom(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
            "heat of mixing": "true",
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_contact_resistance(self):
        options = {
            "contact resistance": "true",
            "thermal": "lumped",
        }
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
        options = {
            "loss of active material": ("stress-driven", "none"),
            "particle mechanics": ("swelling only", "none"),
            "stress-induced diffusion": ("true", "false"),
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_positive(self):
        options = {
            "loss of active material": ("none", "stress-driven"),
            "particle mechanics": ("none", "swelling only"),
            "stress-induced diffusion": ("false", "true"),
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_both(self):
        options = {
            "loss of active material": "stress-driven",
            "particle mechanics": "swelling only",
            "stress-induced diffusion": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_asymmetric_negative(self):
        options = {
            "loss of active material": ("asymmetric stress-driven", "none"),
            "particle mechanics": ("swelling only", "none"),
            "stress-induced diffusion": ("true", "false"),
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_asymmetric_positive(self):
        options = {
            "loss of active material": ("none", "asymmetric stress-driven"),
            "particle mechanics": ("none", "swelling only"),
            "stress-induced diffusion": ("false", "true"),
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_asymmetric_both(self):
        options = {
            "loss of active material": "asymmetric stress-driven",
            "particle mechanics": "swelling only",
            "stress-induced diffusion": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_reaction(self):
        options = {"loss of active material": "reaction-driven"}
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_reaction(self):
        options = {
            "loss of active material": "stress and reaction-driven",
            "particle mechanics": "swelling only",
            "stress-induced diffusion": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_reaction_asymmetric_negative(self):
        options = {
            "loss of active material": (
                "asymmetric stress and reaction-driven",
                "none",
            ),
            "particle mechanics": ("swelling only", "none"),
            "stress-induced diffusion": ("true", "false"),
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_reaction_asymmetric_positive(self):
        options = {
            "loss of active material": (
                "none",
                "asymmetric stress and reaction-driven",
            ),
            "particle mechanics": ("none", "swelling only"),
            "stress-induced diffusion": ("false", "true"),
        }
        self.check_well_posedness(options)

    def test_well_posed_loss_active_material_stress_reaction_asymmetric_both(self):
        options = {
            "loss of active material": "asymmetric stress and reaction-driven",
            "particle mechanics": "swelling only",
            "stress-induced diffusion": "true",
        }
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
        options = {
            "SEI": "constant",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_reaction_limited(self):
        options = {
            "SEI": "reaction limited",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_asymmetric_sei_reaction_limited(self):
        options = {
            "SEI": "reaction limited (asymmetric)",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
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
        options = {
            "SEI": "solvent-diffusion limited",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_electron_migration_limited(self):
        options = {
            "SEI": "electron-migration limited",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_interstitial_diffusion_limited(self):
        options = {
            "SEI": "interstitial-diffusion limited",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_ec_reaction_limited(self):
        options = {
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_asymmetric_ec_reaction_limited(self):
        options = {
            "SEI": "ec reaction limited (asymmetric)",
            "SEI porosity change": "true",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_VonKolzenberg2020_model(self):
        options = {
            "SEI": "VonKolzenberg2020",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_tunnelling_limited(self):
        options = {
            "SEI": "tunnelling limited",
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_mechanics_negative_cracking(self):
        options = {
            "particle mechanics": ("swelling and cracking", "none"),
            "stress-induced diffusion": ("true", "false"),
        }
        self.check_well_posedness(options)

    def test_well_posed_mechanics_positive_cracking(self):
        options = {
            "particle mechanics": ("none", "swelling and cracking"),
            "stress-induced diffusion": ("false", "true"),
        }
        self.check_well_posedness(options)

    def test_well_posed_mechanics_both_cracking(self):
        options = {
            "particle mechanics": "swelling and cracking",
            "stress-induced diffusion": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_mechanics_both_swelling_only(self):
        options = {
            "particle mechanics": "swelling only",
            "stress-induced diffusion": "true",
        }
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
            "SEI film resistance": "distributed",
            "stress-induced diffusion": "true",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_solvent_diffusion_limited_on_cracks(self):
        options = {
            "SEI": "solvent-diffusion limited",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
            "SEI film resistance": "distributed",
            "stress-induced diffusion": "true",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_electron_migration_limited_on_cracks(self):
        options = {
            "SEI": "electron-migration limited",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
            "SEI film resistance": "distributed",
            "stress-induced diffusion": "true",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_interstitial_diffusion_limited_on_cracks(self):
        options = {
            "SEI": "interstitial-diffusion limited",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
            "SEI film resistance": "distributed",
            "stress-induced diffusion": "true",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_ec_reaction_limited_on_cracks(self):
        options = {
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
            "SEI on cracks": "true",
            "particle mechanics": "swelling and cracking",
            "SEI film resistance": "distributed",
            "stress-induced diffusion": "true",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_reversible_plating(self):
        options = {"lithium plating": "reversible"}
        self.check_well_posedness(options)

    def test_well_posed_irreversible_plating(self):
        options = {"lithium plating": "irreversible"}
        self.check_well_posedness(options)

    def test_well_posed_partially_reversible_plating(self):
        options = {
            "lithium plating": "partially reversible",
            "SEI": "constant",
            "SEI film resistance": "none",
        }
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
            "SEI": "constant",
            "SEI film resistance": "none",
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

    def test_well_posed_external_circuit_function_1plus1D(self):
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

        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "operating mode": external_circuit_function,
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_function_2plus1D(self):
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

        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "operating mode": external_circuit_function,
            "cell geometry": "pouch",
        }
        self.check_well_posedness(options)

    def test_well_posed_particle_phases(self):
        options = {"particle phases": "2", "surface form": "algebraic"}
        self.check_well_posedness(options)

        options = {"particle phases": ("2", "1"), "surface form": "algebraic"}
        self.check_well_posedness(options)

        options = {"particle phases": ("1", "2"), "surface form": "algebraic"}
        self.check_well_posedness(options)

    def test_well_posed_particle_phases_thermal(self):
        options = {
            "particle phases": "2",
            "thermal": "lumped",
            "surface form": "algebraic",
        }
        self.check_well_posedness(options)

    def test_well_posed_particle_phases_sei(self):
        options = {
            "particle phases": "2",
            "SEI": "ec reaction limited",
            "SEI film resistance": "distributed",
            "surface form": "algebraic",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_current_sigmoid_ocp(self):
        options = {"open-circuit potential": "current sigmoid"}
        self.check_well_posedness(options)

    def test_well_posed_one_state_differential_capacity_hysteresis_ocp(self):
        options = {
            "open-circuit potential": "one-state differential capacity hysteresis"
        }
        self.check_well_posedness(options)

    def test_well_posed_one_state_hysteresis_ocp(self):
        options = {"open-circuit potential": "one-state hysteresis"}
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

    def test_well_posed_psd_swelling_and_cracking(self):
        options = {
            "particle size": "distribution",
            "particle mechanics": "swelling and cracking",
            "surface form": "algebraic",
            "stress-induced diffusion": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_psd_swelling_only(self):
        options = {
            "particle size": "distribution",
            "particle mechanics": "swelling only",
            "surface form": "algebraic",
            "stress-induced diffusion": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_psd_hysteresis_thermal(self):
        # Regression test: hysteresis OCP + particle size distribution + a
        # non-isothermal thermal submodel previously raised a DomainError in
        # the thermal hysteresis-heating term because the "equilibrium
        # open-circuit potential [V]" variable was left on the particle-size
        # domain while "open-circuit potential [V]" was size-averaged.
        options = {
            "open-circuit potential": "one-state hysteresis",
            "particle size": "distribution",
            "surface form": "algebraic",
            "thermal": "lumped",
        }
        self.check_well_posedness(options)

    def test_well_posed_psd_single_ocp_thermal(self):
        # Regression test: for "single" OCP + particle size distribution, the
        # "equilibrium open-circuit potential [V]" variable used to be
        # published on the particle-size domain. The thermal submodel skips
        # the hysteresis branch for "single" so this did not raise today, but
        # the variable itself was on the wrong domain; this test pins the
        # corrected electrode-domain shape.
        options = {
            "particle size": "distribution",
            "surface form": "algebraic",
            "thermal": "lumped",
        }
        model = self.model(options)
        model.check_well_posedness()
        v = model.variables["Positive electrode equilibrium open-circuit potential [V]"]
        assert v.domains["primary"] == ["positive electrode"]

    def test_well_posed_psd_msmr_thermal(self):
        # Regression test: MSMR + particle size distribution + thermal
        # previously failed building Q_hys because "equilibrium open-circuit
        # potential [V]" was on the particle-size domain.
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "intercalation kinetics": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "particle size": "distribution",
            "surface form": "differential",
            "thermal": "lumped",
        }
        self.check_well_posedness(options)

    def test_well_posed_transport_efficiency_Bruggeman(self):
        options = {"transport efficiency": "Bruggeman"}
        self.check_well_posedness(options)

    def test_well_posed_transport_efficiency_ordered_packing(self):
        options = {"transport efficiency": "ordered packing"}
        self.check_well_posedness(options)

    def test_well_posed_transport_efficiency_overlapping_spheres(self):
        options = {"transport efficiency": "overlapping spheres"}
        self.check_well_posedness(options)

    def test_well_posed_transport_efficiency_random_overlapping_cylinders(self):
        options = {"transport efficiency": "random overlapping cylinders"}
        self.check_well_posedness(options)

    def test_well_posed_transport_efficiency_heterogeneous_catalyst(self):
        options = {"transport efficiency": "heterogeneous catalyst"}
        self.check_well_posedness(options)

    def test_well_posed_transport_efficiency_cation_exchange_membrane(self):
        options = {"transport efficiency": "cation-exchange membrane"}
        self.check_well_posedness(options)

    def test_well_posed_transport_efficiency_hyperbola(self):
        options = {"transport efficiency": "hyperbola of revolution"}
        self.check_well_posedness(options)

    def test_well_posed_transport_efficiency_tortuosity_factor(self):
        options = {"transport efficiency": "tortuosity factor"}
        self.check_well_posedness(options)

    def test_well_posed_composite_kinetic_hysteresis(self):
        options = {
            "particle phases": ("2", "1"),
            "exchange-current density": (
                ("current sigmoid", "single"),
                "current sigmoid",
            ),
            "open-circuit potential": (("current sigmoid", "single"), "single"),
            "surface form": "algebraic",
        }
        self.check_well_posedness(options)

    def test_well_posed_composite_diffusion_hysteresis(self):
        options = {
            "particle phases": ("2", "1"),
            "diffusivity": (("current sigmoid", "current sigmoid"), "current sigmoid"),
            "open-circuit potential": (("current sigmoid", "single"), "single"),
            "surface form": "algebraic",
        }
        self.check_well_posedness(options)

    def test_well_posed_composite_different_degradation(self):
        # phases have same degradation
        options = {
            "particle phases": ("2", "1"),
            "SEI": ("ec reaction limited", "none"),
            "SEI porosity change": "true",
            "lithium plating": ("reversible", "none"),
            "open-circuit potential": (("current sigmoid", "single"), "single"),
            "SEI film resistance": "distributed",
            "surface form": "algebraic",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)
        # phases have different degradation
        options = {
            "particle phases": ("2", "1"),
            "SEI": (("ec reaction limited", "solvent-diffusion limited"), "none"),
            "SEI porosity change": "true",
            "lithium plating": (("reversible", "irreversible"), "none"),
            "open-circuit potential": (("current sigmoid", "single"), "single"),
            "SEI film resistance": "distributed",
            "surface form": "algebraic",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)
        # one of the phases has no degradation
        options = {
            "particle phases": ("2", "1"),
            "SEI": (("none", "solvent-diffusion limited"), "none"),
            "lithium plating": (("none", "irreversible"), "none"),
            "SEI film resistance": "distributed",
            "surface form": "algebraic",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_composite_LAM(self):
        # phases with LAM degradation
        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
            "SEI": "solvent-diffusion limited",
            "loss of active material": "reaction-driven",
            "SEI film resistance": "distributed",
            "surface form": "algebraic",
            "total interfacial current density as a state": "true",
        }
        self.check_well_posedness(options)

        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
            "loss of active material": "stress-driven",
            "particle mechanics": "swelling only",
            "stress-induced diffusion": "true",
            "surface form": "algebraic",
        }
        self.check_well_posedness(options)

    def test_well_posed_composite_differential_surface_form(self):
        options = {
            "particle phases": ("2", "2"),
            "surface form": "differential",
        }
        self.check_well_posedness(options)

    def test_well_posed_composite_algebraic_surface_form(self):
        options = {
            "particle phases": ("2", "2"),
            "surface form": "algebraic",
        }
        self.check_well_posedness(options)
