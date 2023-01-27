#
# Lead-acid Full model
#
import pybamm
from .base_lead_acid_model import BaseModel


class Full(BaseModel):
    """
    Porous electrode model for lead-acid, from [1]_, based on the Newman-Tiedemann
    model.
    See :class:`pybamm.lead_acid.BaseModel` for more details.


    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster lead-acid
           battery simulations from porous-electrode theory: Part II. Asymptotic
           analysis. Journal of The Electrochemical Society 166.12 (2019), A2372–A2382.


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, options=None, name="Full model", build=True):
        super().__init__(options, name)

        self.set_external_circuit_submodel()
        self.set_open_circuit_potential_submodel()
        self.set_intercalation_kinetics_submodel()
        self.set_interface_utilisation_submodel()
        self.set_porosity_submodel()
        self.set_active_material_submodel()
        self.set_transport_efficiency_submodels()
        self.set_convection_submodel()
        self.set_electrolyte_submodel()
        self.set_solid_submodel()
        self.set_thermal_submodel()
        self.set_side_reaction_submodels()
        self.set_current_collector_submodel()
        self.set_sei_submodel()
        self.set_lithium_plating_submodel()
        self.set_total_interface_submodel()

        if build:
            self.build_model()

        pybamm.citations.register("Sulzer2019physical")

    def set_porosity_submodel(self):
        self.submodels["porosity"] = pybamm.porosity.ReactionDrivenODE(
            self.param, self.options, False
        )

    def set_convection_submodel(self):
        if self.options["convection"] == "none":
            self.submodels[
                "transverse convection"
            ] = pybamm.convection.transverse.NoConvection(self.param)
            self.submodels[
                "through-cell convection"
            ] = pybamm.convection.through_cell.NoConvection(self.param)
        else:
            if self.options["convection"] == "uniform transverse":
                self.submodels[
                    "transverse convection"
                ] = pybamm.convection.transverse.Uniform(self.param)
            elif self.options["convection"] == "full transverse":
                self.submodels[
                    "transverse convection"
                ] = pybamm.convection.transverse.Full(self.param)
            self.submodels[
                "through-cell convection"
            ] = pybamm.convection.through_cell.Full(self.param)

    def set_intercalation_kinetics_submodel(self):
        for domain in ["negative", "positive"]:
            intercalation_kinetics = self.get_intercalation_kinetics(domain)
            self.submodels[f"{domain} interface"] = intercalation_kinetics(
                self.param, domain, "lead-acid main", self.options, "primary"
            )

    def set_solid_submodel(self):
        if self.options["surface form"] == "false":
            submod_n = pybamm.electrode.ohm.Full(self.param, "negative")
            submod_p = pybamm.electrode.ohm.Full(self.param, "positive")
        else:
            submod_n = pybamm.electrode.ohm.SurfaceForm(self.param, "negative")
            submod_p = pybamm.electrode.ohm.SurfaceForm(self.param, "positive")

        self.submodels["negative electrode potential"] = submod_n
        self.submodels["positive electrode potential"] = submod_p

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        self.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(
            self.param
        )

        if self.options["surface form"] == "false":
            self.submodels[
                "electrolyte conductivity"
            ] = pybamm.electrolyte_conductivity.Full(self.param)
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.FullDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.FullAlgebraic

        for domain in ["negative", "separator", "positive"]:
            self.submodels[f"{domain} surface potential difference"] = surf_model(
                self.param, domain, self.options
            )

    def set_side_reaction_submodels(self):
        if self.options["hydrolysis"] == "true":
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.Full(
                self.param
            )
            self.submodels["positive oxygen interface"] = pybamm.kinetics.ForwardTafel(
                self.param, "positive", "lead-acid oxygen", self.options, "primary"
            )
            self.submodels[
                "negative oxygen interface"
            ] = pybamm.kinetics.DiffusionLimited(
                self.param, "negative", "lead-acid oxygen", self.options, order="full"
            )
        else:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.NoOxygen(
                self.param
            )
            self.submodels["positive oxygen interface"] = pybamm.kinetics.NoReaction(
                self.param, "positive", "lead-acid oxygen", "primary"
            )
            self.submodels["negative oxygen interface"] = pybamm.kinetics.NoReaction(
                self.param, "negative", "lead-acid oxygen", "primary"
            )
