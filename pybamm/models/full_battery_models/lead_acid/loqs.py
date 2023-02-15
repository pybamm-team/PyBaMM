#
# Lead-acid LOQS model
#
import pybamm
from .base_lead_acid_model import BaseModel


class LOQS(BaseModel):
    """
    Leading-Order Quasi-Static model for lead-acid, from [1]_.
    See :class:`pybamm.lead_acid.BaseModel` for more details.

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster lead-acid
           battery simulations from porous-electrode theory: Part II. Asymptotic
           analysis. Journal of The Electrochemical Society 166.12 (2019), A2372–A2382.
    """

    def __init__(self, options=None, name="LOQS model", build=True):
        super().__init__(options, name)

        self.set_external_circuit_submodel()
        self.set_open_circuit_potential_submodel()
        self.set_intercalation_kinetics_submodel()
        self.set_interface_utilisation_submodel()
        self.set_convection_submodel()
        self.set_porosity_submodel()
        self.set_active_material_submodel()
        self.set_transport_efficiency_submodels()
        self.set_electrolyte_submodel()
        self.set_electrode_submodels()
        self.set_thermal_submodel()
        self.set_side_reaction_submodels()
        self.set_current_collector_submodel()
        self.set_sei_submodel()
        self.set_lithium_plating_submodel()
        self.set_total_interface_submodel()

        if build:
            self.build_model()

        if self.options["dimensionality"] == 0:
            self.use_jacobian = False

        pybamm.citations.register("Sulzer2019asymptotic")

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """
        if self.options["operating mode"] == "current":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.ExplicitCurrentControl(self.param, self.options)
        elif self.options["operating mode"] == "voltage":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.VoltageFunctionControl(self.param, self.options)
        elif self.options["operating mode"] == "power":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.PowerFunctionControl(self.param, self.options)
        elif callable(self.options["operating mode"]):
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.FunctionControl(
                self.param, self.options["operating mode"], self.options
            )

    def set_current_collector_submodel(self):
        if self.options["current collector"] in [
            "uniform",
            "potential pair quite conductive",
        ]:
            submodel = pybamm.current_collector.Uniform(self.param)
        elif self.options["current collector"] == "potential pair":
            if self.options["dimensionality"] == 1:
                submodel = pybamm.current_collector.PotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = pybamm.current_collector.PotentialPair2plus1D(self.param)
        self.submodels["leading-order current collector"] = submodel

    def set_porosity_submodel(self):
        self.submodels["leading-order porosity"] = pybamm.porosity.ReactionDrivenODE(
            self.param, self.options, True
        )

    def set_convection_submodel(self):
        if self.options["convection"] == "none":
            self.submodels[
                "leading-order transverse convection"
            ] = pybamm.convection.transverse.NoConvection(self.param)
            self.submodels[
                "leading-order through-cell convection"
            ] = pybamm.convection.through_cell.NoConvection(self.param)
        else:
            if self.options["convection"] == "uniform transverse":
                self.submodels[
                    "leading-order transverse convection"
                ] = pybamm.convection.transverse.Uniform(self.param)
            elif self.options["convection"] == "full transverse":
                self.submodels[
                    "leading-order transverse convection"
                ] = pybamm.convection.transverse.Full(self.param)
            self.submodels[
                "leading-order through-cell convection"
            ] = pybamm.convection.through_cell.Explicit(self.param)

    def set_intercalation_kinetics_submodel(self):
        if self.options["surface form"] == "false":
            self.submodels[
                "leading-order negative interface"
            ] = pybamm.kinetics.InverseButlerVolmer(
                self.param, "negative", "lead-acid main", self.options
            )
            self.submodels[
                "leading-order positive interface"
            ] = pybamm.kinetics.InverseButlerVolmer(
                self.param, "positive", "lead-acid main", self.options
            )
            self.submodels[
                "negative interface current"
            ] = pybamm.kinetics.CurrentForInverseButlerVolmer(
                self.param, "negative", "lead-acid main"
            )
            self.submodels[
                "positive interface current"
            ] = pybamm.kinetics.CurrentForInverseButlerVolmer(
                self.param, "positive", "lead-acid main"
            )
        else:
            self.submodels[
                "leading-order negative interface"
            ] = pybamm.kinetics.SymmetricButlerVolmer(
                self.param, "negative", "lead-acid main", self.options, "primary"
            )

            self.submodels[
                "leading-order positive interface"
            ] = pybamm.kinetics.SymmetricButlerVolmer(
                self.param, "positive", "lead-acid main", self.options, "primary"
            )
        # always use forward Butler-Volmer for the reaction submodel to be passed to the
        # higher order model
        self.reaction_submodels = {
            "negative": [
                pybamm.kinetics.SymmetricButlerVolmer(
                    self.param, "negative", "lead-acid main", self.options, "primary"
                )
            ],
            "positive": [
                pybamm.kinetics.SymmetricButlerVolmer(
                    self.param, "positive", "lead-acid main", self.options, "primary"
                )
            ],
        }

    def set_electrode_submodels(self):
        self.submodels[
            "leading-order negative electrode potential"
        ] = pybamm.electrode.ohm.LeadingOrder(self.param, "negative")
        self.submodels[
            "leading-order positive electrode potential"
        ] = pybamm.electrode.ohm.LeadingOrder(self.param, "positive")

    def set_electrolyte_submodel(self):
        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if self.options["surface form"] == "false":
            self.submodels[
                "leading-order electrolyte conductivity"
            ] = pybamm.electrolyte_conductivity.LeadingOrder(self.param)
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.LeadingOrderDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.LeadingOrderAlgebraic

        for domain in ["negative", "positive"]:
            self.submodels[f"{domain} surface potential difference"] = surf_model(
                self.param, domain, self.options
            )

        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.LeadingOrder(self.param)

    def set_side_reaction_submodels(self):
        if self.options["hydrolysis"] == "true":
            self.submodels[
                "leading-order oxygen diffusion"
            ] = pybamm.oxygen_diffusion.LeadingOrder(self.param)
            self.submodels[
                "leading-order positive oxygen interface"
            ] = pybamm.kinetics.ForwardTafel(
                self.param, "positive", "lead-acid oxygen", self.options, "primary"
            )
            self.submodels[
                "leading-order negative oxygen interface"
            ] = pybamm.kinetics.DiffusionLimited(
                self.param,
                "negative",
                "lead-acid oxygen",
                self.options,
                order="leading",
            )
        else:
            self.submodels[
                "leading-order oxygen diffusion"
            ] = pybamm.oxygen_diffusion.NoOxygen(self.param)
            self.submodels[
                "leading-order negative oxygen interface"
            ] = pybamm.kinetics.NoReaction(
                self.param, "negative", "lead-acid oxygen", "primary"
            )
            self.submodels[
                "leading-order positive oxygen interface"
            ] = pybamm.kinetics.NoReaction(
                self.param, "positive", "lead-acid oxygen", "primary"
            )
        self.reaction_submodels["negative"].append(
            self.submodels["leading-order negative oxygen interface"]
        )
        self.reaction_submodels["positive"].append(
            self.submodels["leading-order positive oxygen interface"]
        )
