#
# Lead-acid LOQS model
#
import pybamm
from .base_lead_acid_model import BaseModel


class LOQS(BaseModel):
    """Leading-Order Quasi-Static model for lead-acid, from [1]_.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model.
    name : str, optional
        The name of the model.
    build :  bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster lead-acid
           battery simulations from porous-electrode theory: Part II. Asymptotic
           analysis. Journal of The Electrochemical Society 166.12 (2019), A2372–A2382.


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, options=None, name="LOQS model", build=True):
        super().__init__(options, name)

        self.set_external_circuit_submodel()
        self.set_reactions()
        self.set_interfacial_submodel()
        self.set_convection_submodel()
        self.set_porosity_submodel()
        self.set_tortuosity_submodels()
        self.set_negative_electrode_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()
        self.set_side_reaction_submodels()
        self.set_current_collector_submodel()

        if build:
            self.build_model()

        if self.options["dimensionality"] == 0:
            self.use_jacobian = False

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """
        if self.options["operating mode"] == "current":
            self.submodels[
                "leading order external circuit"
            ] = pybamm.external_circuit.LeadingOrderCurrentControl(self.param)
        elif self.options["operating mode"] == "voltage":
            self.submodels[
                "leading order external circuit"
            ] = pybamm.external_circuit.LeadingOrderVoltageFunctionControl(self.param)
        elif self.options["operating mode"] == "power":
            self.submodels[
                "leading order external circuit"
            ] = pybamm.external_circuit.LeadingOrderPowerFunctionControl(self.param)
        elif callable(self.options["operating mode"]):
            self.submodels[
                "leading order external circuit"
            ] = pybamm.external_circuit.LeadingOrderFunctionControl(
                self.param, self.options["operating mode"]
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

        self.submodels["leading-order porosity"] = pybamm.porosity.LeadingOrder(
            self.param
        )

    def set_tortuosity_submodels(self):
        self.submodels[
            "leading-order electrolyte tortuosity"
        ] = pybamm.tortuosity.Bruggeman(self.param, "Electrolyte")
        self.submodels[
            "leading-order electrode tortuosity"
        ] = pybamm.tortuosity.Bruggeman(self.param, "Electrode")

    def set_convection_submodel(self):

        if self.options["convection"] is False:
            self.submodels["leading-order convection"] = pybamm.convection.NoConvection(
                self.param
            )
        if self.options["convection"] is True:
            self.submodels["leading-order convection"] = pybamm.convection.LeadingOrder(
                self.param
            )

    def set_interfacial_submodel(self):

        if self.options["surface form"] is False:
            self.submodels[
                "leading-order negative interface"
            ] = pybamm.interface.lead_acid.InverseButlerVolmer(self.param, "Negative")
            self.submodels[
                "leading-order positive interface"
            ] = pybamm.interface.lead_acid.InverseButlerVolmer(self.param, "Positive")
        else:
            self.submodels[
                "leading-order negative interface"
            ] = pybamm.interface.lead_acid.ButlerVolmer(self.param, "Negative")

            self.submodels[
                "leading-order positive interface"
            ] = pybamm.interface.lead_acid.ButlerVolmer(self.param, "Positive")
        self.reaction_submodels = {
            "Negative": [self.submodels["leading-order negative interface"]],
            "Positive": [self.submodels["leading-order positive interface"]],
        }

    def set_negative_electrode_submodel(self):

        self.submodels[
            "leading-order negative electrode"
        ] = pybamm.electrode.ohm.LeadingOrder(self.param, "Negative")

    def set_positive_electrode_submodel(self):

        self.submodels[
            "leading-order positive electrode"
        ] = pybamm.electrode.ohm.LeadingOrder(self.param, "Positive")

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell
        surf_form = electrolyte.conductivity.surface_potential_form

        if self.options["surface form"] is False:
            self.submodels[
                "leading-order electrolyte conductivity"
            ] = electrolyte.conductivity.LeadingOrder(self.param)

        elif self.options["surface form"] == "differential":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    "leading-order " + domain.lower() + " electrolyte conductivity"
                ] = surf_form.LeadingOrderDifferential(
                    self.param, domain, self.reactions
                )

        elif self.options["surface form"] == "algebraic":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    "leading-order " + domain.lower() + " electrolyte conductivity"
                ] = surf_form.LeadingOrderAlgebraic(self.param, domain, self.reactions)

        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.LeadingOrder(
            self.param, self.reactions
        )

    def set_side_reaction_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels[
                "leading-order oxygen diffusion"
            ] = pybamm.oxygen_diffusion.LeadingOrder(self.param, self.reactions)
            self.submodels[
                "leading-order positive oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.ForwardTafel(self.param, "Positive")
            self.submodels[
                "leading-order negative oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.LeadingOrderDiffusionLimited(
                self.param, "Negative"
            )
        else:
            self.submodels[
                "leading-order oxygen diffusion"
            ] = pybamm.oxygen_diffusion.NoOxygen(self.param)
            self.submodels[
                "leading-order positive oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.NoReaction(self.param, "Positive")
            self.submodels[
                "leading-order negative oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.NoReaction(self.param, "Negative")
        self.reaction_submodels["Negative"].append(
            self.submodels["leading-order negative oxygen interface"]
        )
        self.reaction_submodels["Positive"].append(
            self.submodels["leading-order positive oxygen interface"]
        )
