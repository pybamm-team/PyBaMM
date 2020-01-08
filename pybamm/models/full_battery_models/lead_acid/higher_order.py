#
# Lead-acid higher-order models (FOQS and Composite)
#
import pybamm
from .base_lead_acid_model import BaseModel


class BaseHigherOrderModel(BaseModel):
    """Base model for higher-order models for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

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

    def __init__(self, options=None, name="Composite model", build=True):
        super().__init__(options, name)

        self.set_external_circuit_submodel()
        self.set_leading_order_model()
        self.set_reactions()
        # Electrolyte submodel to get first-order concentrations
        self.set_electrolyte_diffusion_submodel()
        self.set_other_species_diffusion_submodels()
        # Average interface submodel to get average first-order potential differences
        self.set_average_interfacial_submodel()
        # Electrolyte and solid submodels to get full first-order potentials
        self.set_negative_electrode_submodel()
        self.set_electrolyte_conductivity_submodel()
        self.set_positive_electrode_submodel()
        # Update interface, porosity and convection with full potentials
        self.set_full_interface_submodel()
        self.set_full_convection_submodel()
        self.set_full_porosity_submodel()
        self.set_tortuosity_submodels()
        self.set_thermal_submodel()
        self.set_current_collector_submodel()

        if build:
            self.build_model()

    def set_current_collector_submodel(self):
        cc = pybamm.current_collector

        if self.options["current collector"] == "uniform":
            submodel = cc.Uniform(self.param)
        elif self.options["current collector"] == "potential pair quite conductive":
            if self.options["dimensionality"] == 1:
                submodel = cc.QuiteConductivePotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = cc.QuiteConductivePotentialPair2plus1D(self.param)
        elif self.options["current collector"] == "potential pair":
            if self.options["dimensionality"] == 1:
                submodel = cc.CompositePotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = cc.CompositePotentialPair2plus1D(self.param)
        self.submodels["current collector"] = submodel

    def set_leading_order_model(self):
        leading_order_model = pybamm.lead_acid.LOQS(
            self.options, name="LOQS model (for composite model)"
        )
        self.update(leading_order_model)
        self.reaction_submodels = leading_order_model.reaction_submodels

        # Leading-order variables
        leading_order_variables = {}
        for variable in self.variables.keys():
            leading_order_variables[
                "Leading-order " + variable.lower()
            ] = leading_order_model.variables[variable]
        self.variables.update(leading_order_variables)
        self.variables[
            "Leading-order electrolyte concentration change"
        ] = leading_order_model.rhs[
            leading_order_model.variables["X-averaged electrolyte concentration"]
        ]

    def set_average_interfacial_submodel(self):
        self.submodels[
            "x-averaged negative interface"
        ] = pybamm.interface.lead_acid.InverseFirstOrderKinetics(self.param, "Negative")
        self.submodels[
            "x-averaged negative interface"
        ].reaction_submodels = self.reaction_submodels["Negative"]
        self.submodels[
            "x-averaged positive interface"
        ] = pybamm.interface.lead_acid.InverseFirstOrderKinetics(self.param, "Positive")
        self.submodels[
            "x-averaged positive interface"
        ].reaction_submodels = self.reaction_submodels["Positive"]

    def set_electrolyte_conductivity_submodel(self):
        self.submodels[
            "electrolyte conductivity"
        ] = pybamm.electrolyte.stefan_maxwell.conductivity.FirstOrder(self.param)

    def set_negative_electrode_submodel(self):
        self.submodels["negative electrode"] = pybamm.electrode.ohm.Composite(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):
        self.submodels["positive electrode"] = pybamm.electrode.ohm.Composite(
            self.param, "Positive"
        )

    def set_full_interface_submodel(self):
        """
        Set full interface submodel, to get spatially heterogeneous interfacial current
        densities
        """
        # Main reaction
        self.submodels[
            "negative interface"
        ] = pybamm.interface.lead_acid.FirstOrderButlerVolmer(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.lead_acid.FirstOrderButlerVolmer(self.param, "Positive")

        # Oxygen
        if "oxygen" in self.options["side reactions"]:
            self.submodels[
                "positive oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.FirstOrderForwardTafel(
                self.param, "Positive"
            )
            self.submodels[
                "negative oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.FullDiffusionLimited(
                self.param, "Negative"
            )

    def set_full_convection_submodel(self):
        """
        Update convection submodel, now that we have the spatially heterogeneous
        interfacial current densities
        """
        if self.options["convection"] is False:
            self.submodels["full convection"] = pybamm.convection.NoConvection(
                self.param
            )
        if self.options["convection"] is True:
            self.submodels["full convection"] = pybamm.convection.Composite(self.param)

    def set_full_porosity_submodel(self):
        """
        Update porosity submodel, now that we have the spatially heterogeneous
        interfacial current densities
        """
        self.submodels["full porosity"] = pybamm.porosity.Full(self.param)


class FOQS(BaseHigherOrderModel):
    """First-order quasi-static model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

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

    **Extends:** :class:`pybamm.lead_acid.BaseHigherOrderModel`
    """

    def __init__(self, options=None, name="FOQS model", build=True):
        super().__init__(options, name, build=build)

    def set_electrolyte_diffusion_submodel(self):
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte.stefan_maxwell.diffusion.FirstOrder(
            self.param, self.reactions
        )

    def set_other_species_diffusion_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.FirstOrder(
                self.param, self.reactions
            )

    def set_full_porosity_submodel(self):
        """
        Update porosity submodel, now that we have the spatially heterogeneous
        interfacial current densities
        """
        # TODO: fix shape for jacobian
        pass


class Composite(BaseHigherOrderModel):
    """Composite model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    **Extends:** :class:`pybamm.lead_acid.BaseHigherOrderModel`
    """

    def __init__(self, options=None, name="Composite model", build=True):
        super().__init__(options, name, build=build)

    def set_electrolyte_diffusion_submodel(self):
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte.stefan_maxwell.diffusion.Composite(
            self.param, self.reactions
        )

    def set_other_species_diffusion_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.Composite(
                self.param, self.reactions
            )

    def set_full_porosity_submodel(self):
        """
        Update porosity submodel, now that we have the spatially heterogeneous
        interfacial current densities
        """
        self.submodels["full porosity"] = pybamm.porosity.Full(self.param)


class CompositeExtended(BaseHigherOrderModel):
    """Extended composite model for lead-acid, from [2]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

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
    .. [2] V Sulzer. Mathematical modelling of lead-acid batteries. PhD thesis,
           University of Oxford, 2019.


    **Extends:** :class:`pybamm.lead_acid.BaseHigherOrderModel`
    """

    def __init__(self, options=None, name="Extended composite model", build=True):
        super().__init__(options, name, build=build)

    def set_electrolyte_diffusion_submodel(self):
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte.stefan_maxwell.diffusion.Composite(
            self.param, self.reactions, extended=True
        )

    def set_other_species_diffusion_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.Composite(
                self.param, self.reactions, extended=True
            )

    def set_full_porosity_submodel(self):
        """
        Update porosity submodel, now that we have the spatially heterogeneous
        interfacial current densities
        """
        self.submodels["full porosity"] = pybamm.porosity.Full(self.param)
