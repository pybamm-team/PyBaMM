#
# Lead-acid higher-order models (FOQS and Composite)
#
import pybamm
from .base_lead_acid_model import BaseModel


class HigherOrderBaseModel(BaseModel):
    """Base model for higher-order models for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, options=None, name="Composite model"):
        super().__init__(options, name)

        self.set_leading_order_model()
        self.set_reactions()
        self.set_current_collector_submodel()
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
        self.set_thermal_submodel()

        self.build_model()

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

    def set_current_collector_submodel(self):

        if self.options["bc_options"]["dimensionality"] == 0:
            self.submodels["current collector"] = pybamm.current_collector.Uniform(
                self.param
            )
        elif self.options["bc_options"]["dimensionality"] == 1:
            self.submodels[
                "current collector"
            ] = pybamm.current_collector.surface_form.LeadingOrder(self.param)
        elif self.options["bc_options"]["dimensionality"] == 2:
            self.submodels[
                "current collector"
            ] = pybamm.current_collector.SingleParticlePotentialPair(self.param)

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

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self.options["surface form"] == "algebraic":
            return pybamm.ScikitsDaeSolver()
        else:
            return pybamm.ScipySolver()


class FOQS(HigherOrderBaseModel):
    """First-order quasi-static model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.


    **Extends:** :class:`pybamm.lead_acid.HigherOrderBaseModel`
    """

    def __init__(self, options=None, name="FOQS model"):
        super().__init__(options, name)

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


class Composite(HigherOrderBaseModel):
    """Composite model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.


    **Extends:** :class:`pybamm.lead_acid.HigherOrderBaseModel`
    """

    def __init__(self, options=None, name="Composite model"):
        super().__init__(options, name)

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


class CompositeExtended(HigherOrderBaseModel):
    """Extended composite model for lead-acid.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    **Extends:** :class:`pybamm.lead_acid.HigherOrderBaseModel`
    """

    def __init__(self, options=None, name="Extended composite model"):
        super().__init__(options, name)

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
