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
        self.set_active_material_submodel()
        self.set_tortuosity_submodels()
        self.set_thermal_submodel()
        self.set_current_collector_submodel()
        self.set_sei_submodel()
        self.set_lithium_plating_submodel()

        if build:
            self.build_model()

        pybamm.citations.register("Sulzer2019asymptotic")

    def set_current_collector_submodel(self):
        cc = pybamm.current_collector

        if self.options["current collector"] in ["uniform"]:
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
        self.leading_order_reaction_submodels = leading_order_model.reaction_submodels

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

        # Reset sums
        self.variables.update(
            {
                "Sum of electrolyte reaction source terms": 0,
                "Sum of negative electrode electrolyte reaction source terms": 0,
                "Sum of positive electrode electrolyte reaction source terms": 0,
                "Sum of x-averaged negative electrode "
                "electrolyte reaction source terms": 0,
                "Sum of x-averaged positive electrode "
                "electrolyte reaction source terms": 0,
                "Sum of interfacial current densities": 0,
                "Sum of negative electrode interfacial current densities": 0,
                "Sum of positive electrode interfacial current densities": 0,
                "Sum of x-averaged negative electrode interfacial current densities": 0,
                "Sum of x-averaged positive electrode interfacial current densities": 0,
            }
        )

    def set_average_interfacial_submodel(self):
        self.submodels[
            "x-averaged negative interface"
        ] = pybamm.interface.InverseFirstOrderKinetics(
            self.param, "Negative", self.leading_order_reaction_submodels["Negative"]
        )
        self.submodels[
            "x-averaged positive interface"
        ] = pybamm.interface.InverseFirstOrderKinetics(
            self.param, "Positive", self.leading_order_reaction_submodels["Positive"]
        )

    def set_electrolyte_conductivity_submodel(self):
        self.submodels[
            "electrolyte conductivity"
        ] = pybamm.electrolyte_conductivity.Composite(
            self.param, higher_order_terms="first-order"
        )

    def set_negative_electrode_submodel(self):
        self.submodels["negative electrode potential"] = pybamm.electrode.ohm.Composite(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):
        self.submodels["positive electrode potential"] = pybamm.electrode.ohm.Composite(
            self.param, "Positive"
        )

    def set_full_interface_submodel(self):
        """
        Set full interface submodel, to get spatially heterogeneous interfacial current
        densities
        """
        # Main reaction
        self.submodels["negative interface"] = pybamm.interface.FirstOrderKinetics(
            self.param,
            "Negative",
            pybamm.interface.ButlerVolmer(
                self.param, "Negative", "lead-acid main", self.options
            ),
        )
        self.submodels["positive interface"] = pybamm.interface.FirstOrderKinetics(
            self.param,
            "Positive",
            pybamm.interface.ButlerVolmer(
                self.param, "Positive", "lead-acid main", self.options
            ),
        )

        # Oxygen
        if "oxygen" in self.options["side reactions"]:
            self.submodels[
                "positive oxygen interface"
            ] = pybamm.interface.FirstOrderKinetics(
                self.param,
                "Positive",
                pybamm.interface.ForwardTafel(
                    self.param, "Positive", "lead-acid oxygen", self.options
                ),
            )
            self.submodels[
                "negative oxygen interface"
            ] = pybamm.interface.DiffusionLimited(
                self.param, "Negative", "lead-acid oxygen", order="composite"
            )

    def set_full_convection_submodel(self):
        """
        Update convection submodel, now that we have the spatially heterogeneous
        interfacial current densities
        """
        if self.options["convection"] != "none":
            self.submodels[
                "through-cell convection"
            ] = pybamm.convection.through_cell.Explicit(self.param)

    def set_full_porosity_submodel(self):
        """
        Update porosity submodel, now that we have the spatially heterogeneous
        interfacial current densities
        """
        self.submodels["full porosity"] = pybamm.porosity.Full(self.param, self.options)


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
        ] = pybamm.electrolyte_diffusion.FirstOrder(self.param)

    def set_other_species_diffusion_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.FirstOrder(
                self.param
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
        ] = pybamm.electrolyte_diffusion.Composite(self.param)

    def set_other_species_diffusion_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.Composite(
                self.param
            )

    def set_full_porosity_submodel(self):
        """
        Update porosity submodel, now that we have the spatially heterogeneous
        interfacial current densities
        """
        self.submodels["full porosity"] = pybamm.porosity.Full(self.param, self.options)


class CompositeExtended(Composite):
    """Extended composite model for lead-acid.
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

    def __init__(
        self, options=None, name="Extended composite model (distributed)", build=True
    ):
        super().__init__(options, name, build=build)

    def set_electrolyte_diffusion_submodel(self):
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.Composite(self.param, extended="distributed")

    def set_other_species_diffusion_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.Composite(
                self.param, extended="distributed"
            )


class CompositeAverageCorrection(Composite):
    """Extended composite model for lead-acid.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    **Extends:** :class:`pybamm.lead_acid.BaseHigherOrderModel`
    """

    def __init__(self, options=None, name="Extended composite model (average)"):
        super().__init__(options, name)

    def set_electrolyte_diffusion_submodel(self):
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.Composite(self.param, extended="average")

    def set_other_species_diffusion_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.Composite(
                self.param, extended="average"
            )
