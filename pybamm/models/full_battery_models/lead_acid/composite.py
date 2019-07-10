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
        # Average interface submodel to get average first-order potential differences
        self.set_average_interfacial_submodel()
        # Electrolyte and solid submodels to get full first-order potentials
        self.set_electrolyte_conductivity_submodel()
        self.set_solid_submodel()
        # Update interface, porosity and convection with full potentials
        self.set_full_interface_submodel()
        self.set_full_convection_submodel()
        self.set_full_porosity_submodel()
        self.set_thermal_submodel()

        self.build_model()

        import ipdb

        ipdb.set_trace()
        # # Massive hack for consistent delta_phi = phi_s - phi_e
        # # This needs to be corrected
        # for domain in ["Negative", "Positive"]:
        #     phi_s = self.variables[domain + " electrode potential"]
        #     phi_e = self.variables[domain + " electrolyte potential"]
        #     delta_phi = phi_s - phi_e
        #     s = self.submodels[domain.lower() + " interface"]
        #     var = s._get_standard_surface_potential_difference_variables(delta_phi)
        #     self.variables.update(var)

    def set_leading_order_model(self):
        leading_order_model = pybamm.lead_acid.LOQS(self.options)
        self.update(leading_order_model)
        for variable in [
            "Average electrolyte concentration",
            "Average negative electrode surface potential difference",
            "Average positive electrode surface potential difference",
        ]:
            self.variables[
                "Leading-order " + variable.lower()
            ] = leading_order_model.variables[variable]

    def set_current_collector_submodel(self):
        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param
        )

    def set_electrolyte_diffusion_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        if self.options["higher-order concentration"] == "composite":
            self.submodels["electrolyte diffusion"] = electrolyte.diffusion.Full(
                self.param, self.reactions
            )
        elif self.options["higher-order concentration"] == "first-order":
            self.submodels["electrolyte diffusion"] = electrolyte.diffusion.FirstOrder(
                self.param, self.reactions
            )

    def set_average_interfacial_submodel(self):
        self.submodels[
            "average negative interface"
        ] = pybamm.interface.lead_acid.InverseFirstOrderButlerVolmer(
            self.param, "Negative"
        )
        self.submodels[
            "average positive interface"
        ] = pybamm.interface.lead_acid.InverseFirstOrderButlerVolmer(
            self.param, "Positive"
        )

    def set_electrolyte_conductivity_submodel(self):
        self.submodels[
            "electrolyte conductivity"
        ] = pybamm.electrolyte.stefan_maxwell.conductivity.FirstOrder(self.param)

    def set_solid_submodel(self):
        self.submodels["negative electrode"] = pybamm.electrode.ohm.Composite(
            self.param, "Negative"
        )
        self.submodels["positive electrode"] = pybamm.electrode.ohm.Composite(
            self.param, "Positive"
        )

    def set_full_interface_submodel(self):
        """
        Set full interface submodel, to get spatially heterogeneous interfacial current
        densities
        """
        self.submodels["negative interface"] = pybamm.interface.lead_acid.ButlerVolmer(
            self.param, "Negative"
        )
        self.submodels["positive interface"] = pybamm.interface.lead_acid.ButlerVolmer(
            self.param, "Positive"
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
        # Update options
        if options is None:
            options = {}
        options = {**options, "higher-order concentration": "first-order"}
        super().__init__(options, name)


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
        # Update options
        if options is None:
            options = {}
        options = {**options, "higher-order concentration": "composite"}
        super().__init__(options, name)
