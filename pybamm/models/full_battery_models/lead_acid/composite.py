#
# Lead-acid Composite model
#
import pybamm
from .base_lead_acid_model import BaseModel


class Composite(BaseModel):
    """Composite model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Composite model"

        self.set_current_collector_submodel()
        self.set_interfacial_submodel()
        self.set_porosity_submodel()
        self.set_negative_electrode_submodel()
        self.set_convection_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()

        self.build_model()

        # Massive hack for consistent delta_phi = phi_s - phi_e
        # This needs to be corrected
        for domain in ["Negative", "Positive"]:
            phi_s = self.variables[domain + " electrode potential"]
            phi_e = self.variables[domain + " electrolyte potential"]
            delta_phi = phi_s - phi_e
            delta_phi_av = pybamm.average(delta_phi)
            s = self.submodels[domain.lower() + " interface"]
            var = s._get_standard_surface_potential_difference_variables(
                delta_phi, delta_phi_av
            )
            self.variables.update(var)

    def set_current_collector_submodel(self):
        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param
        )

    def set_porosity_submodel(self):
        self.submodels["porosity"] = pybamm.porosity.LeadingOrder(self.param)

    def set_convection_submodel(self):
        if self.options["convection"] is False:
            self.submodels["convection"] = pybamm.convection.NoConvection(self.param)
        if self.options["convection"] is True:
            self.submodels["convection"] = pybamm.convection.LeadingOrder(self.param)

    def set_interfacial_submodel(self):
        self.submodels[
            "negative interface"
        ] = pybamm.interface.inverse_butler_volmer.LeadAcid(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.inverse_butler_volmer.LeadAcid(self.param, "Positive")

    def set_negative_electrode_submodel(self):
        self.submodels["negative electrode"] = pybamm.electrode.ohm.CombinedOrder(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):
        self.submodels["positive electrode"] = pybamm.electrode.ohm.CombinedOrder(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.Full(
            self.param, ocp=True
        )

        self.submodels[
            "electrolyte conductivity"
        ] = electrolyte.conductivity.CombinedOrder(self.param)

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self.options["capacitance"] == "algebraic":
            return pybamm.ScikitsDaeSolver()
        else:
            return pybamm.ScipySolver()
