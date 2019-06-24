#
# Surface form of the Lead-acid Newman-tiedemann model
#
import pybamm
from ..base_lead_acid_model import BaseModel


class NewmanTiedemann(BaseModel):
    """Surface form of newman-tiedemann model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.

    **Extends:** :class:`LeadAcidBaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Newman Tiedeman model"

        self.set_reactions()

        self.set_current_collector_submodel()
        self.set_interfacial_submodel()
        self.set_porosity_submodel()
        self.set_convection_submodel()
        self.set_electrolyte_submodel()
        self.set_negative_electrode_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()

        self.build_model()

    def set_reactions(self):
        # Should probably refactor as this is a bit clunky at the moment
        # Maybe each reaction as a Reaction class so we can just list names of classes
        self.reactions = {
            "main": {
                "neg": {
                    "s_plus": self.param.s_n,
                    "j": "Average negative electrode interfacial current density",
                },
                "pos": {
                    "s_plus": self.param.s_p,
                    "j": "Average positive electrode interfacial current density",
                },
            }
        }

    def set_current_collector_submodel(self):
        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param, "Negative"
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
        ] = pybamm.interface.butler_volmer.surface_form.LeadAcid(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.butler_volmer.surface_form.LeadAcid(self.param, "Positive")

    def set_negative_electrode_submodel(self):
        self.submodels["negative electrode"] = pybamm.electrode.ohm.SurfaceForm(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):
        self.submodels["positive electrode"] = pybamm.electrode.ohm.SurfaceForm(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.FullModel(
            self.param, ocp=True
        )

        surf_form = electrolyte.conductivity.surface_potential_form

        if self.options["capacitance"] is False:
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + "electrolyte conductivity"
                ] = surf_form.LeadingOrderModel(self.param, domain)

        elif self.options["capacitance"] is True:
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + "electrolyte conductivity"
                ] = surf_form.LeadingOrderCapacitanceModel(self.param, domain)

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """

        return pybamm.ScikitsDaeSolver()
