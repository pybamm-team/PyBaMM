#
# Lead-acid Newman-Tiedemann model
#
import pybamm
from .base_lead_acid_model import BaseModel


class NewmanTiedemann(BaseModel):
    """Porous electrode model for lead-acid, from [1]_.

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: I. Physical Model.
           arXiv preprint arXiv:1902.01771, 2019.


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Newman-Tiedemann model"

        self.set_current_collector_submodel()
        self.set_interfacial_submodel()
        self.set_porosity_submodel()
        self.set_negative_electrode_submodel()
        self.set_convection_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()

        self.build_model()

    def set_current_collector_submodel(self):
        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param, "Negative"
        )

    def set_porosity_submodel(self):
        self.submodels["porosity"] = pybamm.porosity.Full(self.param)

    def set_convection_submodel(self):
        if self.options["convection"] is False:
            self.submodels["convection"] = pybamm.convection.NoConvection(self.param)
        if self.options["convection"] is True:
            self.submodels["convection"] = pybamm.convection.Full(self.param)

    def set_interfacial_submodel(self):
        self.submodels["negative interface"] = pybamm.interface.lead_acid.ButlerVolmer(
            self.param, "Negative"
        )
        self.submodels["positive interface"] = pybamm.interface.lead_acid.ButlerVolmer(
            self.param, "Positive"
        )

    def set_negative_electrode_submodel(self):
        self.submodels["negative electrode"] = pybamm.electrode.ohm.Full(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):
        self.submodels["positive electrode"] = pybamm.electrode.ohm.Full(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell
        surf_form = electrolyte.conductivity.surface_potential_form

        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.Full(self.param)

        if self.options["capacitance"] is False:
            self.submodels["electrolyte conductivity"] = electrolyte.conductivity.Full(
                self.param
            )
        elif self.options["capacitance"] == "differential":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.FullDifferential(self.param, domain)
        elif self.options["capacitance"] == "algebraic":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.FullAlgebraic(self.param, domain)

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self.options["capacitance"] == "differential":
            return pybamm.ScipySolver()
        else:
            return pybamm.ScikitsDaeSolver()
