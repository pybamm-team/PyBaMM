#
# Lead-acid LOQS model
#
import pybamm
from .base_lead_acid_model import BaseModel


class LOQS(BaseModel):
    """Leading-Order Quasi-Static model for lead-acid, from [1]_.

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "LOQS model"
        self.use_jacobian = False

        self.set_reactions()
        self.set_current_collector_submodel()
        self.set_interfacial_submodel()
        self.set_convection_submodel()
        self.set_porosity_submodel()
        self.set_negative_electrode_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()
        self.set_side_reaction_submodels()

        self.build_model()

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

        if self.options["surface form"] is False:
            self.submodels[
                "negative interface"
            ] = pybamm.interface.lead_acid.InverseButlerVolmer(self.param, "Negative")
            self.submodels[
                "positive interface"
            ] = pybamm.interface.lead_acid.InverseButlerVolmer(self.param, "Positive")
        else:
            self.submodels[
                "negative interface"
            ] = pybamm.interface.lead_acid.ButlerVolmer(self.param, "Negative")

            self.submodels[
                "positive interface"
            ] = pybamm.interface.lead_acid.ButlerVolmer(self.param, "Positive")

    def set_negative_electrode_submodel(self):

        self.submodels["negative electrode"] = pybamm.electrode.ohm.LeadingOrder(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):

        self.submodels["positive electrode"] = pybamm.electrode.ohm.LeadingOrder(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell
        surf_form = electrolyte.conductivity.surface_potential_form

        if self.options["surface form"] is False:
            self.submodels[
                "electrolyte conductivity"
            ] = electrolyte.conductivity.LeadingOrder(self.param)
        elif self.options["surface form"] == "differential":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.LeadingOrderDifferential(
                    self.param, domain, self.reactions
                )
        elif self.options["surface form"] == "algebraic":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.LeadingOrderAlgebraic(self.param, domain, self.reactions)

        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.LeadingOrder(
            self.param, self.reactions
        )

    def set_side_reaction_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels[
                "positive oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.ForwardTafel(self.param, "Positive")
            self.submodels[
                "negative oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.LeadingOrderDiffusionLimited(
                self.param, "Negative"
            )
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.LeadingOrder(
                self.param, self.reactions
            )
        else:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.NoOxygen(
                self.param
            )

    @property
    def default_spatial_methods(self):
        # ODEs only in the macroscale, so use base spatial method
        return {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteVolume,
        }

    @property
    def default_geometry(self):
        if self.options["bc_options"]["dimensionality"] == 0:
            return pybamm.Geometry("1D macro")
        elif self.options["bc_options"]["dimensionality"] == 1:
            return pybamm.Geometry("1+1D macro")

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """

        if self.options["surface form"] == "algebraic":
            return pybamm.ScikitsDaeSolver()
        else:
            return pybamm.ScipySolver()
