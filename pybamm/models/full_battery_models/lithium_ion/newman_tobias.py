#
# Newman Tobias Model
#
import pybamm
from .dfn import DFN


class NewmanTobias(DFN):
    """
    Newman-Tobias model of a lithium-ion battery based on the formulation in [1]_.
    This model assumes a uniform concentration profile in the electrolyte.
    Unlike the model posed in [1]_, this model accounts for nonlinear Butler-Volmer
    kinetics. It also tracks the average concentration in the solid phase in each
    electrode, which is equivalent to including an equation for the local state of
    charge as in [2]_. The user can pass the "particle" option to include mass
    transport in the particles.

    See :class:`pybamm.lithium_ion.BaseModel` for more details.


    References
    ----------
    .. [1] JS Newman and CW Tobias. "Theoretical Analysis of Current Distribution
           in Porous Electrodes". Journal of The Electrochemical Society,
           109(12):A1183-A1191, 1962
    .. [2] HN Chu, SU Kim, SK Rahimian, JB Siegel and CW Monroe. "Parameterization
           of prismatic lithium–iron–phosphate cells through a streamlined
           thermal/electrochemical model". Journal of Power Sources, 453, p.227787,
           2020


    **Extends:** :class:`pybamm.lithium_ion.DFN`
    """

    def __init__(self, options=None, name="Newman-Tobias model", build=True):
        # Set default option "uniform profile" for particle submodel. Other
        # default options are those given in `pybamm.BatteryModelOptions` defined in
        # `base_battery_model.py`.
        options = options or {}
        if "particle" not in options:
            options["particle"] = "uniform profile"

        super().__init__(options, name, build)

        pybamm.citations.register("Newman1962")
        pybamm.citations.register("Chu2020")

    def set_particle_submodel(self):
        for domain in ["negative", "positive"]:
            particle = getattr(self.options, domain)["particle"]
            for phase in self.options.phases[domain]:
                if particle == "Fickian diffusion":
                    submod = pybamm.particle.FickianDiffusion(
                        self.param, domain, self.options, phase=phase, x_average=True
                    )
                elif particle in [
                    "uniform profile",
                    "quadratic profile",
                    "quartic profile",
                ]:
                    submod = pybamm.particle.XAveragedPolynomialProfile(
                        self.param, domain, self.options, phase=phase
                    )
                self.submodels[f"{domain} {phase} particle"] = submod
                self.submodels[
                    f"{domain} {phase} total particle concentration"
                ] = pybamm.particle.TotalConcentration(
                    self.param, domain, self.options, phase
                )

    def set_electrolyte_concentration_submodel(self):
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.ConstantConcentration(self.param)
