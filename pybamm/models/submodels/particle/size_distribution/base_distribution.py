#
# Base class for particles
#
import pybamm

from ..base_particle import BaseParticle


class BaseSizeDistribution(BaseParticle):
    """
    Base class for molar conservation in a distribution of particle sizes.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.BaseParticle`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_standard_concentration_distribution_variables(self, c_s):
        """
        Forms standard concentration variables that depend on particle size R given
        one concentration distribution variable c_s.
        """
        if self.domain == "Negative":
            c_scale = self.param.c_n_max
        elif self.domain == "Positive":
            c_scale = self.param.c_p_max

        # Note: Currently not possible to broadcast from (r, R) to (r, R, x) since
        # domain x for broadcast is in "tertiary" position.

        # Broadcast and x-average when necessary
        if c_s.domain == [
            self.domain.lower() + " particle size"
        ] and c_s.auxiliary_domains["secondary"] != [
            self.domain.lower() + " electrode"
        ]:
            c_s_xav_distribution = pybamm.PrimaryBroadcast(
                c_s, [self.domain.lower() + " particle"]
            )
            # Placeholder broadcast to x, filled with zeros only
            c_s_distribution = pybamm.FullBroadcast(
                0,
                [self.domain.lower() + " particle"],
                {
                    "secondary": self.domain.lower() + " particle size",
                    "tertiary": self.domain.lower() + " electrode",
                },
            )

            # Surface concentration distribution variables
            c_s_surf_xav_distribution = c_s
            c_s_surf_distribution = pybamm.SecondaryBroadcast(
                c_s_surf_xav_distribution, [self.domain.lower() + " electrode"]
            )
        elif c_s.domain == [self.domain.lower() + " particle"] and (
            c_s.auxiliary_domains["tertiary"] != [self.domain.lower() + " electrode"]
        ):
            c_s_xav_distribution = c_s
            # Placeholder broadcast to x, filled with zeros only
            c_s_distribution = pybamm.FullBroadcast(
                0,
                [self.domain.lower() + " particle"],
                {
                    "secondary": self.domain.lower() + " particle size",
                    "tertiary": self.domain.lower() + " electrode",
                },
            )

            # Surface concentration distribution variables
            c_s_surf_xav_distribution = pybamm.surf(c_s_xav_distribution)
            c_s_surf_distribution = pybamm.SecondaryBroadcast(
                c_s_surf_xav_distribution, [self.domain.lower() + " electrode"]
            )
        elif c_s.domain == [
            self.domain.lower() + " particle size"
        ] and c_s.auxiliary_domains["secondary"] == [
            self.domain.lower() + " electrode"
        ]:
            c_s_surf_distribution = c_s
            c_s_surf_xav_distribution = pybamm.x_average(c_s)

            c_s_xav_distribution = pybamm.PrimaryBroadcast(
                c_s_surf_xav_distribution, [self.domain.lower() + " particle"]
            )
            # Placeholder broadcast to x, filled with zeros only
            c_s_distribution = pybamm.FullBroadcast(
                0,
                [self.domain.lower() + " particle"],
                {
                    "secondary": self.domain.lower() + " particle size",
                    "tertiary": self.domain.lower() + " electrode",
                },
            )
        else:
            c_s_distribution = c_s
            # TODO: x-average the *tertiary* domain. Leave unaltered for now.
            c_s_xav_distribution = c_s

            # Surface concentration distribution variables
            c_s_surf_distribution = pybamm.surf(c_s)
            c_s_surf_xav_distribution = pybamm.x_average(c_s_surf_distribution)

        variables = {
            self.domain
            + " particle concentration distribution": c_s_distribution,
            self.domain
            + " particle concentration distribution "
            + "[mol.m-3]": c_scale * c_s_distribution,
            "X-averaged "
            + self.domain.lower()
            + " particle concentration distribution": c_s_xav_distribution,
            "X-averaged "
            + self.domain.lower()
            + " particle concentration distribution "
            + "[mol.m-3]": c_scale * c_s_xav_distribution,
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration"
            + " distribution": c_s_surf_xav_distribution,
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration distribution "
            + "[mol.m-3]": c_scale * c_s_surf_xav_distribution,
            self.domain
            + " particle surface concentration"
            + " distribution": c_s_surf_distribution,
            self.domain
            + " particle surface concentration"
            + " distribution [mol.m-3]": c_scale * c_s_surf_distribution,
        }
        return variables
