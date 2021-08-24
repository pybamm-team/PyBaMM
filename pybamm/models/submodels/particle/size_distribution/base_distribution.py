#
# Base class for particle-size distributions
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

    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_distribution_variables(self, R):
        """
        Forms the particle-size distributions and mean radii given a spatial variable
        R. The domains of R will be different depending on the submodel, e.g. for the
        `SingleSizeDistribution` classes R does not have an "electrode" domain.
        """
        if self.domain == "Negative":
            R_typ = self.param.R_n_typ
            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_n(R)
        elif self.domain == "Positive":
            R_typ = self.param.R_p_typ
            # Particle-size distribution (area-weighted)
            f_a_dist = self.param.f_a_dist_p(R)

        # Ensure the distribution is normalised, irrespective of discretisation
        # or user input
        f_a_dist = f_a_dist / pybamm.Integral(f_a_dist, R)

        # Volume-weighted particle-size distribution
        f_v_dist = R * f_a_dist / pybamm.Integral(R * f_a_dist, R)

        # Number-based particle-size distribution
        f_num_dist = (f_a_dist / R ** 2) / pybamm.Integral(f_a_dist / R ** 2, R)

        # True mean radii and standard deviations, calculated from the f_a_dist that
        # was given
        R_num_mean = pybamm.Integral(R * f_num_dist, R)
        R_a_mean = pybamm.Integral(R * f_a_dist, R)
        R_v_mean = pybamm.Integral(R * f_v_dist, R)
        sd_num = pybamm.sqrt(pybamm.Integral((R - R_num_mean) ** 2 * f_num_dist, R))
        sd_a = pybamm.sqrt(pybamm.Integral((R - R_a_mean) ** 2 * f_a_dist, R))
        sd_v = pybamm.sqrt(pybamm.Integral((R - R_v_mean) ** 2 * f_v_dist, R))

        # X-average the means and standard deviations to give scalars
        # (to remove the "electrode" domain, if present)
        R_num_mean = pybamm.x_average(R_num_mean)
        R_a_mean = pybamm.x_average(R_a_mean)
        R_v_mean = pybamm.x_average(R_v_mean)
        sd_num = pybamm.x_average(sd_num)
        sd_a = pybamm.x_average(sd_a)
        sd_v = pybamm.x_average(sd_v)

        # X-averaged distributions, or broadcast
        if R.auxiliary_domains["secondary"] == [self.domain.lower() + " electrode"]:
            f_a_dist_xav = pybamm.x_average(f_a_dist)
            f_v_dist_xav = pybamm.x_average(f_v_dist)
            f_num_dist_xav = pybamm.x_average(f_num_dist)
        else:
            f_a_dist_xav = f_a_dist
            f_v_dist_xav = f_v_dist
            f_num_dist_xav = f_num_dist

            # broadcast
            f_a_dist = pybamm.SecondaryBroadcast(
                f_a_dist_xav, [self.domain.lower() + " electrode"]
            )
            f_v_dist = pybamm.SecondaryBroadcast(
                f_v_dist_xav, [self.domain.lower() + " electrode"]
            )
            f_num_dist = pybamm.SecondaryBroadcast(
                f_num_dist_xav, [self.domain.lower() + " electrode"]
            )

        variables = {
            self.domain + " particle sizes": R,
            self.domain + " particle sizes [m]": R * R_typ,
            self.domain + " area-weighted particle-size"
            + " distribution": f_a_dist,
            self.domain + " area-weighted particle-size"
            + " distribution [m-1]": f_a_dist / R_typ,
            self.domain + " volume-weighted particle-size"
            + " distribution": f_v_dist,
            self.domain + " volume-weighted particle-size"
            + " distribution [m-1]": f_v_dist / R_typ,
            self.domain + " number-based particle-size"
            + " distribution": f_num_dist,
            self.domain + " number-based particle-size"
            + " distribution [m-1]": f_num_dist / R_typ,
            self.domain + " area-weighted"
            + " mean particle radius": R_a_mean,
            self.domain + " area-weighted"
            + " mean particle radius [m]": R_a_mean * R_typ,
            self.domain + " volume-weighted"
            + " mean particle radius": R_v_mean,
            self.domain + " volume-weighted"
            + " mean particle radius [m]": R_v_mean * R_typ,
            self.domain + " number-based"
            + " mean particle radius": R_num_mean,
            self.domain + " number-based"
            + " mean particle radius [m]": R_num_mean * R_typ,
            self.domain + " area-weighted particle-size"
            + " standard deviation": sd_a,
            self.domain + " area-weighted particle-size"
            + " standard deviation [m]": sd_a * R_typ,
            self.domain + " volume-weighted particle-size"
            + " standard deviation": sd_v,
            self.domain + " volume-weighted particle-size"
            + " standard deviation [m]": sd_v * R_typ,
            self.domain + " number-based particle-size"
            + " standard deviation": sd_num,
            self.domain + " number-based particle-size"
            + " standard deviation [m]": sd_num * R_typ,
            # X-averaged distributions
            "X-averaged " + self.domain.lower() +
            " area-weighted particle-size distribution": f_a_dist_xav,
            "X-averaged " + self.domain.lower() +
            " area-weighted particle-size distribution [m-1]": f_a_dist_xav / R_typ,
            "X-averaged " + self.domain.lower() +
            " volume-weighted particle-size distribution": f_v_dist_xav,
            "X-averaged " + self.domain.lower() +
            " volume-weighted particle-size distribution [m-1]": f_v_dist_xav / R_typ,
            "X-averaged " + self.domain.lower() +
            " number-based particle-size distribution": f_num_dist_xav,
            "X-averaged " + self.domain.lower() +
            " number-based particle-size distribution [m-1]": f_num_dist_xav / R_typ,
        }

        return variables

    def _get_standard_concentration_distribution_variables(self, c_s):
        """
        Forms standard concentration variables that depend on particle size R given
        the fundamental concentration distribution variable c_s from the submodel.
        """
        if self.domain == "Negative":
            c_scale = self.param.c_n_max
        elif self.domain == "Positive":
            c_scale = self.param.c_p_max

        # Broadcast and x-average when necessary
        if c_s.domain == [
            self.domain.lower() + " particle size"
        ] and c_s.auxiliary_domains["secondary"] != [
            self.domain.lower() + " electrode"
        ]:
            # X-avg concentration distribution
            c_s_xav_distribution = pybamm.PrimaryBroadcast(
                c_s, [self.domain.lower() + " particle"]
            )

            # Surface concentration distribution variables
            c_s_surf_xav_distribution = c_s
            c_s_surf_distribution = pybamm.SecondaryBroadcast(
                c_s_surf_xav_distribution, [self.domain.lower() + " electrode"]
            )

            # Concentration distribution in all domains.
            c_s_distribution = pybamm.PrimaryBroadcast(
                c_s_surf_distribution, [self.domain.lower() + " particle"]
            )
        elif c_s.domain == [self.domain.lower() + " particle"] and (
            c_s.auxiliary_domains["tertiary"] != [self.domain.lower() + " electrode"]
        ):
            # X-avg concentration distribution
            c_s_xav_distribution = c_s

            # Surface concentration distribution variables
            c_s_surf_xav_distribution = pybamm.surf(c_s_xav_distribution)
            c_s_surf_distribution = pybamm.SecondaryBroadcast(
                c_s_surf_xav_distribution, [self.domain.lower() + " electrode"]
            )

            # Concentration distribution in all domains.
            c_s_distribution = pybamm.TertiaryBroadcast(
                c_s_xav_distribution, [self.domain.lower() + " electrode"]
            )
        elif c_s.domain == [
            self.domain.lower() + " particle size"
        ] and c_s.auxiliary_domains["secondary"] == [
            self.domain.lower() + " electrode"
        ]:
            # Surface concentration distribution variables
            c_s_surf_distribution = c_s
            c_s_surf_xav_distribution = pybamm.x_average(c_s)

            # X-avg concentration distribution
            c_s_xav_distribution = pybamm.PrimaryBroadcast(
                c_s_surf_xav_distribution, [self.domain.lower() + " particle"]
            )

            # Concentration distribution in all domains.
            c_s_distribution = pybamm.PrimaryBroadcast(
                c_s_surf_distribution, [self.domain.lower() + " particle"]
            )
        else:
            c_s_distribution = c_s

            # x-average the *tertiary* domain.
            # NOTE: not yet implemented. Make 0.5 everywhere
            c_s_xav_distribution = pybamm.FullBroadcast(
                0.5,
                [self.domain.lower() + " particle"],
                {
                    "secondary": self.domain.lower() + " particle size",
                    "tertiary": "current collector"
                },
            )

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

    def _get_standard_flux_distribution_variables(self, N_s):
        """
        Forms standard flux variables that depend on particle size R given
        the flux variable N_s from the distribution submodel.
        """

        if [self.domain.lower() + " electrode"] in N_s.auxiliary_domains.values():
            # N_s depends on x

            N_s_distribution = N_s
            # x-av the *tertiary* domain
            # NOTE: not yet implemented. Fill with zeros instead
            N_s_xav_distribution = pybamm.FullBroadcast(
                0,
                [self.domain.lower() + " particle"],
                {
                    "secondary": self.domain.lower() + " particle size",
                    "tertiary": "current collector"
                },
            )
        elif isinstance(N_s, pybamm.Scalar):
            # N_s is a constant (zero), as in "fast" submodels

            N_s_distribution = pybamm.FullBroadcastToEdges(
                0,
                [self.domain.lower() + " particle"],
                auxiliary_domains={
                    "secondary": self.domain.lower() + " particle size",
                    "tertiary": self.domain.lower() + " electrode",
                    "quaternary": "current collector",
                },
            )
            N_s_xav_distribution = pybamm.FullBroadcastToEdges(
                0,
                [self.domain.lower() + " particle"],
                auxiliary_domains={
                    "secondary": self.domain.lower() + " particle size",
                    "tertiary": "current collector",
                },
            )
        else:
            N_s_xav_distribution = N_s
            N_s_distribution = pybamm.TertiaryBroadcast(
                N_s, [self.domain.lower() + " electrode"]
            )

        variables = {
            "X-averaged "
            + self.domain.lower()
            + " particle flux distribution": N_s_xav_distribution,
            self.domain
            + " particle flux distribution": N_s_distribution,
        }

        return variables

