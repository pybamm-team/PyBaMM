#
# Base class for particles
#
import pybamm


class BaseParticle(pybamm.BaseSubModel):
    """Base class for molar conservation in particles.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_standard_concentration_variables(self, c_s, c_s_xav):

        c_s_surf = pybamm.surf(c_s)

        c_s_surf_av = pybamm.x_average(c_s_surf)
        geo_param = pybamm.geometric_parameters

        if self.domain == "Negative":
            c_scale = self.param.c_n_max
            active_volume = geo_param.a_n_dim * geo_param.R_n / 3
        elif self.domain == "Positive":
            c_scale = self.param.c_p_max
            active_volume = geo_param.a_p_dim * geo_param.R_p / 3
        c_s_av = pybamm.r_average(c_s_xav)
        c_s_av_vol = active_volume * c_s_av
        variables = {
            self.domain + " particle concentration": c_s,
            self.domain + " particle concentration [mol.m-3]": c_s * c_scale,
            "X-averaged " + self.domain.lower() + " particle concentration": c_s_xav,
            "X-averaged "
            + self.domain.lower()
            + " particle concentration [mol.m-3]": c_s_xav * c_scale,
            self.domain + " particle surface concentration": c_s_surf,
            self.domain
            + " particle surface concentration [mol.m-3]": c_scale * c_s_surf,
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration": c_s_surf_av,
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration [mol.m-3]": c_scale * c_s_surf_av,
            self.domain + " electrode active volume fraction": active_volume,
            self.domain + " electrode volume-averaged concentration": c_s_av_vol,
            self.domain
            + " electrode "
            + "volume-averaged concentration [mol.m-3]": c_s_av_vol * c_scale,
            self.domain + " electrode average extent of lithiation": c_s_av,
        }

        return variables

    def _get_standard_flux_variables(self, N_s, N_s_xav):
        variables = {
            self.domain + " particle flux": N_s,
            "X-averaged " + self.domain.lower() + " particle flux": N_s_xav,
        }

        return variables

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
            self.domain.lower() + " particle-size domain"
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
                    "secondary": self.domain.lower() + " particle-size domain",
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
                    "secondary": self.domain.lower() + " particle-size domain",
                    "tertiary": self.domain.lower() + " electrode",
                },
            )

            # Surface concentration distribution variables
            c_s_surf_xav_distribution = pybamm.surf(c_s_xav_distribution)
            c_s_surf_distribution = pybamm.SecondaryBroadcast(
                c_s_surf_xav_distribution, [self.domain.lower() + " electrode"]
            )
        elif c_s.domain == [
            self.domain.lower() + " particle-size domain"
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
                    "secondary": self.domain.lower() + " particle-size domain",
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

    def set_events(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        tol = 1e-4

        self.events.append(
            pybamm.Event(
                "Minumum " + self.domain.lower() + " particle surface concentration",
                pybamm.min(c_s_surf) - tol,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                "Maximum " + self.domain.lower() + " particle surface concentration",
                (1 - tol) - pybamm.max(c_s_surf),
                pybamm.EventType.TERMINATION,
            )
        )
