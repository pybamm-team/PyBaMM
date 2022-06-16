#
# Base class for active material volume fraction
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for active material volume fraction

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options : dict
        Additional options to pass to the model

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, options):
        super().__init__(param, domain, options=options)

    def _get_standard_active_material_variables(self, eps_solid):
        param = self.param
        if eps_solid.domain == []:
            eps_solid = pybamm.PrimaryBroadcast(eps_solid, "current collector")
        if eps_solid.domain == ["current collector"]:
            eps_solid = pybamm.PrimaryBroadcast(
                eps_solid, self.domain.lower() + " electrode"
            )
        eps_solid_av = pybamm.x_average(eps_solid)

        variables = {
            self.domain + " electrode active material volume fraction": eps_solid,
            "X-averaged "
            + self.domain.lower()
            + " electrode active material volume fraction": eps_solid_av,
        }

        # Update other microstructure variables
        # some models (e.g. lead-acid) do not have particles
        if self.options["particle shape"] == "no particles":
            a = self.domain_param.a
            a_typ = self.domain_param.a_typ
            variables.update(
                {
                    self.domain + " electrode surface area to volume ratio": a,
                    self.domain
                    + " electrode surface area to volume ratio [m-1]": a * a_typ,
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode surface area to volume ratio": pybamm.x_average(a),
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode surface area"
                    + " to volume ratio [m-1]": pybamm.x_average(a) * a_typ,
                }
            )
            return variables

        else:
            # Update electrode capacity variables
            L = self.domain_param.L
            c_s_max = self.domain_param.c_max

            C = (
                pybamm.yz_average(eps_solid_av)
                * L
                * param.A_cc
                * c_s_max
                * param.F
                / 3600
            )
            variables.update({self.domain + " electrode capacity [A.h]": C})

            # If a single particle size at every x, use the parameters
            # R_n, R_p. For a size distribution, calculate the area-weighted
            # mean using the distribution instead. Then the surface area is
            # calculated the same way
            if self.options["particle size"] == "single":
                R = self.domain_param.R
                R_dim = self.domain_param.R_dimensional
            elif self.options["particle size"] == "distribution":
                if self.domain == "Negative":
                    R_ = pybamm.standard_spatial_vars.R_n
                elif self.domain == "Positive":
                    R_ = pybamm.standard_spatial_vars.R_p
                R = pybamm.size_average(R_)
                R_dim = R * self.domain_param.R_typ
            a_typ = self.domain_param.a_typ

            R_dim_av = pybamm.x_average(R_dim)

            # Compute dimensional particle shape
            if self.options["particle shape"] == "spherical":
                a_dim = 3 * eps_solid / R_dim
                a_dim_av = 3 * eps_solid_av / R_dim_av

            # Surface area to volume ratio is scaled with a_typ, so that it is equal to
            # 1 when eps_solid and R are uniform in space and time
            a = a_dim / a_typ
            a_av = a_dim_av / a_typ
            variables.update(
                {
                    self.domain + " particle radius": R,
                    self.domain + " particle radius [m]": R_dim,
                    self.domain + " electrode surface area to volume ratio": a,
                    self.domain
                    + " electrode surface area to volume ratio [m-1]": a_dim,
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode surface area to volume ratio": a_av,
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode surface area to volume ratio [m-1]": a_dim_av,
                }
            )

            return variables

    def _get_standard_active_material_change_variables(self, deps_solid_dt):

        if deps_solid_dt.domain == ["current collector"]:
            deps_solid_dt_av = deps_solid_dt
            deps_solid_dt = pybamm.PrimaryBroadcast(
                deps_solid_dt_av, self.domain.lower() + " electrode"
            )
        else:
            deps_solid_dt_av = pybamm.x_average(deps_solid_dt)

        variables = {
            self.domain
            + " electrode active material volume fraction change": deps_solid_dt,
            "X-averaged "
            + self.domain.lower()
            + " electrode active material volume fraction change": deps_solid_dt_av,
        }

        return variables
