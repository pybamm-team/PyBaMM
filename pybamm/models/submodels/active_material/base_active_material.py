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
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)

    def _get_standard_active_material_variables(self, eps_solid):
        param = self.param
        phase_name = self.phase_name
        domain, Domain = self.domain_Domain

        if eps_solid.domain == []:
            eps_solid = pybamm.PrimaryBroadcast(eps_solid, "current collector")
        if eps_solid.domain == ["current collector"]:
            eps_solid = pybamm.PrimaryBroadcast(eps_solid, f"{domain} electrode")
        eps_solid_av = pybamm.x_average(eps_solid)

        variables = {
            f"{Domain} electrode {phase_name}"
            "active material volume fraction": eps_solid,
            f"X-averaged {domain} electrode {phase_name}"
            "active material volume fraction": eps_solid_av,
        }

        # Update other microstructure variables
        # some models (e.g. lead-acid) do not have particles
        if self.options["particle shape"] == "no particles":
            a = self.phase_param.a
            variables.update(
                {
                    f"{Domain} electrode surface area to volume ratio [m-1]": a,
                    f"X-averaged {domain} electrode surface area "
                    "to volume ratio [m-1]": pybamm.x_average(a),
                }
            )
            return variables

        else:
            # Update electrode capacity variables
            L = self.domain_param.L
            c_s_max = self.phase_param.c_max

            C = (
                pybamm.yz_average(eps_solid_av)
                * L
                * param.A_cc
                * c_s_max
                * param.F
                / 3600
            )
            if phase_name == "":
                variables.update({f"{Domain} electrode capacity [A.h]": C})
            else:
                variables.update(
                    {f"{Domain} electrode {phase_name}phase capacity [A.h]": C}
                )

            # If a single particle size at every x, use the parameters
            # R_n, R_p. For a size distribution, calculate the area-weighted
            # mean using the distribution instead. Then the surface area is
            # calculated the same way
            domain_options = getattr(self.options, domain)
            if domain_options["particle size"] == "single":
                R = self.phase_param.R
            elif domain_options["particle size"] == "distribution":
                if self.domain == "negative":
                    R_ = pybamm.standard_spatial_vars.R_n
                elif self.domain == "positive":
                    R_ = pybamm.standard_spatial_vars.R_p
                R = pybamm.size_average(R_)

            R_av = pybamm.x_average(R)

            # Compute dimensional particle shape
            if self.options["particle shape"] == "spherical":
                a = 3 * eps_solid / R
                a_av = pybamm.x_average(a)

            variables.update(
                {
                    f"{Domain} {phase_name}particle radius": R / self.phase_param.R_typ,
                    f"{Domain} {phase_name}particle radius [m]": R,
                    f"X-averaged {domain} {phase_name}particle radius [m]": R_av,
                    f"{Domain} electrode {phase_name}"
                    "surface area to volume ratio [m-1]": a,
                    f"X-averaged {domain} electrode {phase_name}"
                    "surface area to volume ratio [m-1]": a_av,
                }
            )

            return variables

    def _get_standard_active_material_change_variables(self, deps_solid_dt):
        domain, Domain = self.domain_Domain

        if deps_solid_dt.domain == ["current collector"]:
            deps_solid_dt_av = deps_solid_dt
            deps_solid_dt = pybamm.PrimaryBroadcast(
                deps_solid_dt_av, f"{domain} electrode"
            )
        else:
            deps_solid_dt_av = pybamm.x_average(deps_solid_dt)

        variables = {
            f"{Domain} electrode {self.phase_name}"
            "active material volume fraction change [s-1]": deps_solid_dt,
            f"X-averaged {domain} electrode {self.phase_name}"
            "active material volume fraction change [s-1]": deps_solid_dt_av,
        }

        return variables
