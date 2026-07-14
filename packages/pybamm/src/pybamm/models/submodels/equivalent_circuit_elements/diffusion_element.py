import pybamm


class NoDiffusion(pybamm.BaseSubModel):
    """
    Without Diffusion element for
    equivalent circuits.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param)
        self.model_options = options

    def get_coupled_variables(self, variables):
        z = pybamm.PrimaryBroadcast(variables["SoC"], "ECMD particle")
        x = pybamm.SpatialVariable(
            "x ECMD", domain=["ECMD particle"], coord_sys="Cartesian"
        )
        z_surf = pybamm.surf(z)
        eta_diffusion = pybamm.Scalar(0)

        variables.update(
            {
                "Distributed SoC": z,
                "x ECMD": x,
                "Diffusion overpotential [V]": eta_diffusion,
                "Surface SoC": z_surf,
            }
        )

        return variables


class DiffusionElement(pybamm.BaseSubModel):
    """
    With Diffusion element for
    equivalent circuits.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param)
        pybamm.citations.register("Fan2022")
        self.model_options = options

    def get_fundamental_variables(self):
        z = pybamm.Variable("Distributed SoC", domain="ECMD particle")
        x = pybamm.SpatialVariable(
            "x ECMD", domain=["ECMD particle"], coord_sys="Cartesian"
        )
        variables = {
            "Distributed SoC": z,
            "x ECMD": x,
        }
        return variables

    def get_coupled_variables(self, variables):
        z = variables["Distributed SoC"]
        soc = variables["SoC"]
        z_surf = pybamm.surf(z)
        eta_diffusion = -(self.param.ocv(z_surf) - self.param.ocv(soc))

        variables.update(
            {
                "Diffusion overpotential [V]": eta_diffusion,
                "Surface SoC": z_surf,
            }
        )

        return variables

    def set_rhs(self, variables):
        cell_capacity = self.param.cell_capacity
        current = variables["Current [A]"]
        z = variables["Distributed SoC"]

        # governing equations
        dzdt = pybamm.div(pybamm.grad(z)) / self.param.tau_D
        self.rhs = {z: dzdt}

        # boundary conditions
        lbc = pybamm.Scalar(0)
        rbc = -self.param.tau_D * current / (cell_capacity * 3600)
        self.boundary_conditions = {
            z: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        z = variables["Distributed SoC"]
        self.initial_conditions = {z: self.param.initial_soc}

    def add_events_from(self, variables):
        z_surf = variables["Surface SoC"]
        self.events += [
            pybamm.Event("Minimum surface SoC", z_surf),
            pybamm.Event("Maximum surface SoC", 1 - z_surf),
        ]
