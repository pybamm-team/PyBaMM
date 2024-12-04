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

    def build(self):
        soc = pybamm.CoupledVariable("SoC")
        self.coupled_variables.update({soc.name: soc})
        z = pybamm.PrimaryBroadcast(soc, "ECMD particle")
        x = pybamm.SpatialVariable(
            "x ECMD", domain=["ECMD particle"], coord_sys="Cartesian"
        )
        z_surf = pybamm.surf(z)
        eta_diffusion = pybamm.Scalar(0)

        self.variables.update(
            {
                "Distributed SoC": z,
                "x ECMD": x,
                "Diffusion overpotential [V]": eta_diffusion,
                "Surface SoC": z_surf,
            }
        )


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

    def build(self):
        z = pybamm.Variable("Distributed SoC", domain="ECMD particle")
        x = pybamm.SpatialVariable(
            "x ECMD", domain=["ECMD particle"], coord_sys="Cartesian"
        )
        variables = {
            "Distributed SoC": z,
            "x ECMD": x,
        }

        soc = pybamm.CoupledVariable("SoC")
        self.coupled_variables.update({soc.name: soc})
        z_surf = pybamm.surf(z)
        eta_diffusion = -(self.param.ocv(z_surf) - self.param.ocv(soc))

        variables.update(
            {
                "Diffusion overpotential [V]": eta_diffusion,
                "Surface SoC": z_surf,
            }
        )
        cell_capacity = self.param.cell_capacity
        current = pybamm.CoupledVariable("Current [A]")
        self.coupled_variables.update({current.name: current})

        # governing equations
        dzdt = pybamm.div(pybamm.grad(z)) / self.param.tau_D
        self.rhs = {z: dzdt}

        # boundary conditions
        lbc = pybamm.Scalar(0)
        rbc = -self.param.tau_D * current / (cell_capacity * 3600)
        self.boundary_conditions = {
            z: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}
        }
        self.initial_conditions = {z: self.param.initial_soc}
        self.events += [
            pybamm.Event("Minimum surface SoC", z_surf),
            pybamm.Event("Maximum surface SoC", 1 - z_surf),
        ]
        variables.update(variables)
        self.variables.update(variables)
