import pybamm


def _ecmd_spatial_variable():
    return pybamm.SpatialVariable(
        "x ECMD", domain=["ECMD particle"], coord_sys="Cartesian"
    )


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
        z = pybamm.PrimaryBroadcast(self.coupled_variable("SoC"), "ECMD particle")
        self.variables.update(
            {
                "Distributed SoC": z,
                "x ECMD": _ecmd_spatial_variable(),
                "Diffusion overpotential [V]": pybamm.Scalar(0),
                "Surface SoC": pybamm.surf(z),
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
        soc = self.coupled_variable("SoC")
        current = self.coupled_variable("Current [A]")

        z = pybamm.Variable("Distributed SoC", domain="ECMD particle")
        z_surf = pybamm.surf(z)
        eta_diffusion = self.param.ocv(z_surf) - self.param.ocv(soc)
        self.variables.update(
            {
                "Distributed SoC": z,
                "x ECMD": _ecmd_spatial_variable(),
                "Diffusion overpotential [V]": eta_diffusion,
                "Surface SoC": z_surf,
            }
        )

        self.rhs[z] = pybamm.div(pybamm.grad(z)) / self.param.tau_D
        rbc = -self.param.tau_D * current / (self.param.cell_capacity * 3600)
        self.boundary_conditions[z] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (rbc, "Neumann"),
        }
        self.initial_conditions[z] = self.param.initial_soc
        self.events += [
            pybamm.Event("Minimum surface SoC", z_surf),
            pybamm.Event("Maximum surface SoC", 1 - z_surf),
        ]
