#
# Single Particle Model (SPM)
#
import pybamm


class ToyModel(pybamm.LithiumIonBaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Toy Model"

        param = self.set_of_parameters

        c_n = pybamm.Variable("Negative particle concentration", domain = "negative particle")
        i_boundary_cc = pybamm.Variable("Current collector current density", domain = "current collector")

        # particle
        N = -pybamm.grad(c_n)
        self.rhs = {c_n: -pybamm.div(N)}
        self.algebraic = {}
        self.initial_conditions = {c_n: param.c_n_init}
        rbc = - i_boundary_cc / param.l_n
        self.boundary_conditions = {
            c_n: {"left": (0, "Neumann"), "right": (rbc, "Neumann")}
        }
        self.variables = {"Negative particle concentration": c_n}

        # current collector
        current_collector_model = pybamm.current_collector.OhmTwoDimensional(param)
        current_collector_model.set_uniform_current(i_boundary_cc)
        self.update(current_collector_model)

    @property
    def default_geometry(self):
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            return pybamm.Geometry("1D macro", "1D micro")
        elif dimensionality == 1:
            return pybamm.Geometry("1+1D macro", "(1+0)+1D micro")
        elif dimensionality == 2:
            return pybamm.Geometry("2+1D macro", "(2+0)+1D micro")

    @property
    def default_submesh_types(self):
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality in [0, 1]:
            return {
                "negative electrode": pybamm.Uniform1DSubMesh,
                "separator": pybamm.Uniform1DSubMesh,
                "positive electrode": pybamm.Uniform1DSubMesh,
                "negative particle": pybamm.Uniform1DSubMesh,
                "positive particle": pybamm.Uniform1DSubMesh,
                "current collector": pybamm.Uniform1DSubMesh,
            }
        elif dimensionality == 2:
            return {
                "negative electrode": pybamm.Uniform1DSubMesh,
                "separator": pybamm.Uniform1DSubMesh,
                "positive electrode": pybamm.Uniform1DSubMesh,
                "negative particle": pybamm.Uniform1DSubMesh,
                "positive particle": pybamm.Uniform1DSubMesh,
                "current collector": pybamm.FenicsMesh2D,
            }

    @property
    def default_spatial_methods(self):
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality in [0, 1]:
            return {
                "macroscale": pybamm.FiniteVolume,
                "negative particle": pybamm.FiniteVolume,
                "positive particle": pybamm.FiniteVolume,
                "current collector": pybamm.FiniteVolume,
            }
        elif dimensionality == 2:
            return {
                "macroscale": pybamm.FiniteVolume,
                "negative particle": pybamm.FiniteVolume,
                "positive particle": pybamm.FiniteVolume,
                "current collector": pybamm.FiniteElementFenics,
            }

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality in [1, 2]:
            return pybamm.ScikitsDaeSolver()
        else:
            return pybamm.ScikitsOdeSolver()
