#
# Basic Many Particle Model (MPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicMPM(BaseModel):
    """Many Particle Model (MPM) model of a lithium-ion battery, from [1]_.

    This class differs from the :class:`pybamm.lithium_ion.SPM` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    combining different physical effects, and in general the main SPM class should be
    used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    References
    ----------
    .. [1] TL Kirk, J Evans, CP Please and SJ Chapman. “Modelling electrode heterogeneity
        in lithium-ion batteries: unimodal and bimodal particle-size distributions”.
        In: arXiv preprint arXiv:????? (2020).


    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, name="Many Particle Model"):
        super().__init__({}, name)
        ######################
        # Parameters
        ######################
        # Import all the standard parameters from base_lithium_ion_model.BaseModel
        # (in turn from pybamm.standard_parameters_lithium_ion)
        param = self.param

        # additional parameters
        sd_a_n = pybamm.Parameter("negative area-weighted particle size standard deviation")
        sd_a_p = pybamm.Parameter("positive area-weighted particle size standard deviation")

        def f_a_dist_n(R,R_av_a,sd_a):
            return pybamm.FunctionParameter(
            "negative area-weighted particle size distribution",
            {
            "negative particle size variable": R,
            "negative area-weighted mean particle size": R_av_a,
            "negative area-weighted particle size standard deviation": sd_a,
            }
            )
        def f_a_dist_p(R,R_av_a,sd_a):
            return pybamm.FunctionParameter(
            "positive area-weighted particle size distribution",
            {
            "positive particle size variable": R,
            "positive area-weighted mean particle size": R_av_a,
            "positive area-weighted particle size standard deviation": sd_a,
            }
            )


        ######################
        # Variables
        ######################
        # Discharge capacity
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # X-averaged particle concentrations
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative particle size domain",
                #"tertiary": "negative electrode",
            }
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive particle size domain",
                #"tertiary": "positive electrode"
            }
        )
        # Electrode potentials (leave them with a domain for now)
        phi_s_n = pybamm.Variable(
            "Negative electrode potential"
#            domain="negative particle size domain",
#            auxiliary_domains={
#                "secondary": "negative electrode",
#            }
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential"
#            domain="positive particle size domain",
#            auxiliary_domains={
#                "secondary": "positive electrode",
#            }
        )

        # Spatial Variables
        R_variable_n = pybamm.SpatialVariable(
            "negative particle size variable",
            domain=["negative particle size domain"],
            coord_sys="cartesian"
            )
        R_variable_p = pybamm.SpatialVariable(
            "positive particle size variable",
            domain=["positive particle size domain"],
            coord_sys="cartesian"
            )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time
#        j_n = i_cell / param.l_n
#        j_p = -i_cell / param.l_p

        ######################
        # State of Charge
        ######################
        I = param.dimensional_current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I * param.timescale / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Interfacial reactions
        ######################

        c_s_surf_n = pybamm.surf(c_s_n)
        phi_e = 0


        j0_n = param.j0_n(1, c_s_surf_n, T) / param.C_r_n
        j_n = (
            2
            * j0_n
            * pybamm.sinh(
                param.ne_n / 2 * (phi_s_n
                - phi_e - param.U_n(c_s_surf_n, T))
            )
        )
        c_s_surf_p = pybamm.surf(c_s_p)
        j0_p = param.gamma_p * param.j0_p(1, c_s_surf_p, T) / param.C_r_p

        j_p = (
            2
            * j0_p
            * pybamm.sinh(
                param.ne_p / 2 * (phi_s_p - phi_e - param.U_p(c_s_surf_p, T))
            )
        )

        self.algebraic[phi_s_n] = pybamm.Integral(
            f_a_dist_n(R_variable_n, 1, sd_a_n)*j_n,
            R_variable_n
        ) - i_cell/param.l_n

        self.algebraic[phi_s_p] = pybamm.Integral(
            f_a_dist_p(R_variable_p, 1, sd_a_p)*j_p,
            R_variable_p
        ) + i_cell/param.l_p

        self.initial_conditions[phi_s_n] = pybamm.Scalar(1)#pybamm.PrimaryBroadcast(1, "negative particle size domain")

        self.initial_conditions[phi_s_p] = pybamm.Scalar(1)#pybamm.PrimaryBroadcast(1, "positive particle size domain")

        ######################
        # Particles
        ######################

        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_n = -param.D_n(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.D_p(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -(1 / param.C_n) * pybamm.div(N_s_n) / R_variable_n**2
        self.rhs[c_s_p] = -(1 / param.C_p) * pybamm.div(N_s_p) / R_variable_p**2

        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -R_variable_n * param.C_n * j_n / param.a_n / param.D_n(c_s_surf_n, T),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -R_variable_p * param.C_p * j_p / param.a_p / param.gamma_p / param.D_p(c_s_surf_p, T),
                "Neumann",
            ),
        }
        # c_n_init and c_p_init are functions of x, but for the SPM we evaluate them at x=0
        # and x=1 since there is no x-dependence in the particles
        self.initial_conditions[c_s_n] = param.c_n_init(0)
        self.initial_conditions[c_s_p] = param.c_p_init(1)

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event(
                "Minimum negative particle surface concentration",
                pybamm.min(c_s_surf_n) - 0.01,
            ),
            pybamm.Event(
                "Maximum negative particle surface concentration",
                (1 - 0.01) - pybamm.max(c_s_surf_n),
            ),
            pybamm.Event(
                "Minimum positive particle surface concentration",
                pybamm.min(c_s_surf_p) - 0.01,
            ),
            pybamm.Event(
                "Maximum positive particle surface concentration",
                (1 - 0.01) - pybamm.max(c_s_surf_p),
            ),
        ]


        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors

        V = phi_s_p - phi_s_n
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        self.variables = {
            "Negative particle surface concentration": c_s_surf_n,
            "Electrolyte concentration": pybamm.PrimaryBroadcast(1, whole_cell),
            "Positive particle surface concentration": c_s_surf_p,
            "Current [A]": I,
            "Negative electrode potential": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Electrolyte potential": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Positive electrode potential": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Terminal voltage": V,
        }
        self.events += [
            pybamm.Event("Minimum voltage", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", V - param.voltage_high_cut),
        ]


    @property
    def default_parameter_values(self):
        # Default parameter values
        # Lion parameters left as default parameter set for tests
        import numpy as np

        def f_a_dist_Gaussian(R,R_av_a,sd_a):
            return pybamm.exp(-(R-R_av_a)**2/(2*sd_a**2))/pybamm.sqrt(2*np.pi)/sd_a

        default_params = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
        default_params.update({"negative area-weighted particle size standard deviation": 0.3}, check_already_exists=False)
        default_params.update({"positive area-weighted particle size standard deviation": 0.3}, check_already_exists=False)
        default_params.update({"negative area-weighted particle size distribution": f_a_dist_Gaussian}, check_already_exists=False)
        default_params.update({"positive area-weighted particle size distribution": f_a_dist_Gaussian}, check_already_exists=False)

        return default_params

    @property
    def default_geometry(self):
        default_geom = pybamm.battery_geometry()

        # New Spatial Variables
        R_variable_n = pybamm.SpatialVariable(
            "negative particle size variable",
            domain=["negative particle size domain"],
            coord_sys="cartesian"
            )
        R_variable_p = pybamm.SpatialVariable(
            "positive particle size variable",
            domain=["positive particle size domain"],
            coord_sys="cartesian"
            )

        # input new domains
        default_geom.update({"negative particle size domain":
            {
                R_variable_n: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.Scalar(5),
                }
            }
        }
        )
        default_geom.update({"positive particle size domain":
            {
                R_variable_p: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.Scalar(5),
                }
            }
        }
        )
        return default_geom

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars

        # New Spatial Variables
        R_variable_n = pybamm.SpatialVariable(
            "negative particle size variable",
            domain=["negative particle size domain"],
            coord_sys="cartesian"
            )
        R_variable_p = pybamm.SpatialVariable(
            "positive particle size variable",
            domain=["positive particle size domain"],
            coord_sys="cartesian"
            )

        return {
            var.x_n: 20,
            var.x_s: 20,
            var.x_p: 20,
            var.r_n: 10,
            var.r_p: 10,
            var.y: 10,
            var.z: 10,
            R_variable_n: 30,
            R_variable_p: 30,
        }

    @property
    def default_submesh_types(self):
        base_submeshes = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle size domain": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle size domain": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }
        if self.options["dimensionality"] == 0:
            base_submeshes["current collector"] = pybamm.MeshGenerator(pybamm.SubMesh0D)
        elif self.options["dimensionality"] == 1:
            base_submeshes["current collector"] = pybamm.MeshGenerator(
                pybamm.Uniform1DSubMesh
            )
        elif self.options["dimensionality"] == 2:
            base_submeshes["current collector"] = pybamm.MeshGenerator(
                pybamm.ScikitUniform2DSubMesh
            )
        return base_submeshes

    @property
    def default_spatial_methods(self):
        base_spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "negative particle size domain": pybamm.FiniteVolume(),
            "positive particle size domain": pybamm.FiniteVolume(),
        }
        if self.options["dimensionality"] == 0:
            # 0D submesh - use base spatial method
            base_spatial_methods[
                "current collector"
            ] = pybamm.ZeroDimensionalSpatialMethod()
        elif self.options["dimensionality"] == 1:
            base_spatial_methods["current collector"] = pybamm.FiniteVolume()
        elif self.options["dimensionality"] == 2:
            base_spatial_methods["current collector"] = pybamm.ScikitFiniteElement()
        return base_spatial_methods
