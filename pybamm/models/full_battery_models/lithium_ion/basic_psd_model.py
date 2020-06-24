#
# Basic Many Particle Model (MPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicPSDModel(BaseModel):
    """Particle-Size Distribution (PSD) model of a lithium-ion battery, from [1]_.

    This class is similar to the :class:`pybamm.lithium_ion.SPM` model class in that it
    shows the whole model in a single class.

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

    def __init__(self, name="Particle-Size Distribution Model"):
        super().__init__({}, name)
        ######################
        # Parameters
        ######################
        # Import all the standard parameters from base_lithium_ion_model.BaseModel
        # (in turn from pybamm.standard_parameters_lithium_ion)
        param = self.param

        # Additional parameters for this model
        # Dimensionless standard deviations
        sd_a_n = pybamm.Parameter("negative area-weighted particle-size standard deviation")
        sd_a_p = pybamm.Parameter("positive area-weighted particle-size standard deviation")

        # Particle-size distributions (area-weighted)
        def f_a_dist_n(R,R_av_a,sd_a):
            inputs = {
            "negative particle-size variable": R,
            "negative area-weighted mean particle size": R_av_a,
            "negative area-weighted particle-size standard deviation": sd_a,
            }
            return pybamm.FunctionParameter(
            "negative area-weighted particle-size distribution",
            inputs,
            )
        def f_a_dist_p(R,R_av_a,sd_a):
            inputs = {
            "positive particle-size variable": R,
            "positive area-weighted mean particle size": R_av_a,
            "positive area-weighted particle-size standard deviation": sd_a,
            }
            return pybamm.FunctionParameter(
            "positive area-weighted particle-size distribution",
            inputs,
            )

        # Set length scales for additional domains (particle-size domains)
        self.length_scales.update(
            {
                "negative particle-size domain": param.R_n,
                "positive particle-size domain": param.R_p,
            }
        )

        ######################
        # Variables
        ######################
        # Discharge capacity
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # X-averaged particle concentrations: these now depend continuously on particle size, and so
        # have secondary domains "negative/positive particle-size domain"
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative particle-size domain",
                #"tertiary": "negative electrode",
            }
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive particle-size domain",
                #"tertiary": "positive electrode"
            }
        )
        # Electrode potentials (leave them without a domain for now)
        phi_e = pybamm.Variable(
            "Electrolyte potential"
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential"
#            domain="positive particle-size domain",
#            auxiliary_domains={
#                "secondary": "positive electrode",
#            }
        )

        # Spatial Variables
        R_variable_n = pybamm.SpatialVariable(
            "negative particle-size variable",
            domain=["negative particle-size domain"], #could add auxiliary domains
            coord_sys="cartesian"
            )
        R_variable_p = pybamm.SpatialVariable(
            "positive particle-size variable",
            domain=["positive particle-size domain"], #could add auxiliary domains
            coord_sys="cartesian"
            )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time

        ######################
        # State of Charge
        ######################
        I_dim = param.dimensional_current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I_dim * param.timescale / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Interfacial reactions
        ######################

        c_s_surf_n = pybamm.surf(c_s_n)
        phi_s_n = 0


        j0_n = param.j0_n(1, c_s_surf_n, T) / param.C_r_n # setting c_e = 1
        j_n = (
            2
            * j0_n
            * pybamm.sinh(
                param.ne_n / 2 * (phi_s_n - phi_e - param.U_n(c_s_surf_n, T))
            )
        )
        c_s_surf_p = pybamm.surf(c_s_p)
        j0_p = param.gamma_p * param.j0_p(1, c_s_surf_p, T) / param.C_r_p # setting c_e = 1

        j_p = (
            2
            * j0_p
            * pybamm.sinh(
                param.ne_p / 2 * (phi_s_p - phi_e - param.U_p(c_s_surf_p, T))
            )
        )

        # integral equation for phi_e
        self.algebraic[phi_e] = pybamm.Integral(
            f_a_dist_n(R_variable_n, 1, sd_a_n)*j_n,
            R_variable_n
        ) - i_cell/param.l_n

        # integral equation for phi_s_p
        self.algebraic[phi_s_p] = pybamm.Integral(
            f_a_dist_p(R_variable_p, 1, sd_a_p)*j_p,
            R_variable_p
        ) + i_cell/param.l_p

        self.initial_conditions[phi_e] = pybamm.Scalar(1)#pybamm.PrimaryBroadcast(1, "negative particle-size domain")
        self.initial_conditions[phi_s_p] = pybamm.Scalar(1)#pybamm.PrimaryBroadcast(1, "positive particle-size domain")

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

        # Time and space output variables
        self.set_standard_output_variables()

        # Dimensionless output variables (not already defined)
        V = phi_s_p
        c_e = 1

        c_s_n_size_av = pybamm.Integral(
            f_a_dist_n(R_variable_n, 1, sd_a_n)*c_s_n,
            R_variable_n
        )
        c_s_p_size_av = pybamm.Integral(
            f_a_dist_p(R_variable_p, 1, sd_a_p)*c_s_p,
            R_variable_p
        )
        c_s_surf_n_size_av = pybamm.Integral(
            f_a_dist_n(R_variable_n, 1, sd_a_n)*c_s_surf_n,
            R_variable_n
        )
        c_s_surf_p_size_av = pybamm.Integral(
            f_a_dist_p(R_variable_p, 1, sd_a_p)*c_s_surf_p,
            R_variable_p
        )
        # Dimensional output variables
        V_dim = param.potential_scale * V + (param.U_p_ref - param.U_n_ref)

        c_s_n_dim = c_s_n * param.c_n_max
        c_s_p_dim = c_s_p * param.c_p_max
        c_s_surf_n_dim = c_s_surf_n * param.c_n_max
        c_s_surf_p_dim = c_s_surf_p * param.c_p_max

        c_s_n_size_av_dim = c_s_n_size_av * param.c_n_max
        c_s_p_size_av_dim = c_s_p_size_av * param.c_p_max
        c_s_surf_n_size_av_dim = c_s_surf_n_size_av * param.c_n_max
        c_s_surf_p_size_av_dim = c_s_surf_p_size_av * param.c_p_max


        c_e_dim = c_e * param.c_e_typ
        phi_s_n_dim = phi_s_n * param.potential_scale
        phi_s_p_dim = phi_s_p * param.potential_scale + (param.U_p_ref - param.U_n_ref)
        phi_e_dim = phi_e * param.potential_scale  - param.U_n_ref




        whole_cell = ["negative electrode", "separator", "positive electrode"]

        self.variables.update({
            # New "Distribution" variables, those depending on R_variable_n, R_variable_p
            "Negative particle concentration distribution": c_s_n,
            "Negative particle concentration distribution [mol.m-3]": c_s_n_dim,
            "Negative particle surface concentration distribution": c_s_surf_n,
            "Negative particle surface concentration distribution [mol.m-3]": c_s_surf_n_dim,
            "Positive particle concentration distribution": c_s_p,
            "Positive particle concentration distribution [mol.m-3]": c_s_p_dim,
            "Positive particle surface concentration distribution": c_s_surf_p,
            "Positive particle surface concentration distribution [mol.m-3]": c_s_surf_p_dim,


            # Standard output quantities (no PSD)
            "Negative particle concentration": c_s_n_size_av,
            "Negative particle concentration [mol.m-3]": c_s_n_size_av_dim,
            "Negative particle surface concentration": c_s_surf_n_size_av,
            "Negative particle surface concentration [mol.m-3]": c_s_surf_n_size_av_dim,
            "Electrolyte concentration": pybamm.PrimaryBroadcast(c_e, whole_cell),
            "Electrolyte concentration [mol.m-3]": pybamm.PrimaryBroadcast(c_e_dim, whole_cell),
            "Positive particle concentration": c_s_p_size_av,
            "Positive particle concentration [mol.m-3]": c_s_p_size_av_dim,
            "Positive particle surface concentration": c_s_surf_p_size_av,
            "Positive particle surface concentration [mol.m-3]": c_s_surf_p_size_av_dim,
            "Negative electrode potential": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Negative electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_n_dim, "negative electrode"
            ),
            "Electrolyte potential": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Electrolyte potential [V]": pybamm.PrimaryBroadcast(phi_e_dim, whole_cell),
            "Positive electrode potential": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Positive electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_p_dim, "positive electrode"
            ),
            "Current": i_cell,
            "Current [A]": I_dim,
            "Terminal voltage": V,
            "Terminal voltage [V]": V_dim,
        })


        self.events += [
            pybamm.Event("Minimum voltage", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", V - param.voltage_high_cut),
        ]

    def set_standard_output_variables(self):
        # This overwrites the method in parent class, base_lithium_ion_model.BaseModel

        # Time
        self.variables.update(
            {
                "Time": pybamm.t,
                "Time [s]": pybamm.t * self.timescale,
                "Time [min]": pybamm.t * self.timescale / 60,
                "Time [h]": pybamm.t * self.timescale / 3600,
            }
        )

        # Spatial
        var = pybamm.standard_spatial_vars
        L_x = pybamm.geometric_parameters.L_x
        self.variables.update(
            {
                "x": var.x,
                "x [m]": var.x * L_x,
                "x_n": var.x_n,
                "x_n [m]": var.x_n * L_x,
                "x_s": var.x_s,
                "x_s [m]": var.x_s * L_x,
                "x_p": var.x_p,
                "x_p [m]": var.x_p * L_x,
            }
        )

        # New Spatial Variables
        R_variable_n = pybamm.SpatialVariable(
            "negative particle-size variable",
            domain=["negative particle-size domain"],
            coord_sys="cartesian"
            )
        R_variable_p = pybamm.SpatialVariable(
            "positive particle-size variable",
            domain=["positive particle-size domain"],
            coord_sys="cartesian"
            )
        R_n = pybamm.geometric_parameters.R_n
        R_p = pybamm.geometric_parameters.R_p

        self.variables.update(
            {
                "Negative particle size": R_variable_n,
                "Negative particle size [m]": R_variable_n * R_n,
                "Positive particle size": R_variable_p,
                "Positive particle size [m]": R_variable_p * R_p,
            }
        )

    ####################
    # Overwrite defaults
    ####################
    @property
    def default_parameter_values(self):
        # Default parameter values
        # Lion parameters left as default parameter set for tests
        default_params = super().default_parameter_values


        # append new parameter values

        # lognormal area-weighted particle-size distribution
        def lognormal_distribution(R,R_av,sd):
            import numpy as np
            # inputs are particle radius R, the mean R_av, and standard deviation sd
            # inputs can be dimensional or dimensionless
            mu_ln = pybamm.log(R_av**2/pybamm.sqrt(R_av**2+sd**2))
            sigma_ln = pybamm.sqrt(pybamm.log(1 + sd**2/R_av**2))
            return pybamm.exp(-(pybamm.log(R)-mu_ln)**2/(2*sigma_ln**2))/pybamm.sqrt(2*np.pi*sigma_ln**2)/R

        default_params.update(
            {"negative area-weighted particle-size standard deviation": 0.3},
            check_already_exists=False
        )
        default_params.update(
            {"positive area-weighted particle-size standard deviation": 0.3},
            check_already_exists=False
        )
        default_params.update(
            {"negative area-weighted particle-size distribution": lognormal_distribution},
            check_already_exists=False
        )
        default_params.update(
            {"positive area-weighted particle-size distribution": lognormal_distribution},
            check_already_exists=False
        )

        return default_params

    @property
    def default_geometry(self):
        default_geom = super().default_geometry

        # New Spatial Variables
        R_variable_n = pybamm.SpatialVariable(
            "negative particle-size variable",
            domain=["negative particle-size domain"],
            coord_sys="cartesian"
            )
        R_variable_p = pybamm.SpatialVariable(
            "positive particle-size variable",
            domain=["positive particle-size domain"],
            coord_sys="cartesian"
            )

        # append new domains
        default_geom.update(
            {
                "negative particle-size domain": {
                    R_variable_n: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.Scalar(5),
                    }
                },
                "positive particle-size domain": {
                    R_variable_p: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.Scalar(5),
                    }
                },
            }
        )
        return default_geom

    @property
    def default_var_pts(self):
        defaults = super().default_var_pts

        # New Spatial Variables
        R_variable_n = pybamm.SpatialVariable(
            "negative particle-size variable",
            domain=["negative particle-size domain"],
            coord_sys="cartesian"
        )
        R_variable_p = pybamm.SpatialVariable(
            "positive particle-size variable",
            domain=["positive particle-size domain"],
            coord_sys="cartesian"
        )
        # add to dictionary
        defaults.update(
            {
                R_variable_n: 50,
                R_variable_p: 50,
            }
        )
        return defaults

    @property
    def default_submesh_types(self):
        default_submeshes = super().default_submesh_types

        default_submeshes.update(
            {
                "negative particle-size domain": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
                "positive particle-size domain": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            }
        )
        return default_submeshes

    @property
    def default_spatial_methods(self):
        default_spatials = super().default_spatial_methods

        default_spatials.update(
            {
                "negative particle-size domain": pybamm.FiniteVolume(),
                "positive particle-size domain": pybamm.FiniteVolume(),
            }
        )
        return default_spatials
