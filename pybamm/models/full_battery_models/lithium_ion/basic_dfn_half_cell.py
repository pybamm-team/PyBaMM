#
# Basic Doyle-Fuller-Newman (DFN) Half Cell Model
#
import pybamm
from .base_lithium_ion_model import BaseModel
from pybamm.geometry import half_cell_spatial_vars
from pybamm.geometry.half_cell_geometry import half_cell_geometry


class BasicDFNHalfCell(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery with lithium counter
    electrode, adapted from [2]_.

    This class differs from the :class:`pybamm.lithium_ion.BasicDFN` model class in
    that it is for a cell with a lithium counter electrode (half cell). This is a
    feature under development (for example, it cannot be used with the Simulation class
    for the moment) and in the future it will be incorporated as a standard model with
    the full functionality.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    options : dict
        A dictionary of options to be passed to the model. For the half cell it should
        include which is the working electrode.

    References
    ----------
    .. [2] M Doyle, TF Fuller and JS Nwman. “Modeling of Galvanostatic Charge and
        Discharge of the Lithium/Polymer/Insertion Cell”. Journal of The
        Electrochemical Society, 140(6):1526-1533, 1993

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, name="Doyle-Fuller-Newman half cell model", options=None):
        super().__init__({}, name)
        pybamm.citations.register("Marquis2019")
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param
        options = options or {"working electrode": None}

        if options["working electrode"] not in ["negative", "positive"]:
            raise ValueError(
                "The option 'working electrode' should be either 'positive'"
                " or 'negative'"
            )

        self.options.update(options)
        working_electrode = options["working electrode"]

        if working_electrode == "negative":
            R_w_typ = param.R_n_typ
        else:
            R_w_typ = param.R_p_typ

        # Set default length scales
        self.length_scales = {
            "working electrode": param.L_x,
            "separator": param.L_x,
            "working particle": R_w_typ,
            "current collector y": param.L_z,
            "current collector z": param.L_z,
        }

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")

        # Define some useful scalings
        pot = param.potential_scale
        i_typ = param.current_scale

        # Variables that vary spatially are created with a domain.
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        c_e_w = pybamm.Variable(
            "Working electrolyte concentration", domain="working electrode"
        )
        c_e = pybamm.Concatenation(c_e_s, c_e_w)
        c_s_w = pybamm.Variable(
            "Working particle concentration",
            domain="working particle",
            auxiliary_domains={"secondary": "working electrode"},
        )
        phi_s_w = pybamm.Variable(
            "Working electrode potential", domain="working electrode"
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", domain="separator")
        phi_e_w = pybamm.Variable(
            "Working electrolyte potential", domain="working electrode"
        )
        phi_e = pybamm.Concatenation(phi_e_s, phi_e_w)

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time

        # Define particle surface concentration
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_w = pybamm.surf(c_s_w)

        # Define parameters. We need to assemble them differently depending on the
        # working electrode

        if working_electrode == "negative":
            # Porosity and Tortuosity
            # Primary broadcasts are used to broadcast scalar quantities across a domain
            # into a vector of the right shape, for multiplying with other vectors
            eps_s = pybamm.PrimaryBroadcast(
                pybamm.Parameter("Separator porosity"), "separator"
            )
            eps_w = pybamm.PrimaryBroadcast(
                pybamm.Parameter("Negative electrode porosity"), "working electrode"
            )
            b_e_s = param.b_e_s
            b_e_w = param.b_e_n

            # Interfacial reactions
            j0_w = param.j0_n(c_e_w, c_s_surf_w, T) / param.C_r_n
            U_w = param.U_n
            ne_w = param.ne_n

            # Particle diffusion parameters
            D_w = param.D_n
            C_w = param.C_n
            a_R_w = param.a_R_n
            gamma_w = pybamm.Scalar(1)
            c_w_init = param.c_n_init

            # Electrode equation parameters
            eps_s_w = pybamm.Parameter(
                "Negative electrode active material volume fraction"
            )
            b_s_w = param.b_s_n
            sigma_w = param.sigma_n

            # Other parameters (for outputs)
            c_w_max = param.c_n_max
            U_ref = param.U_n_ref
            phi_s_w_ref = pybamm.Scalar(0)
            L_w = param.L_n

        else:
            # Porosity and Tortuosity
            eps_s = pybamm.PrimaryBroadcast(
                pybamm.Parameter("Separator porosity"), "separator"
            )
            eps_w = pybamm.PrimaryBroadcast(
                pybamm.Parameter("Positive electrode porosity"), "working electrode"
            )
            b_e_s = param.b_e_s
            b_e_w = param.b_e_p

            # Interfacial reactions
            j0_w = param.gamma_p * param.j0_p(c_e_w, c_s_surf_w, T) / param.C_r_p
            U_w = param.U_p
            ne_w = param.ne_p

            # Particle diffusion parameters
            D_w = param.D_p
            C_w = param.C_p
            a_R_w = param.a_R_p
            gamma_w = param.gamma_p
            c_w_init = param.c_p_init

            # Electrode equation parameters
            eps_s_w = pybamm.Parameter(
                "Positive electrode active material volume fraction"
            )
            b_s_w = param.b_s_p
            sigma_w = param.sigma_p

            # Other parameters (for outputs)
            c_w_max = param.c_p_max
            U_ref = param.U_p_ref
            phi_s_w_ref = param.U_p_ref - param.U_n_ref
            L_w = param.L_p

        eps = pybamm.Concatenation(eps_s, eps_w)
        tor = pybamm.Concatenation(eps_s ** b_e_s, eps_w ** b_e_w)

        j_w = (
            2 * j0_w * pybamm.sinh(ne_w / 2 * (phi_s_w - phi_e_w - U_w(c_s_surf_w, T)))
        )
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j = pybamm.Concatenation(j_s, j_w)

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
        # Particles
        ######################
        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_w = -D_w(c_s_w, T) * pybamm.grad(c_s_w)
        self.rhs[c_s_w] = -(1 / C_w) * pybamm.div(N_s_w)

        # Boundary conditions must be provided for equations with spatial
        # derivatives
        self.boundary_conditions[c_s_w] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -C_w * j_w / a_R_w / gamma_w / D_w(c_s_surf_w, T),
                "Neumann",
            ),
        }

        # c_w_init can in general be a function of x
        # Note the broadcasting, for domains
        x_w = pybamm.PrimaryBroadcast(half_cell_spatial_vars.x_w, "working particle")
        self.initial_conditions[c_s_w] = c_w_init(x_w)

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event(
                "Minimum working particle surface concentration",
                pybamm.min(c_s_surf_w) - 0.01,
            ),
            pybamm.Event(
                "Maximum working particle surface concentration",
                (1 - 0.01) - pybamm.max(c_s_surf_w),
            ),
        ]

        ######################
        # Current in the solid
        ######################
        sigma_eff_w = sigma_w * eps_s_w ** b_s_w
        i_s_w = -sigma_eff_w * pybamm.grad(phi_s_w)
        self.boundary_conditions[phi_s_w] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                i_cell / pybamm.boundary_value(-sigma_eff_w, "right"),
                "Neumann",
            ),
        }
        self.algebraic[phi_s_w] = pybamm.div(i_s_w) + j_w
        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent
        # initial conditions
        self.initial_conditions[phi_s_w] = U_w(c_w_init(1), param.T_init)

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e
            + (1 - param.t_plus(c_e, T)) * j / param.gamma_e
        )
        dce_dx = (
            -(1 - param.t_plus(c_e, T))
            * i_cell
            * param.C_e
            / (tor * param.gamma_e * param.D_e(c_e, T))
        )

        self.boundary_conditions[c_e] = {
            "left": (pybamm.boundary_value(dce_dx, "left"), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }

        self.initial_conditions[c_e] = param.c_e_init
        self.events.append(
            pybamm.Event(
                "Zero electrolyte concentration cut-off", pybamm.min(c_e) - 0.002
            )
        )

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e, T) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j

        ref_potential = param.U_n_ref / pot

        self.boundary_conditions[phi_e] = {
            "left": (ref_potential, "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }

        self.initial_conditions[phi_e] = ref_potential

        ######################
        # (Some) variables
        ######################
        L_Li = pybamm.Parameter("Lithium counter electrode thickness [m]")
        sigma_Li = pybamm.Parameter("Lithium counter electrode conductivity [S.m-1]")
        j_Li = pybamm.Parameter(
            "Lithium counter electrode exchange-current density [A.m-2]"
        )

        vdrop_cell = pybamm.boundary_value(phi_s_w, "right") - ref_potential
        vdrop_Li = -(
            2 * pybamm.arcsinh(i_cell * i_typ / j_Li)
            + L_Li * i_typ * i_cell / (sigma_Li * pot)
        )
        voltage = vdrop_cell + vdrop_Li

        c_e_total = pybamm.x_average(eps * c_e)
        c_s_surf_w_av = pybamm.x_average(c_s_surf_w)

        c_s_rav = pybamm.r_average(c_s_w)
        c_s_vol_av = pybamm.x_average(eps_s_w * c_s_rav)

        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Time [s]": param.timescale * pybamm.t,
            "Working particle surface concentration": c_s_surf_w,
            "X-averaged working particle surface concentration": c_s_surf_w_av,
            "Working particle concentration": c_s_w,
            "Working particle surface concentration [mol.m-3]": c_w_max * c_s_surf_w,
            "X-averaged working particle surface concentration "
            "[mol.m-3]": c_w_max * c_s_surf_w_av,
            "Working particle concentration [mol.m-3]": c_w_max * c_s_w,
            "Total lithium in working electrode": c_s_vol_av,
            "Total lithium in working electrode [mol]": c_s_vol_av
            * c_w_max
            * L_w
            * param.A_cc,
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": param.c_e_typ * c_e,
            "Total electrolyte concentration": c_e_total,
            "Total electrolyte concentration [mol]": c_e_total
            * param.c_e_typ
            * L_w
            * param.L_s
            * param.A_cc,
            "Current [A]": I,
            "Working electrode potential": phi_s_w,
            "Working electrode potential [V]": phi_s_w_ref + pot * phi_s_w,
            "Working electrode open circuit potential": U_w(c_s_surf_w, T),
            "Working electrode open circuit potential [V]": U_ref
            + pot * U_w(c_s_surf_w, T),
            "Electrolyte potential": phi_e,
            "Electrolyte potential [V]": -param.U_n_ref + pot * phi_e,
            "Voltage drop in the cell": vdrop_cell,
            "Voltage drop in the cell [V]": phi_s_w_ref
            + param.U_n_ref
            + pot * vdrop_cell,
            "Terminal voltage": voltage,
            "Terminal voltage [V]": phi_s_w_ref + param.U_n_ref + pot * voltage,
        }

    @property
    def default_geometry(self):
        return half_cell_geometry(
            current_collector_dimension=self.options["dimensionality"],
            working_electrode=self.options["working electrode"],
        )

    @property
    def default_var_pts(self):
        var = pybamm.geometry.half_cell_spatial_vars
        base_var_pts = {
            var.x_Li: 20,
            var.x_s: 20,
            var.x_w: 20,
            var.r_w: 30,
            var.y: 10,
            var.z: 10,
        }
        # Reduce the default points for 2D current collectors
        if self.options["dimensionality"] == 2:
            base_var_pts.update({var.x_Li: 10, var.x_s: 10, var.x_w: 10})
        return base_var_pts

    @property
    def default_submesh_types(self):
        base_submeshes = {
            "lithium counter electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "working electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "working particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
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
            "lithium counter electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
            "working electrode": pybamm.FiniteVolume(),
            "working particle": pybamm.FiniteVolume(),
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

    def new_copy(self, build=False):
        new_model = self.__class__(name=self.name, options=self.options)
        new_model.use_jacobian = self.use_jacobian
        new_model.convert_to_format = self.convert_to_format
        new_model.timescale = self.timescale
        new_model.length_scales = self.length_scales
        return new_model
