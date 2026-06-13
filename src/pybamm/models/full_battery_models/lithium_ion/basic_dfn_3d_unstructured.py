#
# Basic Doyle-Fuller-Newman (DFN) Model — 3D Unstructured FVM
#
import pybamm
from pybamm.models.full_battery_models.lithium_ion.base_lithium_ion_model import (
    BaseModel,
)


class BasicDFN3DUnstructured(BaseModel):
    """Doyle-Fuller-Newman (DFN) model on a 3D unstructured mesh.

    Extends :class:`BasicDFN2DUnstructured` to three spatial dimensions
    (x, y, z) using tetrahedral elements.  The through-cell direction is
    *x*, the width direction is *y*, and the height direction is *z*.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(
        self,
        name="Doyle-Fuller-Newman model (3D unstructured)",
    ):
        super().__init__(name=name)
        pybamm.citations.register("Marquis2019")

        ######################
        # Variables
        ######################
        Q = pybamm.Variable("Discharge capacity [A.h]")

        x = pybamm.SpatialVariable(
            "x",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
            direction="lr",
        )
        x_n = pybamm.SpatialVariable(
            "x_n", domain="negative electrode", coord_sys="cartesian", direction="lr"
        )
        x_s = pybamm.SpatialVariable(
            "x_s", domain="separator", coord_sys="cartesian", direction="lr"
        )
        x_p = pybamm.SpatialVariable(
            "x_p", domain="positive electrode", coord_sys="cartesian", direction="lr"
        )
        y_n = pybamm.SpatialVariable(
            "y_n", domain="negative electrode", coord_sys="cartesian", direction="fb"
        )
        y_s = pybamm.SpatialVariable(
            "y_s", domain="separator", coord_sys="cartesian", direction="fb"
        )
        y_p = pybamm.SpatialVariable(
            "y_p", domain="positive electrode", coord_sys="cartesian", direction="fb"
        )
        y = pybamm.SpatialVariable(
            "y",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
            direction="fb",
        )
        z_n = pybamm.SpatialVariable(
            "z_n", domain="negative electrode", coord_sys="cartesian", direction="tb"
        )
        z_s = pybamm.SpatialVariable(
            "z_s", domain="separator", coord_sys="cartesian", direction="tb"
        )
        z_p = pybamm.SpatialVariable(
            "z_p", domain="positive electrode", coord_sys="cartesian", direction="tb"
        )
        z = pybamm.SpatialVariable(
            "z",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
            direction="tb",
        )

        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration [mol.m-3]",
            domain="negative electrode",
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration [mol.m-3]",
            domain="separator",
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration [mol.m-3]",
            domain="positive electrode",
        )
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential [V]",
            domain="negative electrode",
        )
        phi_e_s = pybamm.Variable(
            "Separator electrolyte potential [V]",
            domain="separator",
        )
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential [V]",
            domain="positive electrode",
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)

        phi_s_n = pybamm.Variable(
            "Negative electrode potential [V]", domain="negative electrode"
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential [V]",
            domain="positive electrode",
        )
        c_s_n = pybamm.Variable(
            "Negative particle concentration [mol.m-3]",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration [mol.m-3]",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )

        T = self.param.T_init

        ######################
        # Other set-up
        ######################
        i_cell = self.param.current_density_with_time

        eps_n = pybamm.FunctionParameter(
            "Negative electrode porosity",
            {"Through-cell distance (x) [m]": x_n, "Vertical distance (z) [m]": z_n},
        )
        eps_s = pybamm.FunctionParameter(
            "Separator porosity",
            {"Through-cell distance (x) [m]": x_s, "Vertical distance (z) [m]": z_s},
        )
        eps_p = pybamm.FunctionParameter(
            "Positive electrode porosity",
            {"Through-cell distance (x) [m]": x_p, "Vertical distance (z) [m]": z_p},
        )
        eps = pybamm.concatenation(eps_n, eps_s, eps_p)

        eps_s_n = pybamm.FunctionParameter(
            "Negative electrode active material volume fraction",
            {"Through-cell distance (x) [m]": x_n, "Vertical distance (z) [m]": z_n},
        )
        eps_s_p = pybamm.FunctionParameter(
            "Positive electrode active material volume fraction",
            {"Through-cell distance (x) [m]": x_p, "Vertical distance (z) [m]": z_p},
        )

        tor = pybamm.concatenation(
            eps_n**self.param.n.b_e, eps_s**self.param.s.b_e, eps_p**self.param.p.b_e
        )
        a_n = 3 * self.param.n.prim.epsilon_s_av / self.param.n.prim.R_typ
        a_p = 3 * self.param.p.prim.epsilon_s_av / self.param.p.prim.R_typ

        # Interfacial reactions
        c_s_surf_n = pybamm.surf(c_s_n)
        sto_surf_n = c_s_surf_n / self.param.n.prim.c_max
        j0_n = self.param.n.prim.j0(c_e_n, c_s_surf_n, T)
        delta_phi_n = phi_s_n - phi_e_n
        eta_n = delta_phi_n - self.param.n.prim.U(sto_surf_n, T)
        Feta_RT_n = self.param.F * eta_n / (self.param.R * T)
        j_n = 2 * j0_n * pybamm.sinh(self.param.n.prim.ne / 2 * Feta_RT_n)

        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_p = c_s_surf_p / self.param.p.prim.c_max
        j0_p = self.param.p.prim.j0(c_e_p, c_s_surf_p, T)
        delta_phi_p = phi_s_p - phi_e_p
        eta_p = delta_phi_p - self.param.p.prim.U(sto_surf_p, T)
        Feta_RT_p = self.param.F * eta_p / (self.param.R * T)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = 2 * j0_p * pybamm.sinh(self.param.p.prim.ne / 2 * Feta_RT_p)

        a_j_n = a_n * j_n
        a_j_p = a_p * j_p
        a_j = pybamm.concatenation(a_j_n, j_s, a_j_p)

        ######################
        # State of Charge
        ######################
        current = self.param.current_with_time
        self.rhs[Q] = current / 3600
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################
        N_s_n = -self.param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -self.param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_n / (self.param.F * pybamm.surf(self.param.n.prim.D(c_s_n, T))),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_p / (self.param.F * pybamm.surf(self.param.p.prim.D(c_s_p, T))),
                "Neumann",
            ),
        }
        self.initial_conditions[c_s_n] = self.param.n.prim.c_init
        self.initial_conditions[c_s_p] = self.param.p.prim.c_init

        c_s_n_av = pybamm.RAverage(c_s_n)
        c_s_p_av = pybamm.RAverage(c_s_p)
        solid_lithium_negative = pybamm.Integral(c_s_n_av * eps_s_n, [x_n, y_n, z_n])
        solid_lithium_positive = pybamm.Integral(c_s_p_av * eps_s_p, [x_p, y_p, z_p])
        total_solid_lithium = solid_lithium_negative + solid_lithium_positive

        ######################
        # Current in the solid
        ######################
        sigma_eff_n = self.param.n.sigma(T) * eps_s_n**self.param.n.b_s
        sigma_eff_p = self.param.p.sigma(T) * eps_s_p**self.param.p.b_s
        L_scale = self.param.L_x**2 * self.param.L_z**2
        self.algebraic[phi_s_n] = L_scale * (
            pybamm.div(-sigma_eff_n * pybamm.grad(phi_s_n)) + a_j_n
        )
        self.algebraic[phi_s_p] = L_scale * (
            pybamm.div(-sigma_eff_p * pybamm.grad(phi_s_p)) + a_j_p
        )
        self.boundary_conditions[phi_s_n] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
            "top": (pybamm.Scalar(0), "Neumann"),
            "bottom": (pybamm.Scalar(0), "Neumann"),
            "front": (pybamm.Scalar(0), "Neumann"),
            "back": (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                i_cell / pybamm.boundary_value(-sigma_eff_p, "right"),
                "Neumann",
            ),
            "top": (pybamm.Scalar(0), "Neumann"),
            "bottom": (pybamm.Scalar(0), "Neumann"),
            "front": (pybamm.Scalar(0), "Neumann"),
            "back": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = self.param.ocv_init
        ######################
        # Current in the electrolyte
        ######################
        kappa_eff = self.param.kappa_e(c_e, T) * tor
        kappa_D_eff = kappa_eff * self.param.chiRT_over_Fc(c_e, T)
        self.algebraic[phi_e] = L_scale * (
            pybamm.div(kappa_D_eff * pybamm.grad(c_e))
            - pybamm.div(kappa_eff * pybamm.grad(phi_e))
            - a_j
        )
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
            "top": (pybamm.Scalar(0), "Neumann"),
            "bottom": (pybamm.Scalar(0), "Neumann"),
            "front": (pybamm.Scalar(0), "Neumann"),
            "back": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -self.param.n.prim.U_init

        ######################
        # Electrolyte concentration
        ######################
        D_e_eff = tor * self.param.D_e(c_e, T)
        self.rhs[c_e] = (1 / eps) * (
            pybamm.div(D_e_eff * pybamm.grad(c_e))
            + (1 - self.param.t_plus(c_e, T)) * a_j / self.param.F
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
            "top": (pybamm.Scalar(0), "Neumann"),
            "bottom": (pybamm.Scalar(0), "Neumann"),
            "front": (pybamm.Scalar(0), "Neumann"),
            "back": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = self.param.c_e_init

        ######################
        # (Some) variables
        ######################
        voltage = pybamm.boundary_value(phi_s_p, "top-right")
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        total_lithium = pybamm.Integral(c_e * eps, [x, y, z])
        self.variables = {
            "Negative particle concentration [mol.m-3]": c_s_n,
            "Total lithium [mol]": total_lithium,
            "Negative particle surface concentration [mol.m-3]": c_s_surf_n,
            "Electrolyte concentration [mol.m-3]": c_e,
            "Negative electrolyte concentration [mol.m-3]": c_e_n,
            "Separator electrolyte concentration [mol.m-3]": c_e_s,
            "Positive electrolyte concentration [mol.m-3]": c_e_p,
            "Positive particle concentration [mol.m-3]": c_s_p,
            "Positive particle surface concentration [mol.m-3]": c_s_surf_p,
            "Current [A]": current,
            "Current variable [A]": current,
            "Negative electrode potential [V]": phi_s_n,
            "Electrolyte potential [V]": phi_e,
            "Negative electrolyte potential [V]": phi_e_n,
            "Separator electrolyte potential [V]": phi_e_s,
            "Positive electrolyte potential [V]": phi_e_p,
            "Positive electrode potential [V]": phi_s_p,
            "Voltage [V]": voltage,
            "Battery voltage [V]": voltage * num_cells,
            "Time [s]": pybamm.t,
            "Discharge capacity [A.h]": Q,
            "x": x,
            "y": y,
            "z": z,
            "Current density [A.m-2]": a_j,
            "Electrolyte current density [A.m-2]": a_j,
            "x_n": x_n,
            "x_s": x_s,
            "x_p": x_p,
            "y_n": y_n,
            "y_s": y_s,
            "y_p": y_p,
            "z_n": z_n,
            "z_s": z_s,
            "z_p": z_p,
            "Negative electrode surface concentration [mol.m-3]": c_s_surf_n,
            "Negative electrode surface stoichiometry": sto_surf_n,
            "Positive electrode surface concentration [mol.m-3]": c_s_surf_p,
            "Positive electrode surface stoichiometry": sto_surf_p,
            "Positive electrode surface potential difference [V]": delta_phi_p,
            "Negative electrode surface potential difference [V]": delta_phi_n,
            "Positive electrode overpotential [V]": eta_p,
            "Negative electrode overpotential [V]": eta_n,
            "Positive electrode ocp [V]": self.param.p.prim.U(sto_surf_p, T),
            "Negative electrode ocp [V]": self.param.n.prim.U(sto_surf_n, T),
            "Positive electrode current density [A.m-2]": j_p,
            "Negative electrode current density [A.m-2]": j_n,
            "Electrolyte flux [mol.m-2.s-1]": D_e_eff,
            "Positive solid lithium [mol]": solid_lithium_positive,
            "Negative solid lithium [mol]": solid_lithium_negative,
            "Total solid lithium [mol]": total_solid_lithium,
        }
        self.events += [
            pybamm.Event("Minimum voltage [V]", voltage - self.param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", self.param.voltage_high_cut - voltage),
        ]

    @property
    def default_geometry(self):
        y_3d = pybamm.SpatialVariable(
            "y_3d",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
            direction="fb",
        )
        z_3d = pybamm.SpatialVariable(
            "z_3d",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
            direction="tb",
        )
        return {
            "negative electrode": {
                "x_n": {"min": 0, "max": self.param.n.L},
                y_3d: {"min": 0, "max": self.param.L_y},
                z_3d: {"min": 0, "max": self.param.L_z},
            },
            "separator": {
                "x_s": {
                    "min": self.param.n.L,
                    "max": self.param.n.L + self.param.s.L,
                },
                y_3d: {"min": 0, "max": self.param.L_y},
                z_3d: {"min": 0, "max": self.param.L_z},
            },
            "positive electrode": {
                "x_p": {
                    "min": self.param.n.L + self.param.s.L,
                    "max": self.param.n.L + self.param.s.L + self.param.p.L,
                },
                y_3d: {"min": 0, "max": self.param.L_y},
                z_3d: {"min": 0, "max": self.param.L_z},
            },
            "positive particle": {
                "r_p": {"min": 0, "max": self.param.p.prim.R_typ},
            },
            "negative particle": {
                "r_n": {"min": 0, "max": self.param.n.prim.R_typ},
            },
            "current collector": {
                "z": {"position": 0},
            },
        }

    @property
    def default_spatial_methods(self):
        return {
            "negative electrode": pybamm.FiniteVolumeUnstructured(),
            "separator": pybamm.FiniteVolumeUnstructured(),
            "positive electrode": pybamm.FiniteVolumeUnstructured(),
            "positive particle": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }

    @property
    def default_submesh_types(self):
        return {
            "negative electrode": pybamm.UnstructuredMeshGenerator(),
            "separator": pybamm.UnstructuredMeshGenerator(),
            "positive electrode": pybamm.UnstructuredMeshGenerator(),
            "positive particle": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

    @property
    def default_var_pts(self):
        y_3d = pybamm.SpatialVariable(
            "y_3d",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
            direction="fb",
        )
        z_3d = pybamm.SpatialVariable(
            "z_3d",
            domain=["negative electrode", "separator", "positive electrode"],
            coord_sys="cartesian",
            direction="tb",
        )
        return {
            "x_n": 5,
            "x_s": 5,
            "x_p": 5,
            "r_p": 10,
            "r_n": 10,
            y_3d: 3,
            z_3d: 3,
        }
