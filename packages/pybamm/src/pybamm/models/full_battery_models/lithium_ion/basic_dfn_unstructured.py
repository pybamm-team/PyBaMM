#
# Basic Doyle-Fuller-Newman (DFN) Model — 2D/3D Unstructured FVM
#
from __future__ import annotations

import pybamm
from pybamm.models.full_battery_models.lithium_ion.base_lithium_ion_model import (
    BaseModel,
)


class BasicDFNUnstructured(BaseModel):
    """Doyle-Fuller-Newman (DFN) model discretised with
    :class:`~pybamm.FiniteVolumeUnstructured`.

    The ``"dimensionality"`` option is the number of directions resolved besides
    the through-cell *x*: 1 (default) meshes (x, z) with quads and 2 meshes
    (x, y, z) with hexahedra. Pass ``submesh_types`` to
    :class:`pybamm.Simulation` for triangles or tetrahedra.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model. See
        :class:`pybamm.BatteryModelOptions`.
    name : str, optional
        The name of the model.
    """

    def __init__(self, options=None, name="Doyle-Fuller-Newman model (unstructured)"):
        options = {"dimensionality": 1, **(options or {})}
        super().__init__(options, name)
        if self.options["dimensionality"] not in (1, 2):
            raise pybamm.OptionError(
                "BasicDFNUnstructured needs a 'dimensionality' of 1 (x-z mesh) "
                "or 2 (x-y-z mesh)"
            )
        three_dimensional = self.options["dimensionality"] == 2
        pybamm.citations.register("Marquis2019")

        Q = pybamm.Variable("Discharge capacity [A.h]")

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        axes = ["x", "y", "z"] if three_dimensional else ["x", "z"]
        coords_n = [
            pybamm.SpatialVariable(
                f"{axis}_n", "negative electrode", coord_sys="cartesian"
            )
            for axis in axes
        ]
        coords_s = [
            pybamm.SpatialVariable(f"{axis}_s", "separator", coord_sys="cartesian")
            for axis in axes
        ]
        coords_p = [
            pybamm.SpatialVariable(
                f"{axis}_p", "positive electrode", coord_sys="cartesian"
            )
            for axis in axes
        ]
        coords = [
            pybamm.SpatialVariable(axis, whole_cell, coord_sys="cartesian")
            for axis in axes
        ]
        axis_input_names = {
            "x": "Through-cell distance (x) [m]",
            "y": "Horizontal distance (y) [m]",
            "z": "Vertical distance (z) [m]",
        }
        input_names = [axis_input_names[axis] for axis in axes]
        inputs_n = dict(zip(input_names, coords_n, strict=True))
        inputs_s = dict(zip(input_names, coords_s, strict=True))
        inputs_p = dict(zip(input_names, coords_p, strict=True))

        # A 2D slice stands for a cell of width L_y, so volume integrals are
        # scaled by the width the mesh does not resolve
        width = 1 if three_dimensional else self.param.L_y

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

        eps_n = pybamm.FunctionParameter("Negative electrode porosity", inputs_n)
        eps_s = pybamm.FunctionParameter("Separator porosity", inputs_s)
        eps_p = pybamm.FunctionParameter("Positive electrode porosity", inputs_p)
        eps = pybamm.concatenation(eps_n, eps_s, eps_p)

        eps_s_n = pybamm.FunctionParameter(
            "Negative electrode active material volume fraction",
            inputs_n,
        )
        eps_s_p = pybamm.FunctionParameter(
            "Positive electrode active material volume fraction",
            inputs_p,
        )

        tor = pybamm.concatenation(
            eps_n**self.param.n.b_e, eps_s**self.param.s.b_e, eps_p**self.param.p.b_e
        )
        a_n = 3 * eps_s_n / self.param.n.prim.R_typ
        a_p = 3 * eps_s_p / self.param.p.prim.R_typ

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
        solid_lithium_negative = width * pybamm.Integral(c_s_n_av * eps_s_n, coords_n)
        solid_lithium_positive = width * pybamm.Integral(c_s_p_av * eps_s_p, coords_p)
        total_solid_lithium = solid_lithium_negative + solid_lithium_positive

        ######################
        # Current in the solid
        ######################
        # Multiply by the squared length of each meshed direction to improve
        # conditioning
        L_scale = self.param.L_x**2 * self.param.L_z**2
        sides = ["left", "right", "top", "bottom"]
        if three_dimensional:
            L_scale *= self.param.L_y**2
            sides += ["front", "back"]
        zero_flux = {side: (pybamm.Scalar(0), "Neumann") for side in sides}
        sigma_eff_n = self.param.n.sigma(sto_surf_n, T) * eps_s_n**self.param.n.b_s
        sigma_eff_p = self.param.p.sigma(sto_surf_p, T) * eps_s_p**self.param.p.b_s
        self.algebraic[phi_s_n] = L_scale * (
            pybamm.div(-sigma_eff_n * pybamm.grad(phi_s_n)) + a_j_n
        )
        self.algebraic[phi_s_p] = L_scale * (
            pybamm.div(-sigma_eff_p * pybamm.grad(phi_s_p)) + a_j_p
        )
        self.boundary_conditions[phi_s_n] = {
            **zero_flux,
            "left": (pybamm.Scalar(0), "Dirichlet"),
        }
        self.boundary_conditions[phi_s_p] = {
            **zero_flux,
            "right": (i_cell / pybamm.boundary_value(-sigma_eff_p, "right"), "Neumann"),
        }
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = self.param.ocv_init

        ######################
        # Current in the electrolyte
        ######################
        kappa_eff = self.param.kappa_e(c_e, T) * tor
        kappa_D_eff = kappa_eff * self.param.chiRT_over_Fc(c_e, T)
        i_e = kappa_D_eff * pybamm.grad(c_e) - kappa_eff * pybamm.grad(phi_e)
        # The unstructured TPFA operator only accepts div(D * grad(u)) products,
        # so div(i_e) is written out term by term
        self.algebraic[phi_e] = L_scale * (
            pybamm.div(kappa_D_eff * pybamm.grad(c_e))
            - pybamm.div(kappa_eff * pybamm.grad(phi_e))
            - a_j
        )
        self.boundary_conditions[phi_e] = dict(zero_flux)
        self.initial_conditions[phi_e] = -self.param.n.prim.U_init

        ######################
        # Electrolyte concentration
        ######################
        D_e_eff = tor * self.param.D_e(c_e, T)
        t_plus = self.param.t_plus(c_e, T)
        N_e = -D_e_eff * pybamm.grad(c_e) + t_plus * i_e / self.param.F
        # The migration term t_plus * i_e / F is kept inside the flux so that the
        # balance is conservative when t_plus depends on c_e (see #5745)
        self.rhs[c_e] = (1 / eps) * (
            pybamm.div(D_e_eff * pybamm.grad(c_e))
            - pybamm.div((t_plus * kappa_D_eff / self.param.F) * pybamm.grad(c_e))
            + pybamm.div((t_plus * kappa_eff / self.param.F) * pybamm.grad(phi_e))
            + a_j / self.param.F
        )
        self.boundary_conditions[c_e] = dict(zero_flux)
        self.initial_conditions[c_e] = self.param.c_e_init

        ######################
        # (Some) variables
        ######################
        voltage = pybamm.boundary_value(phi_s_p, "top-right")
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        total_lithium = width * pybamm.Integral(c_e * eps, coords)
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
            "Sum of volumetric interfacial current densities [A.m-3]": a_j,
            "Electrolyte current density [A.m-2]": i_e,
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
            "Positive electrode interfacial current density [A.m-2]": j_p,
            "Negative electrode interfacial current density [A.m-2]": j_n,
            "Electrolyte flux [mol.m-2.s-1]": N_e,
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
        transverse = {"z": {"min": 0, "max": self.param.L_z}}
        if self.options["dimensionality"] == 2:
            transverse = {"y": {"min": 0, "max": self.param.L_y}, **transverse}
        return {
            "negative electrode": {
                "x_n": {"min": 0, "max": self.param.n.L},
                **transverse,
            },
            "separator": {
                "x_s": {"min": self.param.n.L, "max": self.param.n.L + self.param.s.L},
                **transverse,
            },
            "positive electrode": {
                "x_p": {
                    "min": self.param.n.L + self.param.s.L,
                    "max": self.param.n.L + self.param.s.L + self.param.p.L,
                },
                **transverse,
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
        if self.options["dimensionality"] == 1:
            element_type = "quad"
        else:
            element_type = "hexahedron"
        return {
            "negative electrode": pybamm.UnstructuredMeshGenerator(
                element_type=element_type
            ),
            "separator": pybamm.UnstructuredMeshGenerator(element_type=element_type),
            "positive electrode": pybamm.UnstructuredMeshGenerator(
                element_type=element_type
            ),
            "positive particle": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.SubMesh0D,
        }

    @property
    def default_var_pts(self):
        if self.options["dimensionality"] == 1:
            return {"x_n": 20, "x_s": 30, "x_p": 20, "r_n": 20, "r_p": 20, "z": 10}
        return {
            "x_n": 10,
            "x_s": 10,
            "x_p": 10,
            "r_n": 20,
            "r_p": 20,
            "y": 5,
            "z": 5,
        }
