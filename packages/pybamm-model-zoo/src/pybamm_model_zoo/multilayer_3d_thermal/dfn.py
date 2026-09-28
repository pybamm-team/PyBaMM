"""The multilayer stack with each zone a Doyle-Fuller-Newman model."""

from __future__ import annotations

import pybamm
from pybamm_model_zoo.multilayer_3d_thermal.model import (
    MultiLayer3DThermalSPM,
    electrolyte_lithium,
    electrolyte_transport,
)


class MultiLayer3DThermalDFN(MultiLayer3DThermalSPM):
    """:class:`MultiLayer3DThermalSPM` with each zone a DFN.

    Each zone resolves particle concentrations ``c_s(r, x)``, the electrolyte
    concentration and potential, and the electrode potentials across its unit
    cell, with Butler-Volmer kinetics, as :class:`pybamm.lithium_ion.BasicDFN`
    does. Transport and kinetics see the zone's volume-averaged temperature.

    Parameters
    ----------
    num_physical_layers : int, optional
        Unit cells in the stack, at least 2.
    num_subdivisions : int, optional
        Zones the stack is resolved into: at least 2, and a divisor of
        ``num_physical_layers``. Defaults to one zone per unit cell.
    connection : str, optional
        How the zones are connected, ``"parallel"`` or ``"series"``.
    mesh_h : float, optional
        Target element size of each zone's mesh.
    options : dict, optional
        Model options. ``"cell geometry"`` defaults to, and must be, ``"pouch"``.
    name : str, optional
        The model name.
    """

    ELECTROCHEMISTRY_CITATION = "Doyle1993"

    def __init__(
        self,
        num_physical_layers: int = 3,
        num_subdivisions: int | None = None,
        connection: str = "parallel",
        mesh_h: float = 0.1,
        options: dict | None = None,
        name: str = "Multi-Layer 3D Thermal DFN",
    ) -> None:
        super().__init__(
            num_physical_layers=num_physical_layers,
            num_subdivisions=num_subdivisions,
            connection=connection,
            mesh_h=mesh_h,
            options=options,
            name=name,
        )

    def _build_electrochemistry_layer(self, layer_id: int) -> dict:
        """One zone's DFN, and the symbols the stack couples it through."""
        param = self.param
        prefix = f"Layer {layer_id}"
        regions = {
            "negative": "negative electrode",
            "separator": "separator",
            "positive": "positive electrode",
        }
        c_e_n, c_e_s, c_e_p = (
            pybamm.Variable(
                f"{prefix} {region} electrolyte concentration [mol.m-3]",
                domain=domain,
            )
            for region, domain in regions.items()
        )
        phi_e_n, phi_e_s, phi_e_p = (
            pybamm.Variable(
                f"{prefix} {region} electrolyte potential [V]", domain=domain
            )
            for region, domain in regions.items()
        )
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)
        phi_s_n = pybamm.Variable(
            f"{prefix} negative electrode potential [V]", domain="negative electrode"
        )
        phi_s_p = pybamm.Variable(
            f"{prefix} positive electrode potential [V]", domain="positive electrode"
        )
        c_s_n = pybamm.Variable(
            f"{prefix} negative particle concentration [mol.m-3]",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_p = pybamm.Variable(
            f"{prefix} positive particle concentration [mol.m-3]",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )
        # Tied to the volume average of the zone's own field in _set_layer_thermal.
        T = pybamm.Variable(f"{prefix} average temperature [K]")
        fraction, current, i_cell = self._layer_current(layer_id)

        porosity, transport_efficiency = electrolyte_transport(param)
        eps_s_n = pybamm.Parameter("Negative electrode active material volume fraction")
        eps_s_p = pybamm.Parameter("Positive electrode active material volume fraction")
        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ

        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_n = c_s_surf_n / param.n.prim.c_max
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        F_RT = param.F / (param.R * T)
        eta_n = phi_s_n - phi_e_n - param.n.prim.U(sto_surf_n, T)
        eta_p = phi_s_p - phi_e_p - param.p.prim.U(sto_surf_p, T)
        j_n = (
            2
            * param.n.prim.j0(c_e_n, c_s_surf_n, T)
            * pybamm.sinh(param.n.prim.ne / 2 * F_RT * eta_n)
        )
        j_p = (
            2
            * param.p.prim.j0(c_e_p, c_s_surf_p, T)
            * pybamm.sinh(param.p.prim.ne / 2 * F_RT * eta_p)
        )
        a_j_n = a_n * j_n
        a_j_p = a_p * j_p
        a_j = pybamm.concatenation(
            a_j_n, pybamm.PrimaryBroadcast(0, "separator"), a_j_p
        )

        self._set_particle_diffusion(c_s_n, j_n, T, param.n.prim, param.n.prim.c_init)
        self._set_particle_diffusion(c_s_p, j_p, T, param.p.prim, param.p.prim.c_init)
        self._add_stoichiometry_events(prefix, sto_surf_n, sto_surf_p)

        # Scaled by L_x**2 to improve the conditioning of the algebraic equations.
        L_x = param.L_x
        sigma_eff_n = param.n.sigma(sto_surf_n, T) * eps_s_n**param.n.b_s
        sigma_eff_p = param.p.sigma(sto_surf_p, T) * eps_s_p**param.p.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        self.algebraic[phi_s_n] = L_x**2 * (pybamm.div(i_s_n) + a_j_n)
        self.algebraic[phi_s_p] = L_x**2 * (pybamm.div(i_s_p) + a_j_p)
        self.boundary_conditions[phi_s_n] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (i_cell / pybamm.boundary_value(-sigma_eff_p, "right"), "Neumann"),
        }
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = param.ocv_init

        i_e = (param.kappa_e(c_e, T) * transport_efficiency) * (
            param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = L_x**2 * (pybamm.div(i_e) - a_j)
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.n.prim.U_init

        # Migration stays inside the flux so the balance conserves lithium
        # whether or not the transference number depends on c_e.
        flux = (
            -transport_efficiency * param.D_e(c_e, T) * pybamm.grad(c_e)
            + param.t_plus(c_e, T) * i_e / param.F
        )
        self.rhs[c_e] = (-pybamm.div(flux) + a_j / param.F) / porosity
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init

        # Reaction, entropic, and ohmic heat, averaged over the unit cell.
        heat_n = pybamm.x_average(
            a_j_n * (eta_n + T * param.n.prim.dUdT(sto_surf_n))
            - pybamm.inner(i_s_n, pybamm.grad(phi_s_n))
        )
        heat_p = pybamm.x_average(
            a_j_p * (eta_p + T * param.p.prim.dUdT(sto_surf_p))
            - pybamm.inner(i_s_p, pybamm.grad(phi_s_p))
        )
        heat_electrolyte = pybamm.x_average(-pybamm.inner(i_e, pybamm.grad(phi_e)))
        heat = (heat_n * param.n.L + heat_p * param.p.L) / L_x + heat_electrolyte

        return {
            "T_av": T,
            "voltage": pybamm.boundary_value(phi_s_p, "right"),
            "current": current,
            "current_fraction": fraction,
            "heat": heat,
            "variables": {
                c_s_n.name: c_s_n,
                c_s_p.name: c_s_p,
                f"{prefix} X-averaged negative particle concentration [mol.m-3]": (
                    pybamm.x_average(c_s_n)
                ),
                f"{prefix} X-averaged positive particle concentration [mol.m-3]": (
                    pybamm.x_average(c_s_p)
                ),
                f"{prefix} negative particle surface stoichiometry": sto_surf_n,
                f"{prefix} positive particle surface stoichiometry": sto_surf_p,
                f"{prefix} electrolyte concentration [mol.m-3]": c_e,
                f"{prefix} X-averaged electrolyte concentration [mol.m-3]": (
                    pybamm.x_average(c_e)
                ),
                f"{prefix} total lithium in electrolyte per unit cell [mol]": (
                    electrolyte_lithium(param, (c_e_n, c_e_s, c_e_p))
                ),
                f"{prefix} electrolyte potential [V]": phi_e,
                phi_s_n.name: phi_s_n,
                phi_s_p.name: phi_s_p,
            },
        }
