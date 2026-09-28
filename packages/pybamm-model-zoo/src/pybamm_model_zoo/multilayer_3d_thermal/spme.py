"""The multilayer stack with each zone a Single Particle Model with electrolyte."""

from __future__ import annotations

import pybamm
from pybamm_model_zoo.multilayer_3d_thermal.model import (
    MultiLayer3DThermalSPM,
    electrolyte_lithium,
    electrolyte_transport,
)


def _macinnes(ratio: pybamm.Symbol) -> pybamm.Symbol:
    tolerance = pybamm.settings.tolerances["macinnes__c_e"]
    return pybamm.log(pybamm.maximum(ratio, tolerance))


class MultiLayer3DThermalSPMe(MultiLayer3DThermalSPM):
    """:class:`MultiLayer3DThermalSPM` with each zone an SPMe.

    Each zone adds an electrolyte concentration ``c_e(x)`` across its unit cell,
    driven by the SPM's uniform reaction, and the composite concentration
    overpotential and electrolyte and electrode ohmic drops of
    :class:`pybamm.lithium_ion.SPMe`. Those losses are dissipated as heat.

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

    def __init__(
        self,
        num_physical_layers: int = 3,
        num_subdivisions: int | None = None,
        connection: str = "parallel",
        mesh_h: float = 0.1,
        options: dict | None = None,
        name: str = "Multi-Layer 3D Thermal SPMe",
    ) -> None:
        super().__init__(
            num_physical_layers=num_physical_layers,
            num_subdivisions=num_subdivisions,
            connection=connection,
            mesh_h=mesh_h,
            options=options,
            name=name,
        )

    def _electrolyte_and_ohmic_losses(
        self,
        prefix: str,
        current_density: pybamm.Symbol,
        temperature: pybamm.Symbol,
        sto_surf_n: pybamm.Symbol,
        sto_surf_p: pybamm.Symbol,
    ) -> tuple[pybamm.Symbol, pybamm.Symbol, pybamm.Symbol, dict]:
        param = self.param
        L_n, L_s, L_p, L_x = param.n.L, param.s.L, param.p.L, param.L_x
        c_e_n = pybamm.Variable(
            f"{prefix} negative electrolyte concentration [mol.m-3]",
            domain="negative electrode",
        )
        c_e_s = pybamm.Variable(
            f"{prefix} separator electrolyte concentration [mol.m-3]",
            domain="separator",
        )
        c_e_p = pybamm.Variable(
            f"{prefix} positive electrolyte concentration [mol.m-3]",
            domain="positive electrode",
        )
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)
        porosity, transport_efficiency = electrolyte_transport(param)

        # Uniform reaction gives i_e in closed form. Not the standard x_n and x_p,
        # whose current collector domain these zone variables lack.
        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")
        x_p = pybamm.SpatialVariable("x_p", domain="positive electrode")
        i_e = pybamm.concatenation(
            current_density * x_n / L_n,
            pybamm.PrimaryBroadcast(current_density, "separator"),
            current_density * (L_x - x_p) / L_p,
        )
        reaction = pybamm.concatenation(
            pybamm.PrimaryBroadcast(current_density / L_n, "negative electrode"),
            pybamm.PrimaryBroadcast(0, "separator"),
            pybamm.PrimaryBroadcast(-current_density / L_p, "positive electrode"),
        )
        # Migration stays inside the flux so the balance conserves lithium
        # whether or not the transference number depends on c_e.
        flux = (
            -transport_efficiency * param.D_e(c_e, temperature) * pybamm.grad(c_e)
            + param.t_plus(c_e, temperature) * i_e / param.F
        )
        self.rhs[c_e] = (-pybamm.div(flux) + reaction / param.F) / porosity
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init

        c_e_av = pybamm.x_average(c_e)
        kappa = param.kappa_e(c_e_av, temperature)
        efficiency_n, efficiency_s, efficiency_p = (
            pybamm.Parameter(f"{region} porosity") ** bruggeman
            for region, bruggeman in (
                ("Negative electrode", param.n.b_e),
                ("Separator", param.s.b_e),
                ("Positive electrode", param.p.b_e),
            )
        )
        concentration_overpotential = (
            param.chi(c_e_av, temperature)
            * param.R
            * temperature
            / param.F
            * (
                pybamm.x_average(_macinnes(c_e_p / c_e_av))
                - pybamm.x_average(_macinnes(c_e_n / c_e_av))
            )
        )
        electrolyte_ohmic = -current_density * (
            L_n / (3 * kappa * efficiency_n)
            + L_s / (kappa * efficiency_s)
            + L_p / (3 * kappa * efficiency_p)
        )
        sigma_n = param.n.sigma(sto_surf_n, temperature) * pybamm.Parameter(
            "Negative electrode active material volume fraction"
        ) ** (param.n.b_s)
        sigma_p = param.p.sigma(sto_surf_p, temperature) * pybamm.Parameter(
            "Positive electrode active material volume fraction"
        ) ** (param.p.b_s)
        solid_ohmic = -current_density / 3 * (L_n / sigma_n + L_p / sigma_p)

        variables = {
            f"{prefix} electrolyte concentration [mol.m-3]": c_e,
            f"{prefix} X-averaged electrolyte concentration [mol.m-3]": c_e_av,
            f"{prefix} total lithium in electrolyte per unit cell [mol]": (
                electrolyte_lithium(param, (c_e_n, c_e_s, c_e_p))
            ),
            f"{prefix} X-averaged concentration overpotential [V]": (
                concentration_overpotential
            ),
            f"{prefix} X-averaged electrolyte ohmic losses [V]": electrolyte_ohmic,
            f"{prefix} X-averaged solid phase ohmic losses [V]": solid_ohmic,
        }
        return (
            pybamm.x_average(c_e_n),
            pybamm.x_average(c_e_p),
            concentration_overpotential + electrolyte_ohmic + solid_ohmic,
            variables,
        )
