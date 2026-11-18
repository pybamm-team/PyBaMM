"""The multilayer stack with each zone a Single Particle Model with electrolyte."""

from __future__ import annotations

from collections.abc import Callable

import pybamm
from pybamm_model_zoo.multilayer_3d_thermal.model import MultiLayer3DThermalSPM


class MultiLayer3DThermalSPMe(MultiLayer3DThermalSPM):
    """:class:`MultiLayer3DThermalSPM` with each zone PyBaMM's
    :class:`pybamm.lithium_ion.SPMe`.

    Takes the same arguments as :class:`MultiLayer3DThermalSPM`.
    """

    ZONE_MODEL = pybamm.lithium_ion.SPMe

    def __init__(
        self,
        num_physical_layers: int = 3,
        num_subdivisions: int | None = None,
        connection: str = "parallel",
        mesh_h: float = 0.1,
        options: dict | None = None,
        name: str = "Multi-Layer 3D Thermal SPMe",
        *,
        zone_model: Callable[[dict], pybamm.BaseModel] | None = None,
        coating: str = "double-sided",
    ) -> None:
        super().__init__(
            num_physical_layers=num_physical_layers,
            num_subdivisions=num_subdivisions,
            connection=connection,
            mesh_h=mesh_h,
            options=options,
            name=name,
            zone_model=zone_model,
            coating=coating,
        )
