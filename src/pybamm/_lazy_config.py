"""
Single source of truth for lazy loading configuration.

Used by both __init__.py (runtime) and generate_pyi_stub.py (stub generation).

This module defines the SUBMODULE_ALIASES mapping that allows users to access
deeply nested submodules directly from pybamm, e.g., `pybamm.lithium_ion`
instead of `pybamm.models.full_battery_models.lithium_ion`.
"""

SUBMODULE_ALIASES: dict[str, str] = {
    # Battery models (as modules)
    "lead_acid": ".models.full_battery_models.lead_acid",
    "lithium_ion": ".models.full_battery_models.lithium_ion",
    "equivalent_circuit": ".models.full_battery_models.equivalent_circuit",
    "sodium_ion": ".models.full_battery_models.sodium_ion",
    # Submodels
    "active_material": ".models.submodels.active_material",
    "convection": ".models.submodels.convection",
    "current_collector": ".models.submodels.current_collector",
    "electrolyte_conductivity": ".models.submodels.electrolyte_conductivity",
    "electrolyte_diffusion": ".models.submodels.electrolyte_diffusion",
    "electrode": ".models.submodels.electrode",
    "external_circuit": ".models.submodels.external_circuit",
    "interface": ".models.submodels.interface",
    "oxygen_diffusion": ".models.submodels.oxygen_diffusion",
    "particle": ".models.submodels.particle",
    "porosity": ".models.submodels.porosity",
    "thermal": ".models.submodels.thermal",
    "transport_efficiency": ".models.submodels.transport_efficiency",
    "particle_mechanics": ".models.submodels.particle_mechanics",
    "equivalent_circuit_elements": ".models.submodels.equivalent_circuit_elements",
    "kinetics": ".models.submodels.interface.kinetics",
    "sei": ".models.submodels.interface.sei",
    "lithium_plating": ".models.submodels.interface.lithium_plating",
    "interface_utilisation": ".models.submodels.interface.interface_utilisation",
    "open_circuit_potential": ".models.submodels.interface.open_circuit_potential",
    # Geometry
    "standard_spatial_vars": ".geometry.standard_spatial_vars",
    # Parameters
    "constants": ".parameters.constants",
    # Experiments
    "experiment": ".experiment",
    "step": ".experiment.step",
    # Callbacks and telemetry
    "callbacks": ".callbacks",
    "telemetry": ".telemetry",
}
