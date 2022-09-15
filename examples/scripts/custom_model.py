#
# Example showing how to create a custom lithium-ion model from submodels
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load lithium-ion base model
model = pybamm.lithium_ion.BaseModel(name="my li-ion model")

# set choice of submodels
model.submodels["external circuit"] = pybamm.external_circuit.ExplicitCurrentControl(
    model.param, model.options
)
model.submodels["current collector"] = pybamm.current_collector.Uniform(model.param)
model.submodels["thermal"] = pybamm.thermal.isothermal.Isothermal(model.param)
model.submodels["porosity"] = pybamm.porosity.Constant(model.param, model.options)
model.submodels["negative active material"] = pybamm.active_material.Constant(
    model.param, "negative", model.options
)
model.submodels["positive active material"] = pybamm.active_material.Constant(
    model.param, "positive", model.options
)
model.submodels["negative electrode potential"] = pybamm.electrode.ohm.LeadingOrder(
    model.param, "negative"
)
model.submodels["positive electrode potential"] = pybamm.electrode.ohm.LeadingOrder(
    model.param, "positive"
)
particle_n = pybamm.particle.XAveragedPolynomialProfile(
    model.param,
    "negative",
    options={**model.options, "particle": "uniform profile"},
    phase="primary",
)
model.submodels["negative particle"] = particle_n
particle_p = pybamm.particle.XAveragedPolynomialProfile(
    model.param,
    "positive",
    options={**model.options, "particle": "uniform profile"},
    phase="primary",
)
model.submodels["positive particle"] = particle_p

model.submodels[
    "negative open circuit potential"
] = pybamm.open_circuit_potential.SingleOpenCircuitPotential(
    model.param, "negative", "lithium-ion main", options=model.options, phase="primary"
)
model.submodels[
    "positive open circuit potential"
] = pybamm.open_circuit_potential.SingleOpenCircuitPotential(
    model.param, "positive", "lithium-ion main", options=model.options, phase="primary"
)
model.submodels["negative interface"] = pybamm.kinetics.InverseButlerVolmer(
    model.param, "negative", "lithium-ion main", options=model.options
)
model.submodels["positive interface"] = pybamm.kinetics.InverseButlerVolmer(
    model.param, "positive", "lithium-ion main", options=model.options
)
model.submodels["negative interface utilisation"] = pybamm.interface_utilisation.Full(
    model.param, "negative", model.options
)
model.submodels["positive interface utilisation"] = pybamm.interface_utilisation.Full(
    model.param, "positive", model.options
)
model.submodels[
    "negative interface current"
] = pybamm.kinetics.CurrentForInverseButlerVolmer(
    model.param, "negative", "lithium-ion main"
)
model.submodels[
    "positive interface current"
] = pybamm.kinetics.CurrentForInverseButlerVolmer(
    model.param, "positive", "lithium-ion main"
)
model.submodels[
    "electrolyte diffusion"
] = pybamm.electrolyte_diffusion.ConstantConcentration(model.param)
model.submodels[
    "electrolyte conductivity"
] = pybamm.electrolyte_conductivity.LeadingOrder(model.param)
model.submodels[
    "negative surface potential difference"
] = pybamm.electrolyte_conductivity.surface_potential_form.Explicit(
    model.param, "negative"
)
model.submodels[
    "positive surface potential difference"
] = pybamm.electrolyte_conductivity.surface_potential_form.Explicit(
    model.param, "positive"
)
model.submodels["Negative particle mechanics"] = pybamm.particle_mechanics.NoMechanics(
    model.param, "negative", model.options
)
model.submodels["Positive particle mechanics"] = pybamm.particle_mechanics.NoMechanics(
    model.param, "positive", model.options
)
model.submodels["sei"] = pybamm.sei.NoSEI(model.param, model.options)
model.submodels["sei on cracks"] = pybamm.sei.NoSEI(
    model.param, model.options, cracks=True
)
model.submodels["lithium plating"] = pybamm.lithium_plating.NoPlating(model.param)

# build model
model.build_model()

# create geometry
geometry = pybamm.battery_geometry()

# process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
# Note: li-ion base model has defaults for mesh and var_pts
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
# Note: li-ion base model has default spatial methods
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 3600, 100)
solver = pybamm.ScipySolver()
solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(solution)
plot.dynamic_plot()
