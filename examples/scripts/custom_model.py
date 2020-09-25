#
# Example showing how to create a custom lithium-ion model from submodels
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load lithium-ion base model
model = pybamm.lithium_ion.BaseModel(name="my li-ion model")

# set choice of submodels
model.submodels["external circuit"] = pybamm.external_circuit.CurrentControl(
    model.param
)
model.submodels["current collector"] = pybamm.current_collector.Uniform(model.param)
model.submodels["thermal"] = pybamm.thermal.isothermal.Isothermal(model.param)
model.submodels["porosity"] = pybamm.porosity.Constant(model.param)
model.submodels["negative electrode"] = pybamm.electrode.ohm.LeadingOrder(
    model.param, "Negative"
)
model.submodels["positive electrode"] = pybamm.electrode.ohm.LeadingOrder(
    model.param, "Positive"
)
model.submodels["negative particle"] = pybamm.particle.PolynomialSingleParticle(
    model.param, "Negative", "uniform profile"
)
model.submodels["positive particle"] = pybamm.particle.PolynomialSingleParticle(
    model.param, "Positive", "uniform profile"
)
model.submodels["negative interface"] = pybamm.interface.InverseButlerVolmer(
    model.param, "Negative", "lithium-ion main"
)
model.submodels["positive interface"] = pybamm.interface.InverseButlerVolmer(
    model.param, "Positive", "lithium-ion main"
)
model.submodels[
    "negative interface current"
] = pybamm.interface.CurrentForInverseButlerVolmer(
    model.param, "Negative", "lithium-ion main"
)
model.submodels[
    "positive interface current"
] = pybamm.interface.CurrentForInverseButlerVolmer(
    model.param, "Positive", "lithium-ion main"
)
model.submodels[
    "electrolyte diffusion"
] = pybamm.electrolyte_diffusion.ConstantConcentration(model.param)
model.submodels[
    "electrolyte conductivity"
] = pybamm.electrolyte_conductivity.LeadingOrder(model.param)
model.submodels["negative sei"] = pybamm.sei.NoSEI(model.param, "Negative")
model.submodels["positive sei"] = pybamm.sei.NoSEI(model.param, "Positive")

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
