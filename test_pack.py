import pybamm
import numpy as np

a = pybamm.StateVector(slice(0, 2))
A = pybamm.Matrix(np.random.rand(2, 2))
expr = pybamm.numpy_concatenation(A@a, a)

model = pybamm.lithium_ion.DFN(name="DFN")
sim = pybamm.Simulation(model)
sim.build()


expr = pybamm.numpy_concatenation(
    sim.built_model.concatenated_rhs,
    sim.built_model.concatenated_algebraic
)

pack = pybamm.Pack(expr, 50)


myconverter = pybamm.JuliaConverter(parallel=None,inline=False, cache_type="gpu")
myconverter.convert_tree_to_intermediate(pack.built_model)
jl_str = myconverter.build_julia_code()

with open("pack.jl","w") as f:
    f.write(jl_str)
    f.close()