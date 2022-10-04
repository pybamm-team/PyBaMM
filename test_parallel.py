import pybamm
import numpy as np
import copy
import re



model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.build()

expr = pybamm.numpy_concatenation(sim.built_model.concatenated_rhs, sim.built_model.concatenated_algebraic)

converter = pybamm.JuliaConverter(parallel=False, inline=True)
converter._convert_tree_to_intermediate(expr)
my_jl_str = converter.build_julia_code(topcut_options={"race heuristic": "search_and_sync"})


text_file = open("test_parallel.jl", "w")
text_file.write(my_jl_str)
text_file.close()













