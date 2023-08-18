import pybamm
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
import json
import numpy as np
import pprint
import importlib
from scipy.sparse import csr_matrix, csr_array
from collections import defaultdict


class SymbolEncoder(json.JSONEncoder):
    def default(self, node):
        node_dict = {"py/object": str(type(node))[8:-2], "py/id": id(node)}
        if isinstance(node, pybamm.Symbol):
            node_dict.update(node.to_json())  # this doesn't include children
            node_dict["children"] = []
            for c in node.children:
                node_dict["children"].append(self.default(c))

            return node_dict

        json_obj = json.JSONEncoder.default(self, node)
        node_dict["json"] = json_obj
        return node_dict


## DECODE


class _Empty:
    pass


def reconstruct_symbol(dct):
    def recreate_slice(d):
        return slice(d["start"], d["stop"], d["step"])

    # decode non-symbol objects here
    # now for pybamm
    foo = _Empty()
    parts = dct["py/object"].split(".")
    try:
        module = importlib.import_module(".".join(parts[:-1]))
    except Exception as ex:
        print(ex)

    class_ = getattr(module, parts[-1])
    foo.__class__ = class_
    # foo = foo._from_json(dct) -> PL: This is what we want eventually

    if isinstance(foo, pybamm.DomainConcatenation):

        def repack_defaultDict(slices):
            slices = defaultdict(list, slices)
            for domain, sls in slices.items():
                sls = [recreate_slice(s) for s in sls]
                slices[domain] = sls
            return slices

        main_slice = repack_defaultDict(dct["slices"])
        child_slice = [repack_defaultDict(s) for s in dct["children_slices"]]

        foo = foo._from_json(
            dct["children"],
            dct["size"],
            main_slice,
            child_slice,
            dct["secondary_dimensions_npts"],
            dct["domains"],
        )

    elif isinstance(foo, pybamm.NumpyConcatenation):
        foo = foo._from_json(
            dct["children"],
            dct["domains"],
        )

    else:
        foo = foo._from_json(dct)

    return foo


def reconstruct_epression_tree(node):
    if "children" in node:
        for i, c in enumerate(node["children"]):
            child_obj = reconstruct_epression_tree(c)
            node["children"][i] = child_obj

    obj = reconstruct_symbol(node)

    return obj


## Run tests
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# # tested all individual trees in rhs
# # tree1 = list(model.rhs.items())[2][1]
# tree1 = (
#     model.y_slices
# )  # Worked: concatenated_rhs, concat_initial_conditions, concatenated_algebraic.
# # Do we need the 'unconcatenated' rhs etc? if not, this gets much easier.
# tree1.visualise("tree1.png")

# json_tree1 = SymbolEncoder().default(tree1)
# with open("test_tree1.json", "w") as f:
#     json.dump(json_tree1, f)

# # pprint.pprint(json_tree1, sort_dicts=False)

# with open("test_tree1.json", "r") as f:
#     data = json.load(f)

# tree1_recon = reconstruct_epression_tree(data)

# print(tree1 == tree1_recon)


# tree1_recon.visualise("recon1.png")

solver_initial = model.default_solver
solution_initial = solver_initial.solve(model, [0, 3600])

# pybamm.plot(solution_initial)
# solution_initial.plot()

model_json = {
    "py/object": str(type(model))[8:-2],
    "py/id": id(model),
    "name": model.name,
    "concatenated_rhs": SymbolEncoder().default(model._concatenated_rhs),
    "concatenated_algebraic": SymbolEncoder().default(model._concatenated_algebraic),
    "concatenated_initial_conditions": SymbolEncoder().default(
        model._concatenated_initial_conditions
    ),
}

# file_name = f"test_{model.name}_stored"
with open("test_full_model.json", "w") as f:
    json.dump(model_json, f)

with open("test_full_model.json", "r") as f:
    model_data = json.load(f)

recon_model_dict = {
    "name": model_data["name"],
    "concatenated_rhs": reconstruct_epression_tree(model_data["concatenated_rhs"]),
    "concatenated_algebraic": reconstruct_epression_tree(
        model_data["concatenated_algebraic"]
    ),
    "concatenated_initial_conditions": reconstruct_epression_tree(
        model_data["concatenated_initial_conditions"]
    ),
}

new_model = pybamm.lithium_ion.DFN.deserialise(recon_model_dict)

new_solver = new_model.default_solver
new_solution = new_solver.solve(model, [0, 3600])

# THIS WORKS!!!
