from __future__ import annotations

import importlib
import inspect
import json
import numbers
import re
from datetime import datetime
from enum import Enum

import numpy as np

import pybamm


class Serialise:
    """
    Converts a discretised model to and from a JSON file.

    """

    def __init__(self):
        pass

    class _SymbolEncoder(json.JSONEncoder):
        """Converts PyBaMM symbols into a JSON-serialisable format"""

        def default(self, node: dict):
            node_dict = {"py/object": str(type(node))[8:-2], "py/id": id(node)}
            if isinstance(node, pybamm.Symbol):
                node_dict.update(node.to_json())  # this doesn't include children
                node_dict["children"] = []
                for c in node.children:
                    node_dict["children"].append(self.default(c))

                if hasattr(node, "initial_condition"):  # for ExplicitTimeIntegral
                    node_dict["initial_condition"] = self.default(
                        node.initial_condition
                    )

                return node_dict

            if isinstance(node, pybamm.Event):
                node_dict.update(node.to_json())
                node_dict["expression"] = self.default(node._expression)
                return node_dict

            node_dict["json"] = json.JSONEncoder.default(self, node)  # pragma: no cover
            return node_dict  # pragma: no cover

    class _MeshEncoder(json.JSONEncoder):
        """Converts PyBaMM meshes into a JSON-serialisable format"""

        def default(self, node: pybamm.Mesh):
            node_dict = {"py/object": str(type(node))[8:-2], "py/id": id(node)}
            if isinstance(node, pybamm.Mesh):
                node_dict.update(node.to_json())

                submeshes = {}
                for k, v in node.items():
                    if len(k) == 1 and "ghost cell" not in k[0]:
                        submeshes[k[0]] = self.default(v)

                node_dict["sub_meshes"] = submeshes

                return node_dict

            if isinstance(node, pybamm.SubMesh):
                node_dict.update(node.to_json())
                return node_dict

            node_dict["json"] = json.JSONEncoder.default(self, node)  # pragma: no cover
            return node_dict  # pragma: no cover

    class _Empty:
        """A dummy class to aid deserialisation"""

        pass

    class _EmptyDict(dict):
        """A dummy dictionary class to aid deserialisation"""

        pass

    def save_model(
        self,
        model: pybamm.BaseModel,
        mesh: pybamm.Mesh | None = None,
        variables: pybamm.FuzzyDict | None = None,
        filename: str | None = None,
    ):
        """Saves a discretised model to a JSON file.

        As the model is discretised and ready to solve, only the right hand side,
        algebraic and initial condition variables are saved.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The discretised model to be saved
        mesh : :class:`pybamm.Mesh` (optional)
            The mesh the model has been discretised over. Not neccesary to solve
            the model when read in, but required to use pybamm's plotting tools.
        variables: :class:`pybamm.FuzzyDict` (optional)
            The discretised model varaibles. Not necessary to solve a model, but
            required to use pybamm's plotting tools.
        filename: str (optional)
            The desired name of the JSON file. If no name is provided, one will be
            created based on the model name, and the current datetime.
        """
        if model.is_discretised is False:
            raise NotImplementedError(
                "PyBaMM can only serialise a discretised, ready-to-solve model."
            )

        model_json = {
            "py/object": str(type(model))[8:-2],
            "py/id": id(model),
            "pybamm_version": pybamm.__version__,
            "name": model.name,
            "options": model.options,
            "bounds": [bound.tolist() for bound in model.bounds],  # type: ignore[attr-defined]
            "concatenated_rhs": self._SymbolEncoder().default(model._concatenated_rhs),
            "concatenated_algebraic": self._SymbolEncoder().default(
                model._concatenated_algebraic
            ),
            "concatenated_initial_conditions": self._SymbolEncoder().default(
                model._concatenated_initial_conditions
            ),
            "events": [self._SymbolEncoder().default(event) for event in model.events],
            "mass_matrix": self._SymbolEncoder().default(model.mass_matrix),
            "mass_matrix_inv": self._SymbolEncoder().default(model.mass_matrix_inv),
        }

        if mesh:
            model_json["mesh"] = self._MeshEncoder().default(mesh)

        if variables:
            if model._geometry:
                model_json["geometry"] = self._deconstruct_pybamm_dicts(model._geometry)
            model_json["variables"] = {
                k: self._SymbolEncoder().default(v) for k, v in dict(variables).items()
            }

        if filename is None:
            filename = model.name + "_" + datetime.now().strftime("%Y_%m_%d-%p%I_%M")

        with open(filename + ".json", "w") as f:
            json.dump(model_json, f)

    def load_model(
        self, filename: str, battery_model: pybamm.BaseModel | None = None
    ) -> pybamm.BaseModel:
        """
        Loads a discretised, ready to solve model into PyBaMM.

        A new pybamm battery model instance will be created, which can be solved
        and the results plotted as usual.

        Currently only available for pybamm models which have previously been written
        out using the `save_model()` option.

        Warning: This only loads in discretised models. If you wish to make edits to the
        model or initial conditions, a new model will need to be constructed seperately.

        Parameters
        ----------

        filename: str
            Path to the JSON file containing the serialised model file
        battery_model:  :class:`pybamm.BaseModel` (optional)
            PyBaMM model to be created (e.g. pybamm.lithium_ion.SPM), which will
            override any model names within the file. If None, the function will look
            for the saved object path, present if the original model came from PyBaMM.

        Returns
        -------
        :class:`pybamm.BaseModel`
            A PyBaMM model object, of type specified either in the JSON or in
            `battery_model`.
        """

        with open(filename) as f:
            model_data = json.load(f)

        recon_model_dict = {
            "name": model_data["name"],
            "options": self._convert_options(model_data["options"]),
            "bounds": tuple(np.array(bound) for bound in model_data["bounds"]),
            "concatenated_rhs": self._reconstruct_expression_tree(
                model_data["concatenated_rhs"]
            ),
            "concatenated_algebraic": self._reconstruct_expression_tree(
                model_data["concatenated_algebraic"]
            ),
            "concatenated_initial_conditions": self._reconstruct_expression_tree(
                model_data["concatenated_initial_conditions"]
            ),
            "events": [
                self._reconstruct_expression_tree(event)
                for event in model_data["events"]
            ],
            "mass_matrix": self._reconstruct_expression_tree(model_data["mass_matrix"]),
            "mass_matrix_inv": self._reconstruct_expression_tree(
                model_data["mass_matrix_inv"]
            ),
        }

        recon_model_dict["geometry"] = (
            self._reconstruct_pybamm_dict(model_data["geometry"])
            if "geometry" in model_data.keys()
            else None
        )

        recon_model_dict["mesh"] = (
            self._reconstruct_mesh(model_data["mesh"])
            if "mesh" in model_data.keys()
            else None
        )

        recon_model_dict["variables"] = (
            {
                k: self._reconstruct_expression_tree(v)
                for k, v in model_data["variables"].items()
            }
            if "variables" in model_data.keys()
            else None
        )

        if battery_model:
            return battery_model.deserialise(recon_model_dict)

        if "py/object" in model_data.keys():
            model_framework = self._get_pybamm_class(model_data)
            return model_framework.deserialise(recon_model_dict)

        raise TypeError(
            """
            The PyBaMM battery model to use has not been provided.
            """
        )

    @staticmethod
    def _json_encoder(obj):
        if isinstance(obj, pybamm.Scalar):
            return obj.evaluate()
        if isinstance(obj, pybamm.Parameter):
            return {
                "type": "Parameter",
                "name": obj.name,
                "domain": obj.domain,
            }
        if isinstance(obj, (np.integer | np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Enum):
            return obj.name
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    @staticmethod
    def save_custom_model(model, filename=None):
        """
        Save the custom PyBaMM model and parameters to a JSON file.
        """
        model_json = {
            "pybamm_version": pybamm.__version__,
            "name": getattr(model, "name", "unnamed_model"),
            "options": getattr(model, "options", {}),
            "rhs": [
                (
                    Serialise.convert_symbol_to_json(k),
                    Serialise.convert_symbol_to_json(v),
                )
                for k, v in getattr(model, "rhs", {}).items()
            ],
            "algebraic": [
                [
                    Serialise.convert_symbol_to_json(k),
                    Serialise.convert_symbol_to_json(v),
                ]
                for k, v in model.algebraic.items()
            ],
            "initial_conditions": [
                (
                    Serialise.convert_symbol_to_json(k),
                    Serialise.convert_symbol_to_json(v),
                )
                for k, v in getattr(model, "initial_conditions", {}).items()
            ],
            "boundary_conditions": [
                (
                    Serialise.convert_symbol_to_json(var),
                    {
                        side: [Serialise.convert_symbol_to_json(expr), btype]
                        for side, (expr, btype) in conds.items()
                    },
                )
                for var, conds in getattr(model, "boundary_conditions", {}).items()
            ],
            "events": [
                {
                    "name": event.name,
                    "expression": Serialise.convert_symbol_to_json(event.expression),
                    "event_type": event.event_type,
                }
                for event in getattr(model, "events", [])
            ],
            "variables": {
                str(name): Serialise.convert_symbol_to_json(expr)
                for name, expr in getattr(model, "variables", {}).items()
            },
        }

        if filename is None:
            filename = model.name + "_" + datetime.now().strftime("%Y_%m_%d-%p%I_%M")

        with open(filename + ".json", "w") as f:
            json.dump(model_json, f, indent=2, default=Serialise._json_encoder)

    @staticmethod
    def load_custom_model(filename, battery_model=None):
        with open(filename) as f:
            model_data = json.load(f)

        model = battery_model if battery_model is not None else pybamm.BaseModel()
        model.name = model_data["name"]

        all_keys = (
            [k for k, _ in model_data["rhs"]]
            + [k for k, _ in model_data["initial_conditions"]]
            + [k for k, _ in model_data["algebraic"]]
            + [var for var, _ in model_data["boundary_conditions"]]
        )

        symbol_map = {}
        for k_json in all_keys:
            k = Serialise.convert_symbol_from_json(k_json)
            symbol_map[str(k)] = k

        model.rhs = {
            symbol_map[
                str(Serialise.convert_symbol_from_json(k))
            ]: Serialise.convert_symbol_from_json(v)
            for k, v in model_data["rhs"]
        }
        model.algebraic = {
            symbol_map[
                str(Serialise.convert_symbol_from_json(k))
            ]: Serialise.convert_symbol_from_json(v)
            for k, v in model_data["algebraic"]
        }
        model.initial_conditions = {
            symbol_map[
                str(Serialise.convert_symbol_from_json(k))
            ]: Serialise.convert_symbol_from_json(v)
            for k, v in model_data["initial_conditions"]
        }
        model.boundary_conditions = {
            symbol_map[str(Serialise.convert_symbol_from_json(var))]: {
                side: (Serialise.convert_symbol_from_json(expr), btype)
                for side, (expr, btype) in conds.items()
            }
            for var, conds in model_data["boundary_conditions"]
        }
        model.events = [
            pybamm.Event(
                e["name"],
                Serialise.convert_symbol_from_json(e["expression"]),
                e["event_type"],
            )
            for e in model_data["events"]
        ]
        model.variables = {
            name: symbol_map.get(
                str(Serialise.convert_symbol_from_json(expr)),
                Serialise.convert_symbol_from_json(expr),
            )
            for name, expr in model_data["variables"].items()
        }

        return model

    # Helper functions

    def _get_pybamm_class(self, snippet: dict):
        """Find a pybamm class to initialise from object path"""
        parts = snippet["py/object"].split(".")
        module = importlib.import_module(".".join(parts[:-1]))

        class_ = getattr(module, parts[-1])

        try:
            empty_class = self._Empty()
            empty_class.__class__ = class_

            return empty_class

        except TypeError:
            # Mesh objects have a different layouts
            empty_dict_class = self._EmptyDict()
            empty_dict_class.__class__ = class_

            return empty_dict_class

    def _deconstruct_pybamm_dicts(self, dct: dict):
        """
        Converts dictionaries which contain pybamm classes as keys
        into a json serialisable format.

        Dictionary keys present as pybamm objects are given a seperate key
        as "symbol_<symbol name>" to store the dictionary required to reconstruct
        a symbol, and their seperate key is used in the original dictionary. E.G:

        {'rod':
            {SpatialVariable(name='spat_var'): {"min":0.0, "max":2.0} }
            }

        converts to

        {'rod':
            {'symbol_spat_var': {"min":0.0, "max":2.0} },
        'spat_var':
            {"py/object":pybamm....}
        }

        Dictionaries which don't contain pybamm symbols are returned unchanged.
        """

        def nested_convert(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if isinstance(k, pybamm.Symbol):
                        new_k = self._SymbolEncoder().default(k)
                        new_dict["symbol_" + new_k["name"]] = new_k
                        k = new_k["name"]
                    new_dict[k] = nested_convert(v)
                return new_dict
            return obj

        try:
            _ = json.dumps(dct)
            return dict(dct)
        except TypeError:  # dct must contain pybamm objects
            return nested_convert(dct)

    def _reconstruct_symbol(self, dct: dict):
        """Reconstruct an individual pybamm Symbol"""
        symbol_class = self._get_pybamm_class(dct)
        symbol = symbol_class._from_json(dct)
        return symbol

    def _reconstruct_expression_tree(self, node: dict):
        """
        Loop through an expression tree creating pybamm Symbol classes

        Conducts post-order tree traversal to turn each tree node into a
        `pybamm.Symbol` class, starting from leaf nodes without children and
        working upwards.

        Parameters
        ----------
        node: dict
            A node in an expression tree.
        """
        if "children" in node:
            for i, c in enumerate(node["children"]):
                child_obj = self._reconstruct_expression_tree(c)
                node["children"][i] = child_obj
        elif "expression" in node:
            expression_obj = self._reconstruct_expression_tree(node["expression"])
            node["expression"] = expression_obj

        obj = self._reconstruct_symbol(node)

        return obj

    def _reconstruct_mesh(self, node: dict):
        """Reconstructs a Mesh object"""
        if "sub_meshes" in node:
            for k, v in node["sub_meshes"].items():
                sub_mesh = self._reconstruct_symbol(v)
                node["sub_meshes"][k] = sub_mesh

        new_mesh = self._reconstruct_symbol(node)

        return new_mesh

    def _reconstruct_pybamm_dict(self, obj: dict):
        """
        pybamm.Geometry can contain PyBaMM symbols as dictionary keys.

        Converts
        {"rod":
            {"symbol_spat_var":
                {"min":0.0, "max":2.0} },
            "spat_var":
                {"py/object":"pybamm...."}
        }

        from an exported JSON file to

        {"rod":
            {SpatialVariable(name="spat_var"): {"min":0.0, "max":2.0} }
            }
        """

        def recurse(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if "symbol_" in k:
                        new_dict[k] = self._reconstruct_symbol(v)
                    elif isinstance(v, dict):
                        new_dict[k] = recurse(v)
                    else:
                        new_dict[k] = v

                pattern = re.compile("symbol_")
                symbol_keys = {k: v for k, v in new_dict.items() if pattern.match(k)}

                # rearrange the dictionary to make pybamm objects the dictionary keys
                if symbol_keys:
                    for k, v in symbol_keys.items():
                        new_dict[v] = new_dict[k.lstrip("symbol_")]
                        del new_dict[k]
                        del new_dict[k.lstrip("symbol_")]

                return new_dict
            return obj

        return recurse(obj)

    def _convert_options(self, d):
        """
        Converts a dictionary with nested lists to nested tuples,
        used to convert model options back into correct format
        """
        if isinstance(d, dict):
            return {k: self._convert_options(v) for k, v in d.items()}
        elif isinstance(d, list):
            return tuple(self._convert_options(item) for item in d)
        else:
            return d


    @staticmethod
    def convert_symbol_to_json(symbol):
        """
        Converts a PyBaMM symbolic expression to a JSON-serializable dictionary

        Parameters
        ----------
        symbol : pybamm.Symbol
            The PyBaMM symbolic expression to convert

        Returns
        -------
        dict
            The JSON-serializable dictionary
        """
        if isinstance(symbol, numbers.Number | list):
            return symbol
        elif isinstance(symbol, pybamm.Time):
            return {"type": "Time"}

        elif isinstance(symbol, pybamm.Parameter):
            return {"type": "Parameter", "name": symbol.name}

        elif isinstance(symbol, pybamm.Scalar):
            return {"type": "Scalar", "value": symbol.value}

        elif isinstance(symbol, pybamm.PrimaryBroadcast):
            json_dict = {
                "type": "PrimaryBroadcast",
                "broadcast_domain": symbol.broadcast_domain,
                "children": [Serialise.convert_symbol_to_json(symbol.orphans[0])],
            }

        elif isinstance(symbol, pybamm.FunctionParameter):
            input_names = symbol.input_names
            inputs = {
                input_names[i]: Serialise.convert_symbol_to_json(symbol.orphans[i])
                for i in range(len(input_names))
            }
            diff_variable = symbol.diff_variable
            if diff_variable is not None:
                diff_variable = Serialise.convert_symbol_to_json(diff_variable)
            json_dict = {
                "type": symbol.__class__.__name__,
                "inputs": inputs,
                "diff_variable": diff_variable,
                "name": symbol.name,
                "domains": symbol.domains,
            }

        elif isinstance(symbol, pybamm.Interpolant):
            json_dict = {
                "type": symbol.__class__.__name__,
                "x": [x.tolist() for x in symbol.x],
                "y": symbol.y.tolist(),
                "children": [
                    Serialise.convert_symbol_to_json(c) for c in symbol.children
                ],
                "name": symbol.name,
                "interpolator": symbol.interpolator,
                "entries_string": symbol.entries_string,
            }

        elif isinstance(symbol, pybamm.Variable):
            json_dict = {
                "type": "Variable",
                "name": symbol.name,
                "domains": symbol.domains,
                "bounds": [
                    Serialise.convert_symbol_to_json(symbol.bounds[0]),
                    Serialise.convert_symbol_to_json(symbol.bounds[1]),
                ],
            }

        elif isinstance(symbol, pybamm.ConcatenationVariable):
            json_dict = {
                "type": "ConcatenationVariable",
                "name": symbol.name,
                "children": [
                    Serialise.convert_symbol_to_json(child) for child in symbol.children
                ],
            }
        elif isinstance(symbol, pybamm.FullBroadcast):
            json_dict = {
                "type": "FullBroadcast",
                "children": [Serialise.convert_symbol_to_json(symbol.child)],
                "broadcast_domain": symbol.broadcast_domain,
            }
        elif isinstance(symbol, pybamm.SpatialVariable):
            json_dict = {
                "type": "SpatialVariable",
                "name": symbol.name,
                "domains": symbol.domains,
                "coord_sys": symbol.coord_sys,
            }
        elif isinstance(symbol, pybamm.IndefiniteIntegral):
            integration_var = (
                symbol.integration_variable[0]
                if isinstance(symbol.integration_variable, list)
                else symbol.integration_variable
            )
            json_dict = {
                "type": "IndefiniteIntegral",
                "children": [Serialise.convert_symbol_to_json(symbol.child)],
                "integration_variable": Serialise.convert_symbol_to_json(
                    integration_var
                ),
            }
        elif isinstance(symbol, pybamm.BoundaryValue):
            json_dict = {
                "type": "BoundaryValue",
                "side": symbol.side,
                "children": [Serialise.convert_symbol_to_json(symbol.orphans[0])],
            }
        elif isinstance(symbol, pybamm.SpecificFunction):
            if symbol.__class__ == pybamm.SpecificFunction:
                raise NotImplementedError("SpecificFunction is not supported directly")
            json_dict = {
                "type": symbol.__class__.__name__,
                "children": [
                    Serialise.convert_symbol_to_json(c) for c in symbol.children
                ],
            }

        elif isinstance(symbol, pybamm.UnaryOperator | pybamm.BinaryOperator):
            json_dict = {
                "type": symbol.__class__.__name__,
                "children": [
                    Serialise.convert_symbol_to_json(c) for c in symbol.children
                ],
            }

        elif isinstance(symbol, pybamm.Symbol):
            # Generic fallback for other symbols with children
            json_dict = {
                "type": symbol.__class__.__name__,
                "domains": symbol.domains,
                "children": [
                    Serialise.convert_symbol_to_json(c) for c in symbol.children
                ],
            }
            if hasattr(symbol, "name"):
                json_dict["name"] = symbol.name

        else:
            raise ValueError(
                f"Error processing '{symbol.name}'. Unknown symbol type: {type(symbol)}"
            )
        return json_dict

    @staticmethod
    def convert_symbol_from_json(json_data):
        """
        Recursively converts a JSON dictionary back into PyBaMM symbolic expressions

        Parameters
        ----------
        json_data : dict
            Dictionary containing the serialized PyBaMM expression

        Returns
        -------
        pybamm.Symbol
            The reconstructed PyBaMM symbolic expression
        """
        if isinstance(json_data, float | int | bool):
            return json_data

        if isinstance(json_data, str):
            raise ValueError(f"Unexpected raw string in JSON: {json_data}")

        if json_data is None:
            return None

        symbol_type = json_data.get("type")

        if symbol_type == "Parameter":
            return pybamm.Parameter(
                json_data["name"],
            )
        elif symbol_type == "Scalar":
            return pybamm.Scalar(json_data["value"])
        elif symbol_type == "InputParameter":
            return pybamm.InputParameter(json_data["name"])
        elif symbol_type == "Interpolant":
            return pybamm.Interpolant(
                [np.array(x) for x in json_data["x"]],
                np.array(json_data["y"]),
                [Serialise.convert_symbol_from_json(c) for c in json_data["children"]],
                name=json_data["name"],
                interpolator=json_data["interpolator"],
                entries_string=json_data["entries_string"],
            )
        elif symbol_type == "FunctionParameter":
            diff_variable = json_data["diff_variable"]
            if diff_variable is not None:
                diff_variable = Serialise.convert_symbol_from_json(diff_variable)
            return pybamm.FunctionParameter(
                json_data["name"],
                {
                    k: Serialise.convert_symbol_from_json(v)
                    for k, v in json_data["inputs"].items()
                },
                diff_variable=diff_variable,
            )
        elif symbol_type == "PrimaryBroadcast":
            child = Serialise.convert_symbol_from_json(json_data["children"][0])
            domain = json_data["broadcast_domain"]
            return pybamm.PrimaryBroadcast(child, domain)
        elif symbol_type == "FullBroadcast":
            child = Serialise.convert_symbol_from_json(json_data["children"][0])
            broadcast_domain = json_data.get("broadcast_domain", [])
            return pybamm.FullBroadcast(
                child,
                broadcast_domain,
            )
        elif symbol_type == "BoundaryValue":
            child = Serialise.convert_symbol_from_json(json_data["children"][0])
            side = json_data["side"]
            return pybamm.BoundaryValue(child, side)
        elif symbol_type == "Time":
            return pybamm.t
        elif symbol_type == "Variable":
            bounds = tuple(
                Serialise.convert_symbol_from_json(b)
                for b in json_data.get("bounds", [-float("inf"), float("inf")])
            )
            return pybamm.Variable(
                json_data["name"],
                domains=json_data.get(
                    "domains",
                    {
                        "primary": json_data.get("domain", []),
                        "secondary": [],
                        "tertiary": [],
                        "quaternary": [],
                    },
                ),
                bounds=bounds,
            )
        elif symbol_type == "SpatialVariable":
            return pybamm.SpatialVariable(
                json_data["name"],
                coord_sys=json_data.get("coord_sys", "cartesian"),
                domains=json_data.get("domains"),
            )
        elif symbol_type == "IndefiniteIntegral":
            child = Serialise.convert_symbol_from_json(json_data["children"][0])
            integration_var_json = json_data["integration_variable"]
            integration_variable = Serialise.convert_symbol_from_json(
                integration_var_json
            )
            if not isinstance(integration_variable, pybamm.SpatialVariable):
                raise TypeError(
                    f"Expected SpatialVariable, got {type(integration_variable)}"
                )
            return pybamm.IndefiniteIntegral(child, [integration_variable])
        elif symbol_type == "Symbol":
            return pybamm.Symbol(
                json_data["name"],
                domains=json_data.get("domains", {}),
            )
        elif "children" in json_data:
            return getattr(pybamm, symbol_type)(
                *[Serialise.convert_symbol_from_json(c) for c in json_data["children"]]
            )
        else:
            raise ValueError(f"Unhandled symbol type or malformed entry: {json_data}")
