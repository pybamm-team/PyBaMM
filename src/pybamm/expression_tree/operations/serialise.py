from __future__ import annotations

import importlib
import json
import numbers
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

import pybamm

SUPPORTED_SCHEMA_VERSION = "1.0"


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

    def serialise_model(
        self,
        model: pybamm.BaseModel,
        mesh: pybamm.Mesh | None = None,
        variables: pybamm.FuzzyDict | None = None,
    ) -> dict:
        """Converts a discretised model to a JSON-serialisable dictionary.

        As the model is discretised and ready to solve, only the right hand side,
        algebraic and initial condition variables are serialised.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The discretised model to be serialised
        mesh : :class:`pybamm.Mesh` (optional)
            The mesh the model has been discretised over. Not necessary to solve
            the model when read in, but required to use pybamm's plotting tools.
        variables: :class:`pybamm.FuzzyDict` (optional)
            The discretised model variables. Not necessary to solve a model, but
            required to use pybamm's plotting tools.

        Returns
        -------
        dict
            A JSON-serialisable dictionary representation of the model
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

        return model_json

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
        model_json = self.serialise_model(model, mesh, variables)

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
        if isinstance(obj, Enum):
            return obj.name
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable.")

    @staticmethod
    def serialise_custom_model(model: pybamm.BaseModel) -> dict:
        """
        Converts a custom (non-discretised) PyBaMM model to a JSON-serialisable dictionary.

        This includes symbolic expressions for rhs, algebraic, initial and boundary
        conditions, events, and variables. Works for user defined models that are
        subclasses of BaseModel.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The custom symbolic model to be serialised.

        Returns
        -------
        dict
            A JSON-serialisable dictionary representation of the model

        Raises
        ------
        AttributeError
            If the model is missing required sections
        """
        required_attrs = [
            "rhs",
            "algebraic",
            "initial_conditions",
            "boundary_conditions",
            "events",
            "variables",
        ]
        missing = [attr for attr in required_attrs if not hasattr(model, attr)]
        if missing:
            raise AttributeError(f"Model is missing required sections: {missing}")

        base_cls = model.__class__.__bases__[0] if model.__class__.__bases__ else object
        # If the base class is object or builtins.object, use pybamm.BaseModel instead
        if base_cls is object or (
            base_cls.__module__ == "builtins" and base_cls.__name__ == "object"
        ):
            base_cls_str = "pybamm.BaseModel"
        else:
            base_cls_str = f"{base_cls.__module__}.{base_cls.__name__}"

        model_content = {
            "name": getattr(model, "name", "unnamed_model"),
            "base_class": base_cls_str,
            "options": getattr(model, "options", {}),
            "rhs": [
                (
                    Serialise.convert_symbol_to_json(variable),
                    Serialise.convert_symbol_to_json(rhs_expression),
                )
                for variable, rhs_expression in getattr(model, "rhs", {}).items()
            ],
            "algebraic": [
                (
                    Serialise.convert_symbol_to_json(variable),
                    Serialise.convert_symbol_to_json(algebraic_expression),
                )
                for variable, algebraic_expression in getattr(
                    model, "algebraic", {}
                ).items()
            ],
            "initial_conditions": [
                (
                    Serialise.convert_symbol_to_json(variable),
                    Serialise.convert_symbol_to_json(initial_value),
                )
                for variable, initial_value in getattr(
                    model, "initial_conditions", {}
                ).items()
            ],
            "boundary_conditions": [
                (
                    Serialise.convert_symbol_to_json(variable),
                    {
                        side: [
                            Serialise.convert_symbol_to_json(expression),
                            boundary_type,
                        ]
                        for side, (expression, boundary_type) in conditions.items()
                    },
                )
                for variable, conditions in getattr(
                    model, "boundary_conditions", {}
                ).items()
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
                str(variable_name): Serialise.convert_symbol_to_json(expression)
                for variable_name, expression in getattr(model, "variables", {}).items()
            },
        }

        SCHEMA_VERSION = "1.0"
        model_json = {
            "schema_version": SCHEMA_VERSION,
            "pybamm_version": pybamm.__version__,
            "model": model_content,
        }

        return model_json

    @staticmethod
    def save_custom_model(
        model: pybamm.BaseModel, filename: str | Path | None = None
    ) -> None:
        """
        Saves a custom (non-discretised) PyBaMM model to a JSON file. Works for user defined models that are subclasses of BaseModel.

        This includes symbolic expressions for rhs, algebraic, initial and boundary
        conditions, events, and variables. Useful for storing or sharing models
        before discretisation.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The custom symbolic model to be saved.
        filename : str, optional
            The desired name of the JSON file. If not provided, a name will be
            generated from the model name and current datetime.

        Example
        -------
        >>> import pybamm
        >>> model = pybamm.lithium_ion.BasicDFN()
        >>> from pybamm.expression_tree.operations.serialise import Serialise
        >>> Serialise.save_custom_model(model, "basicdfn_model.json")

        """
        try:
            model_json = Serialise.serialise_custom_model(model)

            # Extract model name for filename generation
            model_name = model_json["model"]["name"]

            if filename is None:
                safe_name = re.sub(r"[^\w\-_.]", "_", model_name or "unnamed_model")
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                filename = f"{safe_name}_{timestamp}.json"
                filename = Path(filename)
            else:
                filename = Path(filename)

                if not filename.name.endswith(".json"):
                    raise ValueError(
                        f"Filename '{filename}' must end with '.json' extension."
                    )

                # Sanitize only the filename, not the directory path
                safe_stem = re.sub(r"[^\w\-_.]", "_", filename.stem)
                filename = filename.with_name(f"{safe_stem}.json")

            try:
                with open(filename, "w") as f:
                    json.dump(model_json, f, indent=2, default=Serialise._json_encoder)
            except OSError as file_err:
                raise OSError(
                    f"Failed to write model JSON to file '{filename}': {file_err}"
                ) from file_err

        except AttributeError:
            # Let AttributeError propagate directly
            raise
        except Exception as e:
            raise ValueError(f"Failed to save custom model: {e}") from e

    @staticmethod
    def _create_symbol_key(symbol_json: dict) -> str:
        """
        Given the JSON‐dict for a symbol, return a unique, hashable key.
        We just sort the dict keys and dump to a string.
        """
        return json.dumps(symbol_json, sort_keys=True)

    @staticmethod
    def load_custom_model(filename: str) -> pybamm.BaseModel:
        """
        Loads a custom (symbolic) PyBaMM model from a JSON file.

        Reconstructs a model saved using `save_custom_model`, including its rhs,
        algebraic equations, initial and boundary conditions, events, and variables.
        Returns a fully symbolic model ready for further processing or discretisation.

        Parameters
        ----------
        filename : str
            Path to the JSON file containing the saved model.

        Returns
        -------
        :class:`pybamm.BaseModel` or subclass
            The reconstructed symbolic PyBaMM model.

        Example
        -------
        >>> import pybamm
        >>> model = pybamm.lithium_ion.BasicDFN()
        >>> from pybamm.expression_tree.operations.serialise import Serialise
        >>> Serialise.save_custom_model(model, "basicdfn_model.json")
        >>> loaded_model = Serialise.load_custom_model("basicdfn_model.json")

        """
        try:
            with open(filename) as file:
                data = json.load(file)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Could not find file: {filename}") from err
        except json.JSONDecodeError as e:
            raise pybamm.InvalidModelJSONError(
                f"The model defined in the file '{filename}' contains invalid JSON: {e!s}"
            ) from e

        # Validate outer structure
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: {SUPPORTED_SCHEMA_VERSION}"
            )

        model_data = data.get("model")
        if model_data is None:
            raise KeyError("Missing 'model' section in JSON file.")

        required = [
            "name",
            "rhs",
            "initial_conditions",
            "base_class",
            "algebraic",
            "boundary_conditions",
            "events",
            "variables",
        ]
        missing = [k for k in required if k not in model_data]
        if missing:
            raise KeyError(f"Missing required model sections: {missing}")

        battery_model = model_data.get("base_class")
        if not battery_model or battery_model.strip() == "pybamm.BaseModel":
            base_cls = pybamm.BaseModel
        else:
            module_name, class_name = battery_model.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                base_cls = getattr(module, class_name)
            except (ModuleNotFoundError, AttributeError) as e:
                if battery_model == "builtins.object":
                    base_cls = pybamm.BaseModel
                else:
                    raise ImportError(
                        f"Could not import base class '{battery_model}': {e}"
                    ) from e

        model = base_cls()
        model.name = model_data["name"]
        model.schema_version = schema_version

        all_variable_keys = (
            [lhs_json for lhs_json, _ in model_data["rhs"]]
            + [lhs_json for lhs_json, _ in model_data["initial_conditions"]]
            + [lhs_json for lhs_json, _ in model_data["algebraic"]]
            + [variable_json for variable_json, _ in model_data["boundary_conditions"]]
        )

        symbol_map = {}
        for variable_json in all_variable_keys:
            try:
                symbol = Serialise.convert_symbol_from_json(variable_json)
                key = Serialise._create_symbol_key(variable_json)
                symbol_map[key] = symbol
            except Exception as e:
                raise ValueError(
                    f"Failed to process symbol key for variable {variable_json}: {e!s}"
                ) from e

        model.rhs = {}
        for lhs_json, rhs_expr_json in model_data["rhs"]:
            try:
                lhs = symbol_map[Serialise._create_symbol_key(lhs_json)]
                rhs = Serialise.convert_symbol_from_json(rhs_expr_json)
                model.rhs[lhs] = rhs
            except Exception as e:
                raise ValueError(
                    f"Failed to convert rhs entry for {lhs_json}: {e!s}"
                ) from e

        model.algebraic = {}
        for lhs_json, algebraic_expr_json in model_data["algebraic"]:
            try:
                lhs = symbol_map[Serialise._create_symbol_key(lhs_json)]
                rhs = Serialise.convert_symbol_from_json(algebraic_expr_json)
                model.algebraic[lhs] = rhs
            except Exception as e:
                raise ValueError(
                    f"Failed to convert algebraic entry for {lhs_json}: {e!s}"
                ) from e

        model.initial_conditions = {}
        for lhs_json, initial_value_json in model_data["initial_conditions"]:
            try:
                lhs = symbol_map[Serialise._create_symbol_key(lhs_json)]
                rhs = Serialise.convert_symbol_from_json(initial_value_json)
                model.initial_conditions[lhs] = rhs
            except Exception as e:
                raise ValueError(
                    f"Failed to convert initial condition entry for {lhs_json}: {e!s}"
                ) from e

        model.boundary_conditions = {}
        for variable_json, condition_dict in model_data["boundary_conditions"]:
            try:
                variable = symbol_map[Serialise._create_symbol_key(variable_json)]
                sides = {}
                for side, (expression_json, boundary_type) in condition_dict.items():
                    try:
                        expr = Serialise.convert_symbol_from_json(expression_json)
                        sides[side] = (expr, boundary_type)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert boundary expression for variable {variable_json} on side '{side}': {e!s}"
                        ) from e
                model.boundary_conditions[variable] = sides
            except Exception as e:
                raise ValueError(
                    f"Failed to convert boundary condition entry for variable {variable_json}: {e!s}"
                ) from e

        model.events = []
        for event_data in model_data["events"]:
            try:
                name = event_data["name"]
                expr = Serialise.convert_symbol_from_json(event_data["expression"])
                event_type = event_data["event_type"]
                model.events.append(pybamm.Event(name, expr, event_type))
            except Exception as e:
                raise ValueError(
                    f"Failed to convert event '{event_data.get('name', 'UNKNOWN')}': {e!s}"
                ) from e

        model.variables = {}
        for variable_name, expression_json in model_data["variables"].items():
            try:
                key = Serialise._create_symbol_key(expression_json)
                symbol = symbol_map.get(key)
                if symbol is None:
                    symbol = Serialise.convert_symbol_from_json(expression_json)
                model.variables[variable_name] = symbol
            except Exception as e:
                raise ValueError(
                    f"Failed to convert variable '{variable_name}': {e!s}"
                ) from e

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
    def convert_symbol_to_json(
        symbol: pybamm.Symbol | numbers.Number | list,
    ) -> dict[str, Any] | numbers.Number | list:
        """
        Recursively converts a PyBaMM symbolic expression into a JSON-serializable format.

        Supports all major PyBaMM symbol types, including :class:`pybamm.Scalar`,
        :class:`pybamm.Variable`, :class:`pybamm.Parameter`, :class:`pybamm.Operator`,
        :class:`pybamm.FunctionParameter`, :class:`pybamm.Broadcast`, and
        :class:`pybamm.Interpolant`.

        Parameters
        ----------
        symbol : pybamm.Symbol or compatible type
            The expression or object to convert.

        Returns
        -------
        dict
            A JSON-compatible representation of the input.

        Examples
        --------
        >>> import pybamm
        >>> from pybamm.expression_tree.operations.serialise import Serialise
        >>> s = pybamm.Scalar(5)
        >>> Serialise.convert_symbol_to_json(s)
        {'type': 'Scalar', 'value': np.float64(5.0)}
        >>> v = pybamm.Variable("c")
        >>> Serialise.convert_symbol_to_json(v)
        {'type': 'Variable', 'name': 'c', 'domains': {'primary': [], 'secondary': [], 'tertiary': [], 'quaternary': []}, 'bounds': [{'type': 'Scalar', 'value': np.float64(-inf)}, {'type': 'Scalar', 'value': np.float64(inf)}]}
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
                "children": [Serialise.convert_symbol_to_json(symbol.child)],
                "broadcast_domain": symbol.broadcast_domain,
            }
        elif isinstance(symbol, pybamm.FunctionParameter):
            input_names = symbol.input_names
            inputs = {
                input_names[i]: Serialise.convert_symbol_to_json(symbol.orphans[i])
                for i in range(len(input_names))
            }
            dv = symbol.diff_variable
            if dv is not None:
                dv_json = Serialise.convert_symbol_to_json(dv)
            else:
                dv_json = None
            json_dict = {
                "type": symbol.__class__.__name__,
                "inputs": inputs,
                "diff_variable": dv_json,
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
                "domains": symbol.domains,
            }
        elif isinstance(symbol, pybamm.SecondaryBroadcast):
            json_dict = {
                "type": "SecondaryBroadcast",
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
    def convert_symbol_from_json(
        json_data: dict[str, Any],
    ) -> pybamm.Symbol | float | int | bool:
        """
        Recursively reconstructs a PyBaMM symbolic expression from a JSON dictionary.

        Supports all major PyBaMM symbol types, including :class:`pybamm.Scalar`, :class:`pybamm.Variable`, :class:`pybamm.Parameter`, :class:`pybamm.Operator`, :class:`pybamm.FunctionParameter`, :class:`pybamm.Broadcast`, and :class:`pybamm.Interpolant`.


        Parameters
        ----------
        json_data : dict
            A JSON-serialized representation of a PyBaMM expression, produced
            by `Serialise.convert_symbol_to_json`.

        Returns
        -------
        pybamm.Symbol or primitive
            The reconstructed PyBaMM symbolic expression or a primitive (float, int, bool).

        Examples
        --------
        >>> import pybamm
        >>> from pybamm.expression_tree.operations.serialise import Serialise
        >>> json_expr = {'type': 'Scalar', 'value': 42}
        >>> Serialise.convert_symbol_from_json(json_expr) # doctest: +SKIP
        Scalar(0x21569ea463d7fb2, 42.0, children=[], domains={})

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
            domains = json_data["domains"]
            return pybamm.FullBroadcast(child, broadcast_domains=domains)
        elif symbol_type == "SecondaryBroadcast":
            child = Serialise.convert_symbol_from_json(json_data["children"][0])
            domain = json_data["broadcast_domain"]
            return pybamm.SecondaryBroadcast(child, domain)
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
                domains=json_data["domains"],
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
