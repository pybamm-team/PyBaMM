from __future__ import annotations

import base64
import importlib
import inspect
import json
import numbers
import re
import warnings
import zlib
from datetime import datetime
from enum import Enum
from pathlib import Path

import black
import numpy as np

import pybamm

SUPPORTED_SCHEMA_VERSION = "1.1"


class ExpressionFunctionParameter(pybamm.UnaryOperator):
    def __init__(self, name, child, func_name, func_args):
        super().__init__(name, child)
        self.func_name = func_name
        self.func_args = func_args

    def _unary_evaluate(self, child):
        """Evaluate the symbolic expression (the child)"""
        return child

    def to_source(self):
        """
        Creates python source code for the function.
        """
        src = f"def {self.func_name}({', '.join(self.func_args)}):\n"

        # Fix printing of parameters so they print as Parameter('name'). Do this on a
        # copy to avoid modifying the original expression.
        expression = self.child.create_copy()
        for child in expression.pre_order():
            if isinstance(child, pybamm.FunctionParameter):
                # Replace FunctionParameter with a constructor call
                # Build the inputs dict string mapping input names to actual parameter
                # names
                inputs_str = ", ".join(
                    f'"{input_name}": {child.children[i].name}'
                    for i, input_name in enumerate(child.input_names)
                )
                child.print_name = (
                    f'FunctionParameter("{child.name}", {{{inputs_str}}})'
                )
            elif (
                isinstance(child, pybamm.Parameter) and child.name not in self.func_args
            ):
                child.name = f'Parameter("{child.name}")'

        src += f"    return {expression.to_equation()}"

        formatted_src = black.format_str(src, mode=black.FileMode())
        return formatted_src


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
        variables: None = None,
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
        variables: None (optional)
            This parameter is deprecated and enabled by default.

        Returns
        -------
        dict
            A JSON-serialisable dictionary representation of the model
        """
        if model.is_discretised is False:
            raise NotImplementedError(
                "PyBaMM can only serialise a discretised, ready-to-solve model."
            )
        if variables is not None:
            warnings.warn(
                "The `variables` parameter is deprecated and will be removed in a future version. "
                "Use `model._variables_processed` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        for k in model.variables.keys():
            model.get_processed_variable(k)
        variables_processed = model.get_processed_variables_dict()

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
            "_solution_observable": model._solution_observable.name,
        }

        if mesh:
            model_json["mesh"] = self._MeshEncoder().default(mesh)

        if variables_processed:
            variables_processed = dict(variables_processed)
            if model._geometry:
                model_json["geometry"] = self._deconstruct_pybamm_dicts(model._geometry)
            model_json["_variables_processed"] = {
                k: self._SymbolEncoder().default(v)
                for k, v in variables_processed.items()
            }

        return model_json

    def save_model(
        self,
        model: pybamm.BaseModel,
        mesh: pybamm.Mesh | None = None,
        variables: None = None,
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
        variables: None (optional)
            This parameter is deprecated and enabled by default.
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
        self, filename: str | dict, battery_model: pybamm.BaseModel | None = None
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

        filename: str or dict
            Path to the JSON file containing the serialised model file, or a dictionary
            containing the serialised model data
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

        if isinstance(filename, dict):
            model_data = filename
        else:
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

        vars_processed_data = model_data.get("_variables_processed") or {}
        recon_model_dict["_variables_processed"] = (
            {
                k: self._reconstruct_expression_tree(v)
                for k, v in vars_processed_data.items()
            }
            if vars_processed_data
            else {}
        )

        recon_model_dict["_solution_observable"] = model_data.get(
            "_solution_observable", False
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
    def serialise_custom_model(model: pybamm.BaseModel, compress: bool = False) -> dict:
        """
        Converts a custom (non-discretised) PyBaMM model to a JSON-serialisable dictionary.

        This includes symbolic expressions for rhs, algebraic, initial and boundary
        conditions, events, and variables. Works for user defined models that are
        subclasses of BaseModel.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The custom symbolic model to be serialised.
        compress : bool, optional
            If True, the resulting dictionary will be compressed using zlib and
            encoded as base64. The output will contain a "compressed" flag set to
            True and a "data" field with the compressed payload. Default is False.

        Returns
        -------
        dict
            A JSON-serialisable dictionary representation of the model. If compress
            is True, returns {"compressed": True, "data": <base64-encoded-zlib-data>}.

        Raises
        ------
        AttributeError
            If the model is missing required sections
        """
        if getattr(model, "is_processed", True):
            raise ValueError("Cannot serialise a built model.")

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
                    convert_symbol_to_json(variable),
                    convert_symbol_to_json(rhs_expression),
                )
                for variable, rhs_expression in getattr(model, "rhs", {}).items()
            ],
            "algebraic": [
                (
                    convert_symbol_to_json(variable),
                    convert_symbol_to_json(algebraic_expression),
                )
                for variable, algebraic_expression in getattr(
                    model, "algebraic", {}
                ).items()
            ],
            "initial_conditions": [
                (
                    convert_symbol_to_json(variable),
                    convert_symbol_to_json(initial_value),
                )
                for variable, initial_value in getattr(
                    model, "initial_conditions", {}
                ).items()
            ],
            "boundary_conditions": [
                (
                    convert_symbol_to_json(variable),
                    {
                        side: [
                            convert_symbol_to_json(expression),
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
                    "expression": convert_symbol_to_json(event.expression),
                    "event_type": event.event_type,
                }
                for event in getattr(model, "events", [])
            ],
            "variables": {
                str(variable_name): convert_symbol_to_json(expression)
                for variable_name, expression in getattr(model, "variables", {}).items()
            },
        }

        SCHEMA_VERSION = "1.1"
        model_json = {
            "schema_version": SCHEMA_VERSION,
            "pybamm_version": pybamm.__version__,
            "model": model_content,
        }

        if compress:
            # Serialize to JSON string, compress with zlib, and encode as base64
            json_str = json.dumps(model_json, default=Serialise._json_encoder)
            compressed_bytes = zlib.compress(json_str.encode("utf-8"))
            compressed_b64 = base64.b64encode(compressed_bytes).decode("ascii")
            return {
                "compressed": True,
                "data": compressed_b64,
            }

        return model_json

    @staticmethod
    def save_custom_model(
        model: pybamm.BaseModel,
        filename: str | Path | None = None,
        compress: bool = False,
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
        compress : bool, optional
            If True, the model data will be compressed using zlib before saving.
            This can significantly reduce file size. Default is False.

        Example
        -------
        >>> import pybamm
        >>> model = pybamm.lithium_ion.BasicDFN()
        >>> from pybamm.expression_tree.operations.serialise import Serialise
        >>> Serialise.save_custom_model(model, "basicdfn_model.json")
        >>> # Or with compression:
        >>> Serialise.save_custom_model(model, "basicdfn_model.json", compress=True)

        """
        try:
            model_json = Serialise.serialise_custom_model(model, compress=compress)

            # Extract model name for filename generation
            # When compressed, use the model's name attribute directly
            if compress:
                model_name = getattr(model, "name", "unnamed_model")
            else:
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
    def serialise_custom_geometry(geometry: pybamm.Geometry) -> dict:
        """
        Converts a custom PyBaMM geometry to a JSON-serialisable dictionary.

        Parameters
        ----------
        geometry : :class:`pybamm.Geometry`
            The geometry object to be serialised.

        Returns
        -------
        dict
            A JSON-serialisable dictionary representation of the geometry
        """
        # Serialize the geometry dict using convert_symbol_to_json for nested symbols
        geometry_dict_serialized: dict = {}
        for domain, domain_geom in geometry.items():
            geometry_dict_serialized[domain] = {}
            for key, value in domain_geom.items():
                # Convert SpatialVariable keys to strings and serialize the key itself
                if isinstance(key, pybamm.Symbol):
                    key_str = key.name if hasattr(key, "name") else str(key)
                    geometry_dict_serialized[domain]["symbol_" + key_str] = (
                        convert_symbol_to_json(key)
                    )
                    # Serialize the value dict
                    serialized_value = {}
                    for k, v in value.items():
                        if isinstance(v, pybamm.Symbol):
                            serialized_value[k] = convert_symbol_to_json(v)
                        else:
                            serialized_value[k] = v
                    geometry_dict_serialized[domain][key_str] = serialized_value
                elif isinstance(key, str):
                    # String keys (like 'tabs') - keep as is
                    if isinstance(value, dict):
                        serialized_value = {}
                        for k, v in value.items():
                            if isinstance(v, pybamm.Symbol):
                                serialized_value[k] = convert_symbol_to_json(v)
                            else:
                                serialized_value[k] = v
                        geometry_dict_serialized[domain][key] = serialized_value
                    else:
                        geometry_dict_serialized[domain][key] = value

        SCHEMA_VERSION = "1.1"
        geometry_json = {
            "schema_version": SCHEMA_VERSION,
            "pybamm_version": pybamm.__version__,
            "geometry": geometry_dict_serialized,
        }

        return geometry_json

    @staticmethod
    def save_custom_geometry(
        geometry: pybamm.Geometry, filename: str | Path | None = None
    ) -> None:
        """
        Saves a custom PyBaMM geometry to a JSON file.

        Parameters
        ----------
        geometry : :class:`pybamm.Geometry`
            The geometry object to be saved.
        filename : str or Path, optional
            The desired name of the JSON file. If not provided, a name will be
            generated using current datetime.
        """
        try:
            geometry_json = Serialise.serialise_custom_geometry(geometry)

            if filename is None:
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                filename = f"geometry_{timestamp}.json"
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
                    json.dump(
                        geometry_json, f, indent=2, default=Serialise._json_encoder
                    )
            except OSError as file_err:
                raise OSError(
                    f"Failed to write geometry JSON to file '{filename}': {file_err}"
                ) from file_err

        except Exception as e:
            raise ValueError(f"Failed to save custom geometry: {e}") from e

    @staticmethod
    def load_custom_geometry(filename: str | dict) -> pybamm.Geometry:
        """
        Loads a custom PyBaMM geometry from a JSON file or dictionary.

        Parameters
        ----------
        filename : str or dict
            Path to the JSON file containing the saved geometry, or a dictionary
            containing the serialised geometry data.

        Returns
        -------
        :class:`pybamm.Geometry`
            The reconstructed geometry object.
        """
        if isinstance(filename, dict):
            data = filename
        else:
            try:
                with open(filename) as file:
                    data = json.load(file)
            except FileNotFoundError as err:
                raise FileNotFoundError(f"Could not find file: {filename}") from err
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"The file '{filename}' contains invalid JSON: {e!s}"
                ) from e

        # Validate schema version
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: {SUPPORTED_SCHEMA_VERSION}"
            )

        # Extract geometry data
        geometry_data = data.get("geometry")
        if geometry_data is None:
            raise KeyError("Missing 'geometry' section in JSON data.")

        # Reconstruct geometry
        reconstructed_geometry: dict = {}

        for domain, domain_geom in geometry_data.items():
            reconstructed_geometry[domain] = {}

            # Find symbol keys and reconstruct SpatialVariables
            symbol_keys = {}
            for key in domain_geom.keys():
                if key.startswith("symbol_"):
                    var_name = key[7:]  # Remove "symbol_" prefix
                    symbol_keys[var_name] = convert_symbol_from_json(domain_geom[key])

            # Now reconstruct the domain geometry with proper keys
            for key, value in domain_geom.items():
                if key.startswith("symbol_"):
                    continue  # Skip symbol definitions

                if key in symbol_keys:
                    # Use the reconstructed SpatialVariable as key
                    spatial_var = symbol_keys[key]
                    reconstructed_value = {}
                    for k, v in value.items():
                        if isinstance(v, dict) and "type" in v:
                            # Reconstruct PyBaMM Symbol using convert_symbol_from_json
                            reconstructed_value[k] = convert_symbol_from_json(v)
                        else:
                            reconstructed_value[k] = v
                    reconstructed_geometry[domain][spatial_var] = reconstructed_value
                else:
                    # String key (like 'tabs')
                    if isinstance(value, dict):
                        reconstructed_value = {}
                        for k, v in value.items():
                            if isinstance(v, dict) and "type" in v:
                                reconstructed_value[k] = convert_symbol_from_json(v)
                            else:
                                reconstructed_value[k] = v
                        reconstructed_geometry[domain][key] = reconstructed_value
                    else:
                        reconstructed_geometry[domain][key] = value

        return pybamm.Geometry(reconstructed_geometry)

    @staticmethod
    def serialise_spatial_method_item(method) -> dict:
        """
        Serialise a single spatial method instance.

        Parameters
        ----------
        method : SpatialMethod
            A spatial method instance (e.g. FiniteVolume(), ZeroDimensionalSpatialMethod()).

        Returns
        -------
        dict
            JSON-serialisable dict with "class", "module", and "options".
        """
        return {
            "class": type(method).__name__,
            "module": type(method).__module__,
            "options": method.options if hasattr(method, "options") else {},
        }

    @staticmethod
    def deserialise_spatial_method_item(method_info: dict):
        """
        Deserialise a single spatial method from a dict (one entry from spatial_methods).

        Parameters
        ----------
        method_info : dict
            Dict with "class", "module", and optionally "options".

        Returns
        -------
        SpatialMethod
            A spatial method instance.
        """
        module_name = method_info["module"]
        class_name = method_info["class"]
        options = method_info.get("options") or {}
        module = importlib.import_module(module_name)
        method_class = getattr(module, class_name)
        return method_class(options=options)

    @staticmethod
    def serialise_spatial_methods(spatial_methods: dict) -> dict:
        """
        Converts a dictionary of spatial methods to a JSON-serialisable dictionary.

        Parameters
        ----------
        spatial_methods : dict
            Dictionary mapping domain names to spatial method instances.

        Returns
        -------
        dict
            A JSON-serialisable dictionary representation of the spatial methods
        """
        spatial_methods_dict = {}
        for domain, method in spatial_methods.items():
            spatial_methods_dict[domain] = Serialise.serialise_spatial_method_item(
                method
            )

        SCHEMA_VERSION = "1.1"
        spatial_methods_json = {
            "schema_version": SCHEMA_VERSION,
            "pybamm_version": pybamm.__version__,
            "spatial_methods": spatial_methods_dict,
        }

        return spatial_methods_json

    @staticmethod
    def save_spatial_methods(
        spatial_methods: dict, filename: str | Path | None = None
    ) -> None:
        """
        Saves spatial methods to a JSON file.

        Parameters
        ----------
        spatial_methods : dict
            Dictionary mapping domain names to spatial method instances.
        filename : str or Path, optional
            The desired name of the JSON file. If not provided, a name will be
            generated using current datetime.
        """
        try:
            spatial_methods_json = Serialise.serialise_spatial_methods(spatial_methods)

            if filename is None:
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                filename = f"spatial_methods_{timestamp}.json"
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
                    json.dump(
                        spatial_methods_json,
                        f,
                        indent=2,
                        default=Serialise._json_encoder,
                    )
            except OSError as file_err:
                raise OSError(
                    f"Failed to write spatial methods JSON to file '{filename}': {file_err}"
                ) from file_err

        except Exception as e:
            raise ValueError(f"Failed to save spatial methods: {e}") from e

    @staticmethod
    def load_spatial_methods(filename: str | dict) -> dict:
        """
        Loads spatial methods from a JSON file or dictionary.

        Parameters
        ----------
        filename : str or dict
            Path to the JSON file containing the saved spatial methods, or a dictionary
            containing the serialised spatial methods data.

        Returns
        -------
        dict
            Dictionary mapping domain names to spatial method instances.
        """
        if isinstance(filename, dict):
            data = filename
        else:
            try:
                with open(filename) as file:
                    data = json.load(file)
            except FileNotFoundError as err:
                raise FileNotFoundError(f"Could not find file: {filename}") from err
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"The file '{filename}' contains invalid JSON: {e!s}"
                ) from e

        # Validate schema version
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: {SUPPORTED_SCHEMA_VERSION}"
            )

        # Extract spatial methods data
        spatial_methods_data = data.get("spatial_methods")
        if spatial_methods_data is None:
            raise KeyError("Missing 'spatial_methods' section in JSON data.")

        # Reconstruct spatial methods
        reconstructed_methods = {}
        for domain, method_info in spatial_methods_data.items():
            try:
                reconstructed_methods[domain] = (
                    Serialise.deserialise_spatial_method_item(method_info)
                )
            except (ModuleNotFoundError, AttributeError) as e:
                class_name = method_info.get("class", "?")
                module_name = method_info.get("module", "?")
                raise ImportError(
                    f"Could not import spatial method '{class_name}' from '{module_name}': {e}"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Failed to reconstruct spatial method for domain '{domain}': {e}"
                ) from e

        return reconstructed_methods

    @staticmethod
    def serialise_var_pts(var_pts: dict) -> dict:
        """
        Converts a var_pts dictionary to a JSON-serialisable dictionary.

        Parameters
        ----------
        var_pts : dict
            Dictionary mapping spatial variable names (str or SpatialVariable) to
            number of points (int).

        Returns
        -------
        dict
            A JSON-serialisable dictionary representation of var_pts
        """
        # Convert all keys to strings
        var_pts_dict = {}
        for key, value in var_pts.items():
            if isinstance(key, str):
                var_pts_dict[key] = value
            elif hasattr(key, "name"):
                # SpatialVariable or similar object with name attribute
                var_pts_dict[key.name] = value
            else:
                raise ValueError(f"Unexpected key type in var_pts: {type(key)}")

        SCHEMA_VERSION = "1.1"
        var_pts_json = {
            "schema_version": SCHEMA_VERSION,
            "pybamm_version": pybamm.__version__,
            "var_pts": var_pts_dict,
        }

        return var_pts_json

    @staticmethod
    def save_var_pts(var_pts: dict, filename: str | Path | None = None) -> None:
        """
        Saves var_pts to a JSON file.

        Parameters
        ----------
        var_pts : dict
            Dictionary mapping spatial variable names to number of points.
        filename : str or Path, optional
            The desired name of the JSON file. If not provided, a name will be
            generated using current datetime.
        """
        try:
            var_pts_json = Serialise.serialise_var_pts(var_pts)

            if filename is None:
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                filename = f"var_pts_{timestamp}.json"
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
                    json.dump(
                        var_pts_json, f, indent=2, default=Serialise._json_encoder
                    )
            except OSError as file_err:
                raise OSError(
                    f"Failed to write var_pts JSON to file '{filename}': {file_err}"
                ) from file_err

        except Exception as e:
            raise ValueError(f"Failed to save var_pts: {e}") from e

    @staticmethod
    def load_var_pts(filename: str | dict) -> dict:
        """
        Loads var_pts from a JSON file or dictionary.

        Parameters
        ----------
        filename : str or dict
            Path to the JSON file containing the saved var_pts, or a dictionary
            containing the serialised var_pts data.

        Returns
        -------
        dict
            Dictionary mapping spatial variable names (strings) to number of points.
        """
        if isinstance(filename, dict):
            data = filename
        else:
            try:
                with open(filename) as file:
                    data = json.load(file)
            except FileNotFoundError as err:
                raise FileNotFoundError(f"Could not find file: {filename}") from err
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"The file '{filename}' contains invalid JSON: {e!s}"
                ) from e

        # Validate schema version
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: {SUPPORTED_SCHEMA_VERSION}"
            )

        # Extract var_pts data
        var_pts_data = data.get("var_pts")
        if var_pts_data is None:
            raise KeyError("Missing 'var_pts' section in JSON data.")

        return var_pts_data

    @staticmethod
    def serialise_submesh_item(submesh_item) -> dict:
        """
        Serialise a single submesh type (SubMesh class or MeshGenerator instance).

        Parameters
        ----------
        submesh_item : type or MeshGenerator
            A SubMesh class (e.g. Uniform1DSubMesh) or a MeshGenerator instance.

        Returns
        -------
        dict
            JSON-serialisable dict with "class", "module", and optionally
            "submesh_params" for MeshGenerator.
        """
        if hasattr(submesh_item, "submesh_type"):
            submesh_class = submesh_item.submesh_type
            result = {
                "class": submesh_class.__name__,
                "module": submesh_class.__module__,
            }
            if getattr(submesh_item, "submesh_params", None):
                result["submesh_params"] = dict(submesh_item.submesh_params)
            return result
        # SubMesh class
        return {
            "class": submesh_item.__name__,
            "module": submesh_item.__module__,
        }

    @staticmethod
    def deserialise_submesh_item(
        submesh_info: dict, return_class_only: bool = False
    ):
        """
        Deserialise a single submesh type from a dict (one entry from submesh_types).

        Parameters
        ----------
        submesh_info : dict
            Dict with "class", "module", and optionally "submesh_params".
        return_class_only : bool, optional
            If True, return the SubMesh class. If False, return a MeshGenerator
            instance. Default is False.

        Returns
        -------
        type or MeshGenerator
            The SubMesh class or a MeshGenerator instance.
        """
        module_name = submesh_info["module"]
        class_name = submesh_info["class"]
        module = importlib.import_module(module_name)
        submesh_class = getattr(module, class_name)
        if return_class_only:
            return submesh_class
        params = submesh_info.get("submesh_params") or {}
        return pybamm.MeshGenerator(submesh_class, params)

    @staticmethod
    def serialise_submesh_types(submesh_types: dict) -> dict:
        """
        Converts a dictionary of submesh types to a JSON-serialisable dictionary.

        Parameters
        ----------
        submesh_types : dict
            Dictionary mapping domain names to submesh classes or MeshGenerator objects.

        Returns
        -------
        dict
            A JSON-serialisable dictionary representation of the submesh types
        """
        submesh_types_dict = {}
        for domain, submesh_item in submesh_types.items():
            submesh_types_dict[domain] = Serialise.serialise_submesh_item(
                submesh_item
            )

        SCHEMA_VERSION = "1.1"
        submesh_types_json = {
            "schema_version": SCHEMA_VERSION,
            "pybamm_version": pybamm.__version__,
            "submesh_types": submesh_types_dict,
        }

        return submesh_types_json

    @staticmethod
    def save_submesh_types(
        submesh_types: dict, filename: str | Path | None = None
    ) -> None:
        """
        Saves submesh types to a JSON file.

        Parameters
        ----------
        submesh_types : dict
            Dictionary mapping domain names to submesh classes.
        filename : str or Path, optional
            The desired name of the JSON file. If not provided, a name will be
            generated using current datetime.
        """
        try:
            submesh_types_json = Serialise.serialise_submesh_types(submesh_types)

            if filename is None:
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                filename = f"submesh_types_{timestamp}.json"
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
                    json.dump(
                        submesh_types_json, f, indent=2, default=Serialise._json_encoder
                    )
            except OSError as file_err:
                raise OSError(
                    f"Failed to write submesh types JSON to file '{filename}': {file_err}"
                ) from file_err

        except Exception as e:
            raise ValueError(f"Failed to save submesh types: {e}") from e

    @staticmethod
    def load_submesh_types(filename: str | dict) -> dict:
        """
        Loads submesh types from a JSON file or dictionary.

        Parameters
        ----------
        filename : str or dict
            Path to the JSON file containing the saved submesh types, or a dictionary
            containing the serialised submesh types data.

        Returns
        -------
        dict
            Dictionary mapping domain names to MeshGenerator objects.
        """
        if isinstance(filename, dict):
            data = filename
        else:
            try:
                with open(filename) as file:
                    data = json.load(file)
            except FileNotFoundError as err:
                raise FileNotFoundError(f"Could not find file: {filename}") from err
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"The file '{filename}' contains invalid JSON: {e!s}"
                ) from e

        # Validate schema version
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: {SUPPORTED_SCHEMA_VERSION}"
            )

        # Extract submesh types data
        submesh_types_data = data.get("submesh_types")
        if submesh_types_data is None:
            raise KeyError("Missing 'submesh_types' section in JSON data.")

        # Reconstruct submesh types
        reconstructed_submesh_types = {}
        for domain, submesh_info in submesh_types_data.items():
            try:
                reconstructed_submesh_types[domain] = (
                    Serialise.deserialise_submesh_item(
                        submesh_info, return_class_only=False
                    )
                )
            except (ModuleNotFoundError, AttributeError) as e:
                class_name = submesh_info.get("class", "?")
                module_name = submesh_info.get("module", "?")
                raise ImportError(
                    f"Could not import submesh type '{class_name}' from '{module_name}': {e}"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Failed to reconstruct submesh type for domain '{domain}': {e}"
                ) from e

        return reconstructed_submesh_types

    @staticmethod
    def _create_symbol_key(symbol_json: dict) -> str:
        """
        Given the JSON‐dict for a symbol, return a unique, hashable key.
        We just sort the dict keys and dump to a string.
        """
        return json.dumps(symbol_json, sort_keys=True)

    @staticmethod
    def load_custom_model(filename: str | dict) -> pybamm.BaseModel:
        """
        Loads a custom (symbolic) PyBaMM model from a JSON file or dictionary.

        Reconstructs a model saved using `save_custom_model`, including its rhs,
        algebraic equations, initial and boundary conditions, events, and variables.
        Returns a fully symbolic model ready for further processing or discretisation.

        Automatically detects and decompresses data that was serialised with
        compression enabled (compress=True in serialise_custom_model).

        Parameters
        ----------
        filename : str or dict
            Path to the JSON file containing the saved model, or a dictionary
            containing the serialised model data (optionally compressed).

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
        if isinstance(filename, dict):
            data = filename
        else:
            try:
                with open(filename) as file:
                    data = json.load(file)
            except FileNotFoundError as err:
                raise FileNotFoundError(f"Could not find file: {filename}") from err
            except json.JSONDecodeError as e:
                raise pybamm.InvalidModelJSONError(
                    f"The model defined in the file '{filename}' contains invalid JSON: {e!s}"
                ) from e

        # Check if the data is compressed and decompress if needed
        if data.get("compressed", False):
            try:
                compressed_b64 = data["data"]
                compressed_bytes = base64.b64decode(compressed_b64)
                json_str = zlib.decompress(compressed_bytes).decode("utf-8")
                data = json.loads(json_str)
            except (KeyError, zlib.error, base64.binascii.Error) as e:
                raise ValueError(f"Failed to decompress model data: {e}") from e

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
        # Restore options so round-trip serialisation produces an equivalent model
        opts = model_data.get("options", {})
        if opts is not None:
            model.options = dict(opts)

        all_variable_keys = (
            [lhs_json for lhs_json, _ in model_data["rhs"]]
            + [lhs_json for lhs_json, _ in model_data["initial_conditions"]]
            + [lhs_json for lhs_json, _ in model_data["algebraic"]]
            + [variable_json for variable_json, _ in model_data["boundary_conditions"]]
        )

        symbol_map = {}
        for variable_json in all_variable_keys:
            try:
                symbol = convert_symbol_from_json(variable_json)
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
                rhs = convert_symbol_from_json(rhs_expr_json)
                model.rhs[lhs] = rhs
            except Exception as e:
                raise ValueError(
                    f"Failed to convert rhs entry for {lhs_json}: {e!s}"
                ) from e

        model.algebraic = {}
        for lhs_json, algebraic_expr_json in model_data["algebraic"]:
            try:
                lhs = symbol_map[Serialise._create_symbol_key(lhs_json)]
                rhs = convert_symbol_from_json(algebraic_expr_json)
                model.algebraic[lhs] = rhs
            except Exception as e:
                raise ValueError(
                    f"Failed to convert algebraic entry for {lhs_json}: {e!s}"
                ) from e

        model.initial_conditions = {}
        for lhs_json, initial_value_json in model_data["initial_conditions"]:
            try:
                lhs = symbol_map[Serialise._create_symbol_key(lhs_json)]
                rhs = convert_symbol_from_json(initial_value_json)
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
                        expr = convert_symbol_from_json(expression_json)
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
                expr = convert_symbol_from_json(event_data["expression"])
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
                    symbol = convert_symbol_from_json(expression_json)
                model.variables[variable_name] = symbol
            except Exception as e:
                raise ValueError(
                    f"Failed to convert variable '{variable_name}': {e!s}"
                ) from e

        # Restore observable state
        model._solution_observable = False

        return model

    @staticmethod
    def save_parameters(parameters: dict, filename=None):
        """
        Serializes a dictionary of parameters to a JSON file.
        The values can be numbers, PyBaMM symbols, or callables.

        Parameters
        ----------
        parameters : dict
            A dictionary of parameter names and values.
            Values can be numeric, PyBaMM symbols, or callables.

        filename : str, optional
            If given, saves the serialized parameters to this file.
        """
        parameter_values_dict = {}

        for k, v in parameters.items():
            if callable(v):
                parameter_values_dict[k] = convert_symbol_to_json(
                    convert_function_to_symbolic_expression(v, k)
                )
            else:
                parameter_values_dict[k] = convert_symbol_to_json(v)

        if filename is not None:
            with open(filename, "w") as f:
                json.dump(parameter_values_dict, f, indent=4)

    @staticmethod
    def load_parameters(filename):
        """
        Load a JSON file of parameters (either from Serialise.save_parameters
        or from a standard pybamm.ParameterValues.save), and return a
        pybamm.ParameterValues object.

        - If a value is a dict with a "type" key, deserialize it as a PyBaMM symbol.
        - Otherwise (float, int, bool, str, list, dict-without-type), leave it as-is.
        """
        with open(filename) as f:
            raw_dict = json.load(f)

        deserialized = {}
        for key, val in raw_dict.items():
            if isinstance(val, dict) and "type" in val:
                deserialized[key] = convert_symbol_from_json(val)

            elif isinstance(val, list):
                deserialized[key] = val

            elif isinstance(val, (numbers.Number | bool)):
                deserialized[key] = val

            elif isinstance(val, str):
                deserialized[key] = val

            elif isinstance(val, dict):
                deserialized[key] = val

            else:
                raise ValueError(
                    f"Unsupported parameter format for key '{key}': {val!r}"
                )

        return pybamm.ParameterValues(deserialized)

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
    def _to_json_safe(value):
        """Convert a value to a JSON-serializable form (native Python types).

        Handles numpy scalars, arrays, booleans, and nested dicts/lists.
        """
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, dict):
            return {k: Serialise._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Serialise._to_json_safe(v) for v in value]
        return value

    @staticmethod
    def serialise_experiment(experiment) -> dict:
        """Convert a :class:`pybamm.Experiment` to a JSON-serialisable dict.

        Returns ``{"cycles": [[step_config, ...], ...]}``, grouping steps
        into cycles according to ``experiment.cycle_lengths``.

        Parameters
        ----------
        experiment : :class:`pybamm.Experiment`
            The experiment to serialise.

        Returns
        -------
        dict
            Config dict with key ``"cycles"``.
        """
        step_type_map = {
            "Current": "current",
            "Voltage": "voltage",
            "Power": "power",
            "CRate": "c-rate",
        }
        termination_type_map = {
            "VoltageTermination": "voltage",
            "CurrentTermination": "current",
            "CrateTermination": "c-rate",
            "CRateTermination": "c-rate",
        }

        def _serialise_step(step):
            step_class_name = step.__class__.__name__
            step_type = step_type_map.get(step_class_name, step_class_name.lower())

            # Current with value 0 is a rest step
            if step_class_name == "Current" and step.value == 0:
                step_type = "rest"

            step_config = {"type": step_type, "duration": step.duration}

            if step_type != "rest":
                value = step.value
                if isinstance(value, pybamm.InputParameter):
                    param_name = value.name
                    step_config["value"] = (
                        param_name if isinstance(param_name, str) else str(value)
                    )
                elif isinstance(value, (int, float, str)):
                    step_config["value"] = value
                else:
                    step_config["value"] = str(value)

            if step.termination:
                terminations = []
                for term in step.termination:
                    term_class_name = term.__class__.__name__
                    term_type = termination_type_map.get(
                        term_class_name, term_class_name.lower()
                    )
                    term_config = {"type": term_type, "value": term.value}
                    if hasattr(term, "operator") and term.operator:
                        term_config["operator"] = term.operator
                    terminations.append(term_config)
                step_config["terminations"] = terminations

            return step_config

        steps_config = [_serialise_step(step) for step in experiment.steps]

        cycles_config = []
        step_idx = 0
        for cycle_length in experiment.cycle_lengths:
            cycles_config.append(steps_config[step_idx : step_idx + cycle_length])
            step_idx += cycle_length

        return {"cycles": cycles_config}

    @staticmethod
    def deserialise_experiment(data: dict):
        """Convert a config dict to a :class:`pybamm.Experiment`.

        Accepts ``{"cycles": [[step_config, ...], ...]}`` (new format) or
        ``{"steps": [step_config, ...]}`` (legacy flat format).

        Parameters
        ----------
        data : dict
            Config dict as produced by :meth:`serialise_experiment`.

        Returns
        -------
        :class:`pybamm.Experiment`
        """
        step_func_map = {
            "current": pybamm.step.current,
            "voltage": pybamm.step.voltage,
            "power": pybamm.step.power,
            "c-rate": pybamm.step.c_rate,
            "rest": pybamm.step.current,
        }
        term_class_map = {
            "voltage": pybamm.step.VoltageTermination,
            "current": pybamm.step.CurrentTermination,
            "c-rate": pybamm.step.CrateTermination,
        }

        def _parse_termination(term_dict):
            term_type = term_dict.get("type")
            if term_type not in term_class_map:
                raise ValueError(
                    f"Unknown termination type: {term_type!r}. "
                    f"Expected one of {list(term_class_map)!r}."
                )
            value = float(term_dict["value"])
            operator = term_dict.get("operator")
            return term_class_map[term_type](value, operator=operator)

        def _parse_step(step_dict):
            step_type = step_dict.get("type")
            if step_type not in step_func_map:
                raise ValueError(
                    f"Unknown step type: {step_type!r}. "
                    f"Expected one of {list(step_func_map)!r}."
                )
            step_func = step_func_map[step_type]

            if step_type == "rest":
                value = 0.0
            elif "value" in step_dict and step_dict["value"] is not None:
                raw = step_dict["value"]
                try:
                    value = float(raw)
                except (ValueError, TypeError):
                    if isinstance(raw, str):
                        value = pybamm.InputParameter(raw)
                    else:
                        raise
            else:
                raise ValueError(f"Value is required for {step_type!r} steps.")

            duration = float(step_dict.get("duration", 86400))
            terminations = None
            if step_dict.get("terminations"):
                terminations = [
                    _parse_termination(t) for t in step_dict["terminations"]
                ]

            return step_func(value, duration=duration, termination=terminations)

        if "cycles" in data and data["cycles"] is not None:
            processed_cycles = []
            for cycle_steps in data["cycles"]:
                processed_cycle = tuple(_parse_step(s) for s in cycle_steps)
                processed_cycles.append(processed_cycle)
            return pybamm.Experiment(processed_cycles)
        elif "steps" in data and data["steps"] is not None:
            processed_steps = [_parse_step(s) for s in data["steps"]]
            return pybamm.Experiment(processed_steps)
        else:
            raise ValueError(
                "Experiment config must have 'steps' or 'cycles'."
            )

    @staticmethod
    def serialise_solver(solver) -> dict:
        """Convert a :class:`pybamm.BaseSolver` to a JSON-serialisable config dict.

        Uses ``inspect.signature`` to discover ``__init__`` parameters, reads
        the corresponding attribute values from the instance (trying both
        ``solver.<name>`` and ``solver._<name>``), and filters out values that
        are not JSON-serialisable.  Handles ``CompositeSolver`` recursively.

        Parameters
        ----------
        solver : :class:`pybamm.BaseSolver`
            The solver to serialise.

        Returns
        -------
        dict
            Config dict with a ``"type"`` key and one key per serialisable
            init parameter.
        """
        if solver.__class__.__name__ == "CompositeSolver":
            return {
                "type": "CompositeSolver",
                "sub_solvers": [
                    Serialise.serialise_solver(sub) for sub in solver.sub_solvers
                ],
            }

        config = {"type": solver.__class__.__name__}

        sig = inspect.signature(solver.__class__.__init__)
        for param_name in sig.parameters:
            if param_name == "self":
                continue

            value = None
            found = False
            for attr_name in (param_name, f"_{param_name}"):
                if hasattr(solver, attr_name):
                    value = getattr(solver, attr_name)
                    found = True
                    break

            if not found:
                continue

            value = Serialise._to_json_safe(value)
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue

            config[param_name] = value

        return config

    @staticmethod
    def deserialise_solver(data: dict):
        """Convert a config dict to a :class:`pybamm.BaseSolver` instance.

        Handles ``CompositeSolver`` by recursively deserialising ``sub_solvers``.

        Parameters
        ----------
        data : dict
            Config dict as produced by :meth:`serialise_solver`.

        Returns
        -------
        :class:`pybamm.BaseSolver`
        """
        data = dict(data)
        solver_type = data.pop("type")
        solver_class = getattr(pybamm, solver_type, None)
        if solver_class is None:
            raise ValueError(
                f"Unknown solver type '{solver_type}'. "
                "Must be a class available on the pybamm module."
            )

        if solver_type == "CompositeSolver":
            sub_solvers_config = data.pop("sub_solvers", None)
            if sub_solvers_config is None:
                raise ValueError(
                    "CompositeSolver config must include a 'sub_solvers' list."
                )
            sub_solvers = [Serialise.deserialise_solver(c) for c in sub_solvers_config]
            return solver_class(sub_solvers)

        return solver_class(**data)


def convert_function_to_symbolic_expression(func, name=None):
    """
    Converts a Python function to a PyBaMM symbolic expression

    Parameters
    ----------
    func : callable
        The Python function to convert

    name : str, optional
        The name of the function to use in the symbolic expression. If not provided,
        the name of the function is used.

    Returns
    -------
    pybamm.Symbol
        The PyBaMM symbolic expression
    """
    # Create symbolic parameters for each input argument
    try:
        func_name = func.get_name()
        func_args = func.get_args()
        # Use the underlying function for evaluation
        func_to_eval = func.func
    except AttributeError:
        try:
            func_name = func.__name__
            func_args = list(inspect.signature(func).parameters)
            func_to_eval = func
        except AttributeError:
            # One more fallback, in case it's a partial
            func_name = func.func.__name__
            func_args = list(inspect.signature(func).parameters)
            func_to_eval = func

    sym_inputs = [pybamm.Parameter(arg) for arg in func_args]

    # Evaluate the function with symbolic inputs to get symbolic expression
    sym_output = func_to_eval(*sym_inputs)

    # Wrap the symbolic expression in an ExpressionFunctionParameter to allow access
    # to the function name and arguments
    name = name or func_name
    return ExpressionFunctionParameter(name, sym_output, func_name, func_args)


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
    if "type" not in json_data:
        raise ValueError(f"Missing 'type' key in JSON data: {json_data}")
    if isinstance(json_data, numbers.Number | list):
        return json_data
    elif json_data["type"] == "Parameter":
        # Convert stored parameters back to PyBaMM Parameter objects
        return pybamm.Parameter(json_data["name"])
    elif json_data["type"] == "InputParameter":
        return pybamm.InputParameter(json_data["name"])
    elif json_data["type"] == "Scalar":
        # Convert stored numerical values back to PyBaMM Scalar objects
        return pybamm.Scalar(json_data["value"])
    elif json_data["type"] == "Interpolant":
        return pybamm.Interpolant(
            [np.array(x) for x in json_data["x"]],
            np.array(json_data["y"]),
            [convert_symbol_from_json(c) for c in json_data["children"]],
            name=json_data["name"],
            interpolator=json_data["interpolator"],
            entries_string=json_data["entries_string"],
        )
    elif json_data["type"] == "FunctionParameter":
        diff_variable = json_data["diff_variable"]
        if diff_variable is not None:
            diff_variable = convert_symbol_from_json(diff_variable)
        # Use the parameter name as print_name to avoid showing
        # 'convert_symbol_from_json' in displays
        return pybamm.FunctionParameter(
            json_data["name"],
            {k: convert_symbol_from_json(v) for k, v in json_data["inputs"].items()},
            diff_variable=diff_variable,
            print_name=json_data["name"],
        )
    elif json_data["type"] == "ExpressionFunctionParameter":
        return ExpressionFunctionParameter(
            json_data["name"],
            convert_symbol_from_json(json_data["children"][0]),
            json_data["func_name"],
            json_data["func_args"],
        )
    elif json_data["type"] == "PrimaryBroadcast":
        child = convert_symbol_from_json(json_data["children"][0])
        domain = json_data["broadcast_domain"]
        return pybamm.PrimaryBroadcast(child, domain)
    elif json_data["type"] == "FullBroadcast":
        child = convert_symbol_from_json(json_data["children"][0])
        domains = json_data["domains"]
        return pybamm.FullBroadcast(child, broadcast_domains=domains)
    elif json_data["type"] == "SecondaryBroadcast":
        child = convert_symbol_from_json(json_data["children"][0])
        domain = json_data["broadcast_domain"]
        return pybamm.SecondaryBroadcast(child, domain)
    elif json_data["type"] == "BoundaryValue":
        child = convert_symbol_from_json(json_data["children"][0])
        side = json_data["side"]
        return pybamm.BoundaryValue(child, side)
    elif json_data["type"] == "Variable":
        bounds_data = json_data.get("bounds")
        if bounds_data is None:
            bounds = (
                pybamm.Scalar(-np.inf),
                pybamm.Scalar(np.inf),
            )
        else:
            bounds = tuple(
                convert_symbol_from_json(b) for b in bounds_data
            )
        return pybamm.Variable(
            json_data["name"],
            domains=json_data["domains"],
            bounds=bounds,
        )
    elif json_data["type"] == "IndefiniteIntegral":
        child = convert_symbol_from_json(json_data["children"][0])
        integration_var_json = json_data["integration_variable"]
        integration_variable = convert_symbol_from_json(integration_var_json)
        if not isinstance(integration_variable, pybamm.SpatialVariable):
            raise TypeError(
                f"Expected SpatialVariable, got {type(integration_variable)}"
            )
        return pybamm.IndefiniteIntegral(child, [integration_variable])
    elif json_data["type"] == "SpatialVariable":
        return pybamm.SpatialVariable(
            json_data["name"],
            coord_sys=json_data.get("coord_sys", "cartesian"),
            domains=json_data.get("domains"),
        )
    elif json_data["type"] == "Time":
        return pybamm.Time()
    elif json_data["type"] == "CoupledVariable":
        return pybamm.CoupledVariable(
            json_data["name"],
            domain=json_data.get("domains", {}).get("primary", None),
        )
    elif json_data["type"] == "Symbol":
        return pybamm.Symbol(
            json_data["name"],
            domains=json_data.get("domains", {}),
        )
    elif "children" in json_data:
        return getattr(pybamm, json_data["type"])(
            *[convert_symbol_from_json(c) for c in json_data["children"]]
        )
    else:
        raise ValueError(f"Unknown symbol type: {json_data['type']}")


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
    if isinstance(symbol, ExpressionFunctionParameter):
        return {
            "type": "ExpressionFunctionParameter",
            "name": symbol.name,
            "children": [convert_symbol_to_json(symbol.child)],
            "func_name": symbol.func_name,
            "func_args": symbol.func_args,
        }
    elif isinstance(symbol, numbers.Number | list):
        return symbol
    elif isinstance(symbol, pybamm.Parameter):
        # Parameters are stored with their type and name
        return {"type": "Parameter", "name": symbol.name}
    elif isinstance(symbol, pybamm.Scalar):
        # Scalar values are stored with their numerical value
        return {"type": "Scalar", "value": symbol.value}
    elif isinstance(symbol, pybamm.SpecificFunction):
        if symbol.__class__ == pybamm.SpecificFunction:
            raise NotImplementedError("SpecificFunction is not supported directly")
        else:
            # Subclasses of SpecificFunction (e.g. Exp, Sin, etc.) can be reconstructed
            # from only the children
            return {
                "type": symbol.__class__.__name__,
                "children": [convert_symbol_to_json(c) for c in symbol.children],
            }
    elif isinstance(symbol, pybamm.PrimaryBroadcast):
        json_dict = {
            "type": "PrimaryBroadcast",
            "children": [convert_symbol_to_json(symbol.child)],
            "broadcast_domain": symbol.broadcast_domain,
        }
        return json_dict
    elif isinstance(symbol, pybamm.IndefiniteIntegral):
        integration_var = (
            symbol.integration_variable[0]
            if isinstance(symbol.integration_variable, list)
            else symbol.integration_variable
        )
        json_dict = {
            "type": "IndefiniteIntegral",
            "children": [convert_symbol_to_json(symbol.child)],
            "integration_variable": convert_symbol_to_json(integration_var),
        }
        return json_dict
    elif isinstance(symbol, pybamm.BoundaryValue):
        json_dict = {
            "type": "BoundaryValue",
            "side": symbol.side,
            "children": [convert_symbol_to_json(symbol.orphans[0])],
        }
        return json_dict
    elif isinstance(symbol, pybamm.SecondaryBroadcast):
        json_dict = {
            "type": "SecondaryBroadcast",
            "children": [convert_symbol_to_json(symbol.child)],
            "broadcast_domain": symbol.broadcast_domain,
        }
        return json_dict
    elif isinstance(symbol, pybamm.FullBroadcast):
        json_dict = {
            "type": "FullBroadcast",
            "children": [convert_symbol_to_json(symbol.child)],
            "domains": symbol.domains,
        }
        return json_dict
    elif isinstance(symbol, pybamm.Interpolant):
        return {
            "type": symbol.__class__.__name__,
            "x": [x.tolist() for x in symbol.x],
            "y": symbol.y.tolist(),
            "children": [convert_symbol_to_json(c) for c in symbol.children],
            "name": symbol.name,
            "interpolator": symbol.interpolator,
            "entries_string": symbol.entries_string,
        }
    elif isinstance(symbol, pybamm.Variable):
        lb, ub = symbol.bounds[0], symbol.bounds[1]
        if (
            isinstance(lb, pybamm.Scalar)
            and isinstance(ub, pybamm.Scalar)
            and np.isinf(lb.value)
            and np.isinf(ub.value)
            and lb.value < 0
            and ub.value > 0
        ):
            bounds_json = None
        else:
            bounds_json = [
                convert_symbol_to_json(lb),
                convert_symbol_to_json(ub),
            ]
        json_dict = {
            "type": "Variable",
            "name": symbol.name,
            "domains": symbol.domains,
            "bounds": bounds_json,
        }
        return json_dict
    elif isinstance(symbol, pybamm.ConcatenationVariable):
        json_dict = {
            "type": "ConcatenationVariable",
            "name": symbol.name,
            "children": [convert_symbol_to_json(child) for child in symbol.children],
        }
        return json_dict
    elif isinstance(symbol, pybamm.Time):
        return {"type": "Time"}
    elif isinstance(symbol, pybamm.FunctionParameter):
        input_names = symbol.input_names
        inputs = {
            input_names[i]: convert_symbol_to_json(symbol.orphans[i])
            for i in range(len(input_names))
        }
        diff_variable = symbol.diff_variable
        if diff_variable is not None:
            diff_variable = convert_symbol_to_json(diff_variable)
        return {
            "type": symbol.__class__.__name__,
            "inputs": inputs,
            "diff_variable": diff_variable,
            "name": symbol.name,
        }
    elif isinstance(symbol, pybamm.Symbol):
        # Generic fallback for other symbols with children
        json_dict = {
            "type": symbol.__class__.__name__,
            "domains": symbol.domains,
            "children": [convert_symbol_to_json(c) for c in symbol.children],
        }
        if hasattr(symbol, "name"):
            json_dict["name"] = symbol.name
        return json_dict
    else:
        raise ValueError(
            f"Error processing '{symbol.name}'. Unknown symbol type: {type(symbol)}"
        )
