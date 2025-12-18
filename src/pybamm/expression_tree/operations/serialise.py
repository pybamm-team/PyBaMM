from __future__ import annotations

import importlib
import inspect
import json
import numbers
import re
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path

import black
import numpy as np

import pybamm

SUPPORTED_SCHEMA_VERSION = "1.1"

# Module-level caches for memoization during serialization/deserialization
_serialized_symbols = {}  # Maps symbol id -> (reference_id, JSON representation)
_serialized_ref_counter = 0  # Counter for generating unique reference IDs
_deserialized_symbols = {}  # Maps reference_id -> deserialized symbol


def _reset_serialization_caches():
    """Reset the serialization and deserialization caches.
    Useful for testing or when serializing multiple independent models.
    """
    global _serialized_symbols, _serialized_ref_counter, _deserialized_symbols
    _serialized_symbols = {}
    _serialized_ref_counter = 0
    _deserialized_symbols = {}


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
        # Reset caches for a clean serialization
        _reset_serialization_caches()
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
                    # Store just the enum name as a string
                    "event_type": event.event_type.name,
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

        # Validate schema version (accept 1.0 for backward compatibility)
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version not in ["1.0", SUPPORTED_SCHEMA_VERSION]:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: 1.0 or {SUPPORTED_SCHEMA_VERSION}"
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
            spatial_methods_dict[domain] = {
                "class": type(method).__name__,
                "module": type(method).__module__,
                "options": method.options if hasattr(method, "options") else {},
            }

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

        # Validate schema version (accept 1.0 for backward compatibility)
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version not in ["1.0", SUPPORTED_SCHEMA_VERSION]:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: 1.0 or {SUPPORTED_SCHEMA_VERSION}"
            )

        # Extract spatial methods data
        spatial_methods_data = data.get("spatial_methods")
        if spatial_methods_data is None:
            raise KeyError("Missing 'spatial_methods' section in JSON data.")

        # Reconstruct spatial methods
        reconstructed_methods = {}
        for domain, method_info in spatial_methods_data.items():
            try:
                module_name = method_info["module"]
                class_name = method_info["class"]
                options = method_info.get("options", {})

                # Import module and get class
                module = importlib.import_module(module_name)
                method_class = getattr(module, class_name)

                # Instantiate with options
                reconstructed_methods[domain] = method_class(options=options)

            except (ModuleNotFoundError, AttributeError) as e:
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

        # Validate schema version (accept 1.0 for backward compatibility)
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version not in ["1.0", SUPPORTED_SCHEMA_VERSION]:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: 1.0 or {SUPPORTED_SCHEMA_VERSION}"
            )

        # Extract var_pts data
        var_pts_data = data.get("var_pts")
        if var_pts_data is None:
            raise KeyError("Missing 'var_pts' section in JSON data.")

        return var_pts_data

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
            # Handle MeshGenerator wrapper objects
            if hasattr(submesh_item, "submesh_type"):
                submesh_class = submesh_item.submesh_type
            else:
                submesh_class = submesh_item

            submesh_types_dict[domain] = {
                "class": submesh_class.__name__,
                "module": submesh_class.__module__,
            }

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

        # Validate schema version (accept 1.0 for backward compatibility)
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version not in ["1.0", SUPPORTED_SCHEMA_VERSION]:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: 1.0 or {SUPPORTED_SCHEMA_VERSION}"
            )

        # Extract submesh types data
        submesh_types_data = data.get("submesh_types")
        if submesh_types_data is None:
            raise KeyError("Missing 'submesh_types' section in JSON data.")

        # Reconstruct submesh types
        reconstructed_submesh_types = {}
        for domain, submesh_info in submesh_types_data.items():
            try:
                module_name = submesh_info["module"]
                class_name = submesh_info["class"]

                # Import module and get class
                module = importlib.import_module(module_name)
                submesh_class = getattr(module, class_name)

                # Wrap in MeshGenerator to match the expected format
                reconstructed_submesh_types[domain] = pybamm.MeshGenerator(
                    submesh_class
                )

            except (ModuleNotFoundError, AttributeError) as e:
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

        Parameters
        ----------
        filename : str or dict
            Path to the JSON file containing the saved model, or a dictionary
            containing the serialised model data.

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
        # Reset caches for a clean deserialization
        _reset_serialization_caches()

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

        # Validate outer structure (accept 1.0 for backward compatibility)
        schema_version = data.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if schema_version not in ["1.0", SUPPORTED_SCHEMA_VERSION]:
            raise ValueError(
                f"Unsupported schema version: {schema_version}. "
                f"Expected: 1.0 or {SUPPORTED_SCHEMA_VERSION}"
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
        # Restore model options if present
        # Convert lists back to tuples (JSON converts tuples to lists)
        if "options" in model_data:
            options_dict = model_data["options"]
            # Recursively convert lists to tuples for options that expect tuples
            # (e.g., SEI option can be a 2-tuple for different electrode behavior)
            if isinstance(options_dict, dict):
                converted_options = {}
                for key, value in options_dict.items():
                    if isinstance(value, list):
                        # Convert list to tuple (for options like SEI that can be tuples)
                        converted_options[key] = tuple(value)
                    else:
                        converted_options[key] = value
                model.options = converted_options
            else:
                model.options = options_dict

        # Pre-populate cache by deserializing all unique symbols
        # This ensures references are resolved when building the model structure
        # Collect all JSON symbol definitions (not pure references)
        all_symbols = []
        for _, rhs_json in model_data["rhs"]:
            all_symbols.append(rhs_json)
        for _, alg_json in model_data["algebraic"]:
            all_symbols.append(alg_json)
        for _, ic_json in model_data["initial_conditions"]:
            all_symbols.append(ic_json)
        for variable_json, bc_dict in model_data["boundary_conditions"]:
            for side, bc_value in bc_dict.items():
                try:
                    expr_json, _ = bc_value
                    all_symbols.append(expr_json)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Failed to convert boundary condition for variable {variable_json} "
                        f"on side '{side}': {e!s}"
                    ) from e
        for event_data in model_data["events"]:
            all_symbols.append(event_data["expression"])
        for var_json in model_data["variables"].values():
            all_symbols.append(var_json)

        # Also collect LHS variable definitions
        all_variable_keys = (
            [lhs_json for lhs_json, _ in model_data["rhs"]]
            + [lhs_json for lhs_json, _ in model_data["initial_conditions"]]
            + [lhs_json for lhs_json, _ in model_data["algebraic"]]
            + [variable_json for variable_json, _ in model_data["boundary_conditions"]]
        )
        all_symbols.extend(all_variable_keys)

        # Deserialize all symbols to populate cache (ignore pure references)
        # Do multiple passes until no new symbols are cached (handles forward references)
        max_passes = 3
        for _ in range(max_passes):
            newly_cached = 0
            for symbol_json in all_symbols:
                if isinstance(symbol_json, dict):
                    # Skip pure references
                    if len(symbol_json) == 1 and (
                        "py/ref" in symbol_json or "r" in symbol_json
                    ):
                        continue
                    # Check if already cached
                    ref_id = symbol_json.get("py/ref") or symbol_json.get("r")
                    if ref_id is not None and ref_id in _deserialized_symbols:
                        continue
                    # Deserialize to populate cache
                    try:
                        convert_symbol_from_json(symbol_json)
                        newly_cached += 1
                    except Exception:
                        # If it fails due to forward reference, it will work on next pass
                        pass
            # If nothing new was cached, we're done
            if newly_cached == 0:
                break

        # Build symbol_map for LHS lookups
        symbol_map = {}
        for variable_json in all_variable_keys:
            # Skip pure references
            if isinstance(variable_json, dict):
                if len(variable_json) == 1 and (
                    "py/ref" in variable_json or "r" in variable_json
                ):
                    continue
            try:
                # Should now work since cache is populated
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
                lhs_key = Serialise._create_symbol_key(lhs_json)
                # Check if it's in symbol_map, otherwise deserialize from cache
                if lhs_key in symbol_map:
                    lhs = symbol_map[lhs_key]
                else:
                    lhs = convert_symbol_from_json(lhs_json)
                rhs = convert_symbol_from_json(rhs_expr_json)
                model.rhs[lhs] = rhs
            except Exception as e:
                raise ValueError(
                    f"Failed to convert rhs entry for {lhs_json}: {e!s}"
                ) from e

        model.algebraic = {}
        for lhs_json, algebraic_expr_json in model_data["algebraic"]:
            try:
                lhs_key = Serialise._create_symbol_key(lhs_json)
                if lhs_key in symbol_map:
                    lhs = symbol_map[lhs_key]
                else:
                    lhs = convert_symbol_from_json(lhs_json)
                rhs = convert_symbol_from_json(algebraic_expr_json)
                model.algebraic[lhs] = rhs
            except Exception as e:
                raise ValueError(
                    f"Failed to convert algebraic entry for {lhs_json}: {e!s}"
                ) from e

        model.initial_conditions = {}
        for lhs_json, initial_value_json in model_data["initial_conditions"]:
            try:
                lhs_key = Serialise._create_symbol_key(lhs_json)
                if lhs_key in symbol_map:
                    lhs = symbol_map[lhs_key]
                else:
                    lhs = convert_symbol_from_json(lhs_json)
                rhs = convert_symbol_from_json(initial_value_json)
                model.initial_conditions[lhs] = rhs
            except Exception as e:
                raise ValueError(
                    f"Failed to convert initial condition entry for {lhs_json}: {e!s}"
                ) from e

        model.boundary_conditions = {}
        for variable_json, condition_dict in model_data["boundary_conditions"]:
            try:
                var_key = Serialise._create_symbol_key(variable_json)
                if var_key in symbol_map:
                    variable = symbol_map[var_key]
                else:
                    variable = convert_symbol_from_json(variable_json)
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
                # Convert event_type from string to EventType enum
                event_type_str = event_data.get("event_type", "TERMINATION")
                event_type = pybamm.EventType[event_type_str]
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

        # Ensure the model is in a clean, unprocessed state
        # Reset any attributes that might interfere with processing
        if hasattr(model, "_processed"):
            model._processed = False
        if hasattr(model, "_built"):
            model._built = False
        # Clear any cached geometry or mesh
        if hasattr(model, "_geometry"):
            model._geometry = None
        if hasattr(model, "_mesh"):
            model._mesh = None
        if hasattr(model, "_disc"):
            model._disc = None
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
    # Handle non-dict types
    if isinstance(json_data, float | int | bool | numbers.Number | list):
        return json_data

    if isinstance(json_data, str):
        raise ValueError(f"Unexpected raw string in JSON: {json_data}")

    if json_data is None:
        return None

    # Check for reference first (handle both abbreviated "r" and full "py/ref")
    if isinstance(json_data, dict):
        # Check for pure reference (either "r" or "py/ref" as only key)
        if len(json_data) == 1:
            ref_id = None
            if "r" in json_data:
                ref_id = json_data["r"]
            elif "py/ref" in json_data:
                ref_id = json_data["py/ref"]

            if ref_id is not None:
                if ref_id in _deserialized_symbols:
                    return _deserialized_symbols[ref_id]
                else:
                    # Reference seen before definition - this shouldn't happen in normal flow
                    # but if it does, raise an error that will be caught and retried
                    raise ValueError(
                        f"Reference {ref_id} encountered before its definition. "
                        "This may indicate the symbol needs to be deserialized first."
                    )

    if not isinstance(json_data, dict):
        raise ValueError(f"Expected dict, got {type(json_data)}: {json_data}")

    # Check for type key (handles both "type" and abbreviated "t")
    if "type" not in json_data and "t" not in json_data:
        raise ValueError(f"Missing 'type' key in JSON data: {json_data}")

    # Check cache - prefer ref_id if available (faster lookup, no key computation)
    ref_id = json_data.get("py/ref") or json_data.get("r")
    if ref_id is not None and ref_id in _deserialized_symbols:
        return _deserialized_symbols[ref_id]

    # Deserialize the symbol
    symbol = _deserialize_symbol_from_json(json_data)

    # Store in cache using ref_id (if available)
    if ref_id is not None:
        _deserialized_symbols[ref_id] = symbol

    return symbol


def _deserialize_symbol_from_json(json_data):
    """Internal helper to deserialize a symbol without caching.
    The caching is handled by convert_symbol_from_json.
    """
    # Get type (handle both "type" and "t" keys)
    type_name = json_data.get("type") or json_data.get("t")
    if type_name is None:
        raise ValueError(f"Missing 'type' key in JSON data: {json_data}")
    # Expand type abbreviation if present
    type_name = _TYPE_EXPANSIONS.get(type_name, type_name)

    # Helper to expand domains (handle both "domains" and "d" keys)
    def get_domains():
        domains_data = json_data.get("domains") or json_data.get("d")
        if domains_data is not None:
            return _expand_domains(domains_data)
        return {"primary": [], "secondary": [], "tertiary": [], "quaternary": []}

    if type_name == "Parameter":
        # Convert stored parameters back to PyBaMM Parameter objects
        return pybamm.Parameter(json_data.get("name") or json_data.get("n"))
    elif type_name == "Scalar":
        return pybamm.Parameter(json_data["name"])
    elif json_data["type"] == "InputParameter":
        return pybamm.InputParameter(json_data["name"])
    elif json_data["type"] == "Scalar":
        # Convert stored numerical values back to PyBaMM Scalar objects
        # Use explicit check to handle 0 correctly (can't use 'or' since 0 is falsy)
        if "value" in json_data:
            value = json_data["value"]
        elif "v" in json_data:
            value = json_data["v"]
        else:
            value = 0  # Default to 0 if not found
        # Handle infinity strings
        if value == "Inf":
            value = float("inf")
        elif value == "-Inf":
            value = float("-inf")
        return pybamm.Scalar(value)

    # Helper to get value with fallback for abbreviated keys
    def get_key(key, abbrev=None, default=None):
        if abbrev is None:
            abbrev = _KEY_ABBREVIATIONS.get(key, key)
        if key in json_data:
            return json_data[key]
        if abbrev in json_data:
            return json_data[abbrev]
        return default

    # Helper to get children (handle both "children" and "c")
    def get_children():
        if "children" in json_data:
            return json_data["children"]
        if "c" in json_data:
            return json_data["c"]
        return []

    if type_name == "Interpolant":
        return pybamm.Interpolant(
            [np.array(x) for x in json_data["x"]],
            np.array(json_data["y"]),
            [convert_symbol_from_json(c) for c in get_children()],
            name=get_key("name", "n"),
            interpolator=get_key("interpolator", "i"),
            entries_string=get_key("entries_string", "es"),
        )
    elif type_name == "FunctionParameter":
        diff_variable = get_key("diff_variable", "dv")
        if diff_variable is not None:
            diff_variable = convert_symbol_from_json(diff_variable)
        inputs_key = get_key("inputs", "in", {})
        return pybamm.FunctionParameter(
            get_key("name", "n"),
            {k: convert_symbol_from_json(v) for k, v in inputs_key.items()},
            diff_variable=diff_variable,
            print_name=get_key("name", "n"),
        )
    elif type_name == "ExpressionFunctionParameter":
        children = get_children()
        return ExpressionFunctionParameter(
            get_key("name", "n"),
            convert_symbol_from_json(children[0]) if children else None,
            get_key("func_name", "fn"),
            get_key("func_args", "fa"),
        )
    elif type_name == "PrimaryBroadcast":
        children = get_children()
        domain = get_key("broadcast_domain", "bd")
        return pybamm.PrimaryBroadcast(
            convert_symbol_from_json(children[0]) if children else None, domain
        )
    elif type_name == "FullBroadcast":
        children = get_children()
        domains = _expand_domains(get_key("domains", "d"))
        return pybamm.FullBroadcast(
            convert_symbol_from_json(children[0]) if children else None,
            broadcast_domains=domains,
        )
    elif type_name == "SecondaryBroadcast":
        children = get_children()
        domain = get_key("broadcast_domain", "bd")
        return pybamm.SecondaryBroadcast(
            convert_symbol_from_json(children[0]) if children else None, domain
        )
    elif type_name == "BoundaryValue":
        children = get_children()
        side = get_key("side", "s")
        return pybamm.BoundaryValue(
            convert_symbol_from_json(children[0]) if children else None, side
        )
    elif type_name == "Variable":
        bounds_data = get_key("bounds", "b", [-float("inf"), float("inf")])
        bounds = tuple(convert_symbol_from_json(b) for b in bounds_data)
        return pybamm.Variable(
            get_key("name", "n"),
            domains=get_domains(),
            bounds=bounds,
        )
    elif type_name == "IndefiniteIntegral":
        children = get_children()
        integration_var_json = get_key("integration_variable", "iv")
        integration_variable = convert_symbol_from_json(integration_var_json)
        if not isinstance(integration_variable, pybamm.SpatialVariable):
            raise TypeError(
                f"Expected SpatialVariable, got {type(integration_variable)}"
            )
        return pybamm.IndefiniteIntegral(
            convert_symbol_from_json(children[0]) if children else None,
            [integration_variable],
        )
    elif type_name == "SpatialVariable":
        return pybamm.SpatialVariable(
            get_key("name", "n"),
            coord_sys=get_key("coord_sys", "cs", "cartesian"),
            domains=_expand_domains(get_key("domains", "d")),
        )
    elif type_name == "Time":
        return pybamm.Time()
    elif type_name == "Symbol":
        return pybamm.Symbol(
            get_key("name", "n"),
            domains=get_domains(),
        )
    elif type_name == "ConcatenationVariable":
        children = get_children()
        # Convert children to symbols, ensuring they're all Symbols
        deserialized_children = []
        for i, child_json in enumerate(children):
            if isinstance(child_json, str):
                # If child is a string, it might be a reference or name - this shouldn't happen
                raise ValueError(
                    f"ConcatenationVariable child [{i}] is a string '{child_json}' "
                    f"instead of a symbol dict. This may indicate a serialization issue."
                )
            try:
                child_symbol = convert_symbol_from_json(child_json)
            except Exception as e:
                raise ValueError(
                    f"Failed to deserialize ConcatenationVariable child [{i}]: {e!s}. "
                    f"Child JSON: {child_json}"
                ) from e
            if not isinstance(child_symbol, pybamm.Symbol):
                raise ValueError(
                    f"ConcatenationVariable child [{i}] deserialized to {type(child_symbol).__name__} "
                    f"instead of a Symbol. Got: {child_symbol} (value: {child_symbol!r})"
                )
            deserialized_children.append(child_symbol)
        # ConcatenationVariable automatically derives its name from children
        # Only pass name if it was explicitly stored and is different
        return pybamm.ConcatenationVariable(*deserialized_children)
    elif "children" in json_data or "c" in json_data:
        # Use expanded type name for getattr
        return getattr(pybamm, type_name)(
            *[convert_symbol_from_json(c) for c in get_children()]
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
    # Handle non-symbol types (numbers, lists) - these don't need memoization
    if isinstance(symbol, numbers.Number | list):
        return symbol

    # Check cache first for memoization (only for Symbol types)
    if isinstance(symbol, pybamm.Symbol):
        symbol_id = id(symbol)
        if symbol_id in _serialized_symbols:
            # Return a reference to the already-serialized symbol
            ref_id, _ = _serialized_symbols[symbol_id]
            return {"py/ref": ref_id}

    # Serialize the symbol
    json_dict = _serialize_symbol_to_json(symbol)

    # Store in cache if it's a Symbol type
    if isinstance(symbol, pybamm.Symbol):
        global _serialized_ref_counter
        symbol_id = id(symbol)
        ref_id = _serialized_ref_counter
        _serialized_ref_counter += 1
        # Store both ref_id and full JSON in cache (before compaction for cache lookup)
        _serialized_symbols[symbol_id] = (ref_id, json_dict)
        # Include ref_id in the JSON so we can resolve it during deserialization
        json_dict["py/ref"] = ref_id

    # Compact the JSON dict (abbreviate keys, omit nulls, etc.)
    return _compact_json_dict(json_dict)


# Type name abbreviations to reduce JSON size
# Use mathematical symbols where appropriate for maximum compression
_TYPE_ABBREVIATIONS = {
    # Binary operators - use symbols
    "Multiplication": "*",
    "Division": "/",
    "Addition": "+",
    "Subtraction": "-",
    "Power": "**",
    "Modulo": "%",
    "MatrixMultiplication": "@",
    "Equality": "==",
    "Minimum": "min",
    "Maximum": "max",
    # Unary operators - use short names
    "Negate": "neg",  # Use "neg" to avoid conflict with binary "-"
    "AbsoluteValue": "abs",
    "Transpose": "T",
    "Sign": "sign",
    "Floor": "floor",
    "Ceiling": "ceil",
    # Other operators
    "PrimaryBroadcast": "PBroad",
    "SecondaryBroadcast": "SBroad",
    "FullBroadcast": "FBroad",
    "IndefiniteIntegral": "Int",
    "BoundaryValue": "BVal",
    "ConcatenationVariable": "ConcatVar",
    "ExpressionFunctionParameter": "ExprFP",
    "FunctionParameter": "FP",
}

# Reverse mapping for deserialization
_TYPE_EXPANSIONS = {v: k for k, v in _TYPE_ABBREVIATIONS.items()}

# Key abbreviations to reduce JSON size
_KEY_ABBREVIATIONS = {
    "type": "t",
    "children": "c",
    "domains": "d",
    "name": "n",
    "value": "v",
    "broadcast_domain": "bd",
    "integration_variable": "iv",
    "side": "s",
    "inputs": "in",
    "diff_variable": "dv",
    "func_name": "fn",
    "func_args": "fa",
    "bounds": "b",
    "coord_sys": "cs",
    "interpolator": "i",
    "entries_string": "es",
    "py/ref": "r",  # Keep short already
}

# Reverse mapping for deserialization
_KEY_EXPANSIONS = {v: k for k, v in _KEY_ABBREVIATIONS.items()}


def _compact_json_dict(d):
    """Compact a JSON dictionary by:
    1. Using abbreviated keys
    2. Omitting null values
    3. Omitting empty arrays
    4. Omitting redundant names (when name == type or name == '*')
    5. Recursively compacting nested structures
    """
    if isinstance(d, dict):
        compact = {}
        type_val = d.get("type") or d.get("t")
        name_val = d.get("name") or d.get("n")

        for key, value in d.items():
            # Skip null values
            if value is None:
                continue

            # Skip empty arrays
            if isinstance(value, list) and len(value) == 0:
                continue

            # Skip redundant names
            if key == "name" and (name_val == type_val or name_val == "*"):
                continue

            # Recursively compact nested structures
            if isinstance(value, dict):
                value = _compact_json_dict(value)
            elif isinstance(value, list):
                value = [
                    _compact_json_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]

            # Use abbreviated key
            abbrev_key = _KEY_ABBREVIATIONS.get(key, key)
            compact[abbrev_key] = value

        return compact
    elif isinstance(d, list):
        return [
            _compact_json_dict(item) if isinstance(item, dict) else item for item in d
        ]
    else:
        return d


def _compact_domains(domains):
    """Compact domain representation by omitting empty domains and using
    shorter format when possible.
    """
    if not domains:
        return None

    # Check if all domains are empty
    if all(not v for v in domains.values()):
        return None

    # Check if only primary domain is non-empty (common case)
    if (
        domains.get("primary")
        and not domains.get("secondary")
        and not domains.get("tertiary")
        and not domains.get("quaternary")
    ):
        return domains["primary"]

    # Return compact dict with only non-empty domains
    compact = {}
    for key, value in domains.items():
        if value:  # Only include non-empty domains
            compact[key] = value

    # If only one domain is non-empty, return just that value
    if len(compact) == 1:
        return next(iter(compact.values()))

    return compact if compact else None


def _expand_domains(domains_data):
    """Expand compact domain representation back to full format."""
    if domains_data is None:
        return {"primary": [], "secondary": [], "tertiary": [], "quaternary": []}

    # If it's a list, it's the primary domain
    if isinstance(domains_data, list):
        return {
            "primary": domains_data,
            "secondary": [],
            "tertiary": [],
            "quaternary": [],
        }

    # If it's a dict, expand with defaults
    if isinstance(domains_data, dict):
        result = {"primary": [], "secondary": [], "tertiary": [], "quaternary": []}
        result.update(domains_data)
        return result

    # Fallback
    return {"primary": [], "secondary": [], "tertiary": [], "quaternary": []}


def _expand_json_dict(d):
    """Expand abbreviated keys back to full keys for deserialization."""
    if isinstance(d, dict):
        expanded = {}
        for key, value in d.items():
            # Expand key
            full_key = _KEY_EXPANSIONS.get(key, key)

            # Recursively expand nested structures
            if isinstance(value, dict):
                value = _expand_json_dict(value)
            elif isinstance(value, list):
                value = [
                    _expand_json_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]

            expanded[full_key] = value
        return expanded
    elif isinstance(d, list):
        return [
            _expand_json_dict(item) if isinstance(item, dict) else item for item in d
        ]
    else:
        return d


def _serialize_symbol_to_json(symbol):
    """Internal helper to serialize a symbol without caching.
    The caching is handled by convert_symbol_to_json.
    """
    if isinstance(symbol, ExpressionFunctionParameter):
        return {
            "type": "ExpressionFunctionParameter",
            "name": symbol.name,
            "children": [convert_symbol_to_json(symbol.child)],
            "func_name": symbol.func_name,
            "func_args": symbol.func_args,
        }
    elif isinstance(symbol, pybamm.Parameter):
        # Parameters are stored with their type and name
        return {"type": "Parameter", "name": symbol.name}
    elif isinstance(symbol, pybamm.Scalar):
        # Scalar values are stored with their numerical value
        # Use special values for infinity to save space
        value = symbol.value
        if value == float("inf"):
            value = "Inf"
        elif value == float("-inf"):
            value = "-Inf"
        return {"type": "Scalar", "value": value}
    elif isinstance(symbol, pybamm.SpecificFunction):
        if symbol.__class__ == pybamm.SpecificFunction:
            raise NotImplementedError("SpecificFunction is not supported directly")
        else:
            # Subclasses of SpecificFunction (e.g. Exp, Sin, etc.) can be reconstructed
            # from only the children
            type_name = symbol.__class__.__name__
            return {
                "type": _TYPE_ABBREVIATIONS.get(type_name, type_name),
                "children": [convert_symbol_to_json(c) for c in symbol.children],
            }
    elif isinstance(symbol, pybamm.PrimaryBroadcast):
        json_dict = {
            "type": "PBroad",
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
            "type": "Int",
            "children": [convert_symbol_to_json(symbol.child)],
            "integration_variable": convert_symbol_to_json(integration_var),
        }
        return json_dict
    elif isinstance(symbol, pybamm.BoundaryValue):
        json_dict = {
            "type": "BVal",
            "side": symbol.side,
            "children": [convert_symbol_to_json(symbol.orphans[0])],
        }
        return json_dict
    elif isinstance(symbol, pybamm.SecondaryBroadcast):
        json_dict = {
            "type": "SBroad",
            "children": [convert_symbol_to_json(symbol.child)],
            "broadcast_domain": symbol.broadcast_domain,
        }
        return json_dict
    elif isinstance(symbol, pybamm.FullBroadcast):
        json_dict = {
            "type": "FBroad",
            "children": [convert_symbol_to_json(symbol.child)],
            "domains": _compact_domains(symbol.domains),
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
        json_dict = {
            "type": "Variable",
            "name": symbol.name,
            "bounds": [
                convert_symbol_to_json(symbol.bounds[0]),
                convert_symbol_to_json(symbol.bounds[1]),
            ],
        }
        # Only include domains if non-empty
        compact_domains = _compact_domains(symbol.domains)
        if compact_domains is not None:
            json_dict["domains"] = compact_domains
        return json_dict
    elif isinstance(symbol, pybamm.ConcatenationVariable):
        json_dict = {
            "type": "ConcatVar",
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
        type_name = symbol.__class__.__name__
        return {
            "type": _TYPE_ABBREVIATIONS.get(type_name, type_name),
            "inputs": inputs,
            "diff_variable": diff_variable,
            "name": symbol.name,
        }
    elif isinstance(symbol, pybamm.Symbol):
        # Generic fallback for other symbols with children
        type_name = symbol.__class__.__name__
        json_dict = {
            "type": _TYPE_ABBREVIATIONS.get(type_name, type_name),
            "children": [convert_symbol_to_json(c) for c in symbol.children],
        }
        # Only include domains if non-empty
        compact_domains = _compact_domains(symbol.domains)
        if compact_domains is not None:
            json_dict["domains"] = compact_domains
        if hasattr(symbol, "name"):
            json_dict["name"] = symbol.name
        return json_dict
    else:
        raise ValueError(
            f"Error processing '{symbol.name}'. Unknown symbol type: {type(symbol)}"
        )
