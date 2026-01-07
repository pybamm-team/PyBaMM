from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np

import pybamm
from pybamm.expression_tree.operations.serialise import (
    Serialise,
    convert_function_to_symbolic_expression,
    convert_symbol_from_json,
    convert_symbol_to_json,
)
from pybamm.models.full_battery_models.lithium_ion.msmr import (
    is_deprecated_msmr_name,
    replace_deprecated_msmr_name,
)

from .parameter_store import (
    ParameterCategory,
    ParameterDiff,
    ParameterInfo,
    ParameterStore,
)
from .symbol_processor import SymbolProcessor

if TYPE_CHECKING:
    from collections.abc import Mapping


class ParameterValues:
    """
    The parameter values for a simulation.

    Note that this class does not inherit directly from the python dictionary class as
    this causes issues with saving and loading simulations.

    Parameters
    ----------
    values : dict or string or ParameterValues
        Explicit set of parameters, or reference to an inbuilt parameter set.
        If string and matches one of the inbuilt parameter sets, returns that
        parameter set.

    Examples
    --------
    >>> values = {"some parameter": 1, "another parameter": 2}
    >>> param = pybamm.ParameterValues(values)
    >>> param["some parameter"]
    1
    >>> param = pybamm.ParameterValues("Marquis2019")
    >>> param["Reference temperature [K]"]
    298.15

    >>> info = param.get_info("Reference temperature [K]")
    >>> info.units
    'K'
    >>> electrode_params = param.list_by_category("negative electrode")
    >>> diff = param.diff(other_param)
    """

    # Physical constants are deprecated in ParameterValues
    _DEPRECATED_CONSTANTS = {
        "Ideal gas constant [J.K-1.mol-1]": "pybamm.constants.R",
        "Faraday constant [C.mol-1]": "pybamm.constants.F",
        "Boltzmann constant [J.K-1]": "pybamm.constants.k_b",
        "Electron charge [C]": "pybamm.constants.q_e",
    }

    def __init__(self, values: dict[str, Any] | str | ParameterValues) -> None:
        # Initialize the store
        self._store = ParameterStore({})

        # Initialize the processor (uses this instance's store)
        self._processor = SymbolProcessor(self._store)

        if isinstance(values, dict | ParameterValues):
            # Copy to avoid mutating input
            if isinstance(values, ParameterValues):
                values_dict = dict(values._store._data)
            else:
                values_dict = dict(values)
            # Remove the "chemistry" key if it exists
            chemistry = values_dict.pop("chemistry", None)
            self._update_internal(values_dict, check_already_exists=False)
        else:
            # Check if values is a named parameter set
            if isinstance(values, str) and values in pybamm.parameter_sets.keys():
                values_dict = dict(pybamm.parameter_sets[values])
                chemistry = values_dict.pop("chemistry", None)
                self._update_internal(values_dict, check_already_exists=False)
            else:
                valid_sets = "\n".join(pybamm.parameter_sets.keys())
                raise ValueError(
                    f"'{values}' is not a valid parameter set. "
                    f"Parameter set must be one of:\n{valid_sets}"
                )

        if chemistry == "ecm":
            self._set_initial_state = pybamm.equivalent_circuit.set_initial_state
        else:
            self._set_initial_state = pybamm.lithium_ion.set_initial_state

        # Save citations
        if "citations" in self._store:
            for citation in self._store["citations"]:
                pybamm.citations.register(citation)

    # Factory methods
    @classmethod
    def create_from_bpx(
        cls, filename: str | Path, target_soc: float = 1.0
    ) -> ParameterValues:
        """
        Create ParameterValues from a BPX file.

        Parameters
        ----------
        filename : str or Path
            The filename of the `BPX <https://bpxstandard.com/>`_ file.
        target_soc : float, optional
            Target state of charge. Must be between 0 and 1. Default is 1.

        Returns
        -------
        ParameterValues
            A parameter values object with the parameters from the BPX file.

        Examples
        --------
        >>> param = pybamm.ParameterValues.create_from_bpx("battery_params.json")
        >>> param = pybamm.ParameterValues.create_from_bpx("battery_params.json", target_soc=0.5)
        """
        from bpx import parse_bpx_file

        bpx = parse_bpx_file(str(filename))
        return cls._create_from_bpx(bpx, target_soc)

    @classmethod
    def create_from_bpx_obj(
        cls, bpx_obj: dict, target_soc: float = 1.0
    ) -> ParameterValues:
        """
        Create ParameterValues from a BPX dictionary object.

        Parameters
        ----------
        bpx_obj : dict
            A dictionary containing the parameters in the
            `BPX <https://bpxstandard.com/>`_ format.
        target_soc : float, optional
            Target state of charge. Must be between 0 and 1. Default is 1.

        Returns
        -------
        ParameterValues
            A parameter values object with the parameters in the BPX file.

        Examples
        --------
        >>> bpx_dict = {"Header": {...}, "Cell": {...}, "Parameterisation": {...}}
        >>> param = pybamm.ParameterValues.create_from_bpx_obj(bpx_dict)
        >>> param = pybamm.ParameterValues.create_from_bpx_obj(bpx_dict, target_soc=0.8)
        """
        from bpx import parse_bpx_obj

        bpx = parse_bpx_obj(bpx_obj)
        return cls._create_from_bpx(bpx, target_soc)

    @classmethod
    def _create_from_bpx(cls, bpx, target_soc: float) -> ParameterValues:
        """Internal method to create ParameterValues from a parsed BPX object."""
        from bpx import get_electrode_concentrations
        from bpx.schema import ElectrodeBlended, ElectrodeBlendedSPM

        from .bpx import bpx_to_param_dict

        if target_soc < 0 or target_soc > 1:
            raise ValueError("Target SOC should be between 0 and 1")

        pybamm_dict = bpx_to_param_dict(bpx)

        if "Open-circuit voltage at 0% SOC [V]" not in pybamm_dict:
            pybamm_dict["Open-circuit voltage at 0% SOC [V]"] = pybamm_dict[
                "Lower voltage cut-off [V]"
            ]
            warn(
                "'Open-circuit voltage at 0% SOC [V]' not found in BPX file. Using "
                "'Lower voltage cut-off [V]'.",
                stacklevel=2,
            )
        if "Open-circuit voltage at 100% SOC [V]" not in pybamm_dict:
            pybamm_dict["Open-circuit voltage at 100% SOC [V]"] = pybamm_dict[
                "Upper voltage cut-off [V]"
            ]
            warn(
                "'Open-circuit voltage at 100% SOC [V]' not found in BPX file. Using "
                "'Upper voltage cut-off [V]'.",
                stacklevel=2,
            )

        # Get initial concentrations based on SOC
        bpx_neg = bpx.parameterisation.negative_electrode
        bpx_pos = bpx.parameterisation.positive_electrode
        if isinstance(bpx_neg, ElectrodeBlended | ElectrodeBlendedSPM) or isinstance(
            bpx_pos, ElectrodeBlended | ElectrodeBlendedSPM
        ):
            pybamm.logger.warning(
                "Initial concentrations cannot be set using stoichiometry limits for "
                "blend electrodes. Please set the initial concentrations manually."
            )
        else:
            c_n_init, c_p_init = get_electrode_concentrations(target_soc, bpx)
            pybamm_dict["Initial concentration in negative electrode [mol.m-3]"] = (
                c_n_init
            )
            pybamm_dict["Initial concentration in positive electrode [mol.m-3]"] = (
                c_p_init
            )

        return cls(pybamm_dict)

    @staticmethod
    def from_json(filename_or_dict: str | Path | dict) -> ParameterValues:
        """
        Load a ParameterValues object from a JSON file or a dictionary.

        Parameters
        ----------
        filename_or_dict : str, Path, or dict
            The filename to load the JSON file from, or a dictionary.

        Returns
        -------
        ParameterValues
            The ParameterValues object.

        Examples
        --------
        >>> param = pybamm.ParameterValues.from_json("parameters.json")
        >>> param_dict = {"Temperature [K]": 298.15}
        >>> param = pybamm.ParameterValues.from_json(param_dict)
        """
        if isinstance(filename_or_dict, str | Path):
            with open(filename_or_dict) as f:
                parameter_values_dict = json.load(f)
        elif isinstance(filename_or_dict, dict):
            parameter_values_dict = filename_or_dict.copy()
        else:
            raise TypeError("Input must be a filename (str or pathlib.Path) or a dict")

        for key, value in parameter_values_dict.items():
            if isinstance(value, dict):
                parameter_values_dict[key] = convert_symbol_from_json(value)

        return ParameterValues(parameter_values_dict)

    def to_json(self, filename: str | None = None) -> dict:
        """
        Convert the parameter values to a JSON-serializable dictionary.

        Optionally saves to a file.

        Parameters
        ----------
        filename : str, optional
            The filename to save the JSON file to. If not provided, the
            dictionary is not saved.

        Returns
        -------
        dict
            The JSON-serializable dictionary.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> param_dict = param.to_json()  # Get dictionary
        >>> param.to_json("parameters.json")  # Save to file
        """
        return convert_parameter_values_to_json(self, filename)

    # Dictionary-like interface
    def __getitem__(self, key: str) -> Any:
        """Get a parameter value by key."""
        if key in self._DEPRECATED_CONSTANTS:
            try:
                _ = self._store[key]
            except KeyError as e:
                raise KeyError(
                    f"Accessing '{key}' from ParameterValues is deprecated. "
                    f"Use {self._DEPRECATED_CONSTANTS[key]} instead."
                ) from e
        try:
            return self._store[key]
        except KeyError as err:
            if (
                "Exchange-current density for lithium metal electrode [A.m-2]"
                in str(err)
                and "Exchange-current density for plating [A.m-2]" in self._store
            ):
                raise KeyError(
                    "'Exchange-current density for plating [A.m-2]' has been renamed "
                    "to 'Exchange-current density for lithium metal electrode [A.m-2]' "
                    "when referring to the reaction at the surface of a lithium metal "
                    "electrode. This is to avoid confusion with the exchange-current "
                    "density for the lithium plating reaction in a porous negative "
                    "electrode. To avoid this error, change your parameter file to use "
                    "the new name."
                ) from err
            else:
                raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Return item corresponding to key if it exists, otherwise return default.

        Parameters
        ----------
        key : str
            The parameter name to retrieve.
        default : Any, optional
            The default value to return if key is not found. Default is None.

        Returns
        -------
        Any
            The parameter value if found, otherwise the default value.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> param.get("Current function [A]")
        5.0
        >>> param.get("NonExistent Parameter", 42)
        42
        """
        if key in self._DEPRECATED_CONSTANTS:
            warn(
                f"Accessing '{key}' from ParameterValues is deprecated. "
                f"Use {self._DEPRECATED_CONSTANTS[key]} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._store.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a parameter value (allows new parameters)."""
        # Process special string values like "[input]"
        if isinstance(value, str):
            if value == "[input]":
                value = pybamm.InputParameter(key)
            elif (
                value.startswith("[function]")
                or value.startswith("[current data]")
                or value.startswith("[data]")
                or value.startswith("[2D data]")
            ):
                raise ValueError(
                    "Specifying parameters via [function], [current data], [data] "
                    "or [2D data] is no longer supported. For functions, pass in a "
                    "python function object. For data, pass in a python function "
                    "that returns a pybamm Interpolant object. "
                    "See the Ai2020 parameter set for an example with both."
                )
            else:
                # Try to convert to float
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string if not convertible
        self._store[key] = value
        self._processor.clear_cache()

    def __delitem__(self, key: str) -> None:
        """Delete a parameter."""
        del self._store._data[key]
        self._processor.clear_cache()

    def __contains__(self, key: str) -> bool:
        """Check if a parameter exists."""
        return key in self._store

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter keys."""
        return iter(self._store)

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._store)

    def __repr__(self) -> str:
        """Return a string representation."""
        return pformat(dict(self._store._data), width=1)

    def __eq__(self, other: object) -> bool:
        """Check equality with another ParameterValues."""
        if not isinstance(other, ParameterValues):
            return NotImplemented
        return dict(self._store._data) == dict(other._store._data)

    def keys(self):
        """
        Return parameter keys.

        Returns
        -------
        dict_keys
            The parameter keys.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> keys = list(param.keys())
        >>> "Current function [A]" in keys
        True
        """
        return self._store.keys()

    def values(self):
        """
        Return parameter values.

        Returns
        -------
        dict_values
            The parameter values.

        Examples
        --------
        >>> param = pybamm.ParameterValues({"Temperature [K]": 298.15})
        >>> vals = list(param.values())
        >>> 298.15 in vals
        True
        """
        return self._store.values()

    def items(self):
        """
        Return parameter items.

        Returns
        -------
        dict_items
            The parameter items as (key, value) pairs.

        Examples
        --------
        >>> param = pybamm.ParameterValues({"Temperature [K]": 298.15})
        >>> items = list(param.items())
        >>> ("Temperature [K]", 298.15) in items
        True
        """
        return self._store.items()

    def pop(self, *args, **kwargs) -> Any:
        """
        Remove and return a parameter value.

        Example
        -------
        >>> params = pybamm.ParameterValues("Chen2020")
        >>> val = params.pop("Current function [A]")
        """
        result = self._store.pop(*args, **kwargs)
        self._processor.clear_cache()
        return result

    def copy(self) -> ParameterValues:
        """
        Return a copy of the parameter values.

        Example
        -------
        >>> params = pybamm.ParameterValues("Chen2020")
        >>> params_copy = params.copy()
        >>> params_copy["Current function [A]"] = 10.0  # Original unchanged
        """
        new_copy = ParameterValues(dict(self._store._data))
        new_copy._set_initial_state = self._set_initial_state
        return new_copy

    def search(self, key: str, print_values: bool = True) -> None:
        """
        Search dictionary for keys containing 'key'.

        Example
        -------
        >>> params = pybamm.ParameterValues("Chen2020")
        >>> params.search("electrolyte")  # Prints matching parameters
        """
        return self._store.search(key, print_values)

    # Update methods
    def update(
        self,
        values: Mapping[str, Any],
        *,
        check_conflict: bool = False,
        check_already_exists: bool = True,
        path: str = "",
    ) -> None:
        """
        Update parameter values.

        Parameters
        ----------
        values : dict
            Dictionary of parameter values to update.
        check_conflict : bool, optional
            Whether to check that a parameter has not already been defined
            with a different value. Default is False.
        check_already_exists : bool, optional
            Deprecated. Use `set` method instead.
        path : str, optional
            Path from which to load functions (legacy parameter).

        Example
        -------
        >>> params = pybamm.ParameterValues("Chen2020")
        >>> params.update({"Current function [A]": 2.0})  # Update existing

        Notes
        -----
        The `check_already_exists` parameter is deprecated. Use `set`
        to add new parameters.
        """
        # Handle deprecated parameter
        if check_already_exists is not None:
            if not check_already_exists:
                warn(
                    "Passing check_already_exists=False is deprecated. "
                    "Use param.set(values) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                check_already_exists = False

        self._update_internal(
            values,
            check_conflict=check_conflict,
            check_already_exists=check_already_exists,
        )

    def set(self, values: Mapping[str, Any], path: str = "") -> None:
        """
        Set parameter values without existence checks.

        This method allows adding new parameters or updating existing ones
        without raising an error. Use this when adding custom parameters
        that are not part of the base parameter set.

        Parameters
        ----------
        values : dict
            Dictionary of parameter values to set.
        path : str, optional
            Path from which to load functions (legacy parameter).

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> param.set({"My custom parameter": 42})
        """
        self._update_internal(values, check_conflict=False, check_already_exists=False)

    def _update_internal(
        self,
        values: Mapping[str, Any],
        check_conflict: bool = False,
        check_already_exists: bool = True,
    ) -> None:
        """
        Internal update method that performs the actual parameter update logic.
        """
        # Convert ParameterValues to dict
        if isinstance(values, ParameterValues):
            values = dict(values._store._data)

        # Check and transform parameter values
        values = self.check_parameter_values(dict(values))

        for name, value in values.items():
            # Check for conflicts
            if (
                check_conflict
                and name in self._store
                and not (
                    self._store[name] == float(value)
                    if isinstance(value, (int, float))
                    else self._store[name] == value
                )
            ):
                raise ValueError(
                    f"parameter '{name}' already defined with value '{self._store[name]}'"
                )

            # Check parameter already exists
            if check_already_exists and name not in self._store:
                try:
                    # Raises a KeyError and catch it
                    _ = self._store[name]
                except KeyError as err:
                    raise KeyError(
                        f"Cannot update parameter '{name}' as it does not "
                        f"have a default value. ({err.args[0] if err.args else ''}). "
                        "Use param.set({name: value}) to add new parameters."
                    ) from err

            # Process value
            if isinstance(value, str):
                if (
                    value.startswith("[function]")
                    or value.startswith("[current data]")
                    or value.startswith("[data]")
                    or value.startswith("[2D data]")
                ):
                    raise ValueError(
                        "Specifying parameters via [function], [current data], [data] "
                        "or [2D data] is no longer supported. For functions, pass in a "
                        "python function object. For data, pass in a python function "
                        "that returns a pybamm Interpolant object. "
                        "See the Ai2020 parameter set for an example with both."
                    )
                elif value == "[input]":
                    self._store[name] = pybamm.InputParameter(name)
                else:
                    # Convert to float
                    self._store[name] = float(value)
            elif isinstance(value, tuple) and isinstance(value[1], np.ndarray):
                # If data is provided as a 2-column array (1D data),
                # convert to two arrays for compatibility with 2D data
                func_name, data = value
                data = ([data[:, 0]], data[:, 1])
                self._store[name] = (func_name, data)
            else:
                self._store[name] = value

        # Clear processor cache
        self._processor.clear_cache()

    @staticmethod
    def check_parameter_values(values: dict) -> dict:
        """
        Check and transform parameter values.

        Parameters
        ----------
        values : dict
            Dictionary of parameter values to check.

        Returns
        -------
        dict
            Checked and transformed parameter values.

        Raises
        ------
        ValueError
            If parameter names contain deprecated or invalid formats.

        Examples
        --------
        >>> values = {"Electrode height [m]": 0.065}
        >>> checked = pybamm.ParameterValues.check_parameter_values(values)
        """
        values = scalarize_dict(values)
        for param in list(values.keys()):
            if "propotional term" in param:
                raise ValueError(
                    f"The parameter '{param}' has been renamed to "
                    "'... proportional term [s-1]', and its value should now be divided"
                    "by 3600 to get the same results as before."
                )
            # specific check for renamed parameter "1 + dlnf/dlnc"
            if "1 + dlnf/dlnc" in param:
                raise ValueError(
                    f"parameter '{param}' has been renamed to 'Thermodynamic factor'"
                )
            if "electrode diffusivity" in param:
                new_param = param.replace("electrode", "particle")
                warn(
                    f"The parameter '{param}' has been renamed to '{new_param}'",
                    DeprecationWarning,
                    stacklevel=2,
                )
                values[new_param] = values.get(param)
            if is_deprecated_msmr_name(param):
                new_param = replace_deprecated_msmr_name(param)
                warn(
                    f"The parameter '{param}' has been renamed to '{new_param}'",
                    DeprecationWarning,
                    stacklevel=2,
                )
                values[new_param] = values.get(param)

        return values

    # Initial state methods
    def set_initial_state(
        self,
        initial_value,
        direction=None,
        param=None,
        inplace: bool = True,
        options=None,
        inputs=None,
    ):
        """
        Set the initial state of the battery.

        Delegates to chemistry-specific implementation.

        Parameters
        ----------
        initial_value : float
            The initial state value (e.g., SOC or voltage).
        direction : str, optional
            Direction for setting state. Default is None.
        param : pybamm.ParameterValues, optional
            Parameter values to use. Default is None (uses self).
        inplace : bool, optional
            If True, modify parameters in place. Default is True.
        options : dict, optional
            Model options. Default is None.
        inputs : dict, optional
            Input parameters. Default is None.

        Returns
        -------
        ParameterValues or None
            Updated parameter values if inplace=False, otherwise None.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> param.set_initial_state(0.5)  # Set initial SOC to 50%
        """
        return self._set_initial_state(
            initial_value,
            self,
            direction=direction,
            param=param,
            inplace=inplace,
            options=options,
            inputs=inputs,
        )

    def set_initial_stoichiometry_half_cell(
        self,
        initial_value,
        direction=None,
        param=None,
        known_value="cyclable lithium capacity",
        inplace=True,
        options=None,
        inputs=None,
    ):
        """Deprecated: Use set_initial_state instead."""
        msg = "pybamm.parameter_values.set_initial_stoichiometry_half_cell is deprecated, please use set_initial_state."
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._set_initial_state(
            initial_value,
            self,
            direction=direction,
            param=param,
            known_value=known_value,
            inplace=inplace,
            options=options,
            inputs=inputs,
        )

    def set_initial_stoichiometries(
        self,
        initial_value,
        direction=None,
        param=None,
        known_value="cyclable lithium capacity",
        inplace=True,
        options=None,
        inputs=None,
        tol=1e-6,
    ):
        """Deprecated: Use set_initial_state instead."""
        msg = "pybamm.parameter_values.set_initial_stoichiometries is deprecated, please use set_initial_state."
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._set_initial_state(
            initial_value,
            self,
            direction=direction,
            param=param,
            known_value=known_value,
            inplace=inplace,
            options=options,
            inputs=inputs,
            tol=tol,
        )

    def set_initial_ocps(
        self,
        initial_value,
        direction=None,
        param=None,
        known_value="cyclable lithium capacity",
        inplace=True,
        options=None,
        inputs=None,
    ):
        """Deprecated: Use set_initial_state instead."""
        msg = "pybamm.parameter_values.set_initial_ocps is deprecated, please use set_initial_state."
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._set_initial_state(
            initial_value,
            self,
            direction=direction,
            param=param,
            known_value=known_value,
            inplace=inplace,
            options=options,
            inputs=inputs,
        )

    # Processing methods
    # These are delegated to the SymbolProcessor
    def process_model(
        self,
        unprocessed_model: pybamm.BaseModel,
        inplace: bool = True,
        delayed_variable_processing: bool | None = None,
    ) -> pybamm.BaseModel:
        """
        Assign parameter values to a model.

        Parameters
        ----------
        unprocessed_model : pybamm.BaseModel
            Model to assign parameter values for.
        inplace : bool, optional
            If True (default), replace parameters in place.
            If False, return a new model with parameter values set.
        delayed_variable_processing : bool, optional
            If True, make variable processing a post-processing step.
            Default is False.

        Returns
        -------
        pybamm.BaseModel
            The parameterized model.

        Raises
        ------
        pybamm.ModelError
            If an empty model is passed.

        Example
        -------
        >>> model = pybamm.lithium_ion.SPM()
        >>> params = pybamm.ParameterValues("Chen2020")
        >>> processed_model = params.process_model(model)
        """
        model = self._processor.process_model(
            unprocessed_model,
            inplace=inplace,
            delayed_variable_processing=delayed_variable_processing,
        )
        # Attach parameter_values to the model's symbol_processor
        pybamm.logger.debug(
            "Attaching the `parameter_values` to the `symbol_processor`"
        )
        model.symbol_processor.parameter_values = self
        return model

    def process_geometry(self, geometry: dict) -> None:
        """
        Assign parameter values to a geometry (inplace).

        Parameters
        ----------
        geometry : dict
            Geometry specs to assign parameter values to.

        Examples
        --------
        >>> geometry = pybamm.battery_geometry()
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> param.process_geometry(geometry)
        """
        self._processor.process_geometry(geometry)

    def process_symbol(self, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """
        Walk through the symbol and replace any Parameter with a Value.

        If a symbol has already been processed, the cached value is returned.

        Parameters
        ----------
        symbol : pybamm.Symbol
            Symbol or Expression tree to set parameters for.

        Returns
        -------
        pybamm.Symbol
            Symbol with Parameter instances replaced by values.

        Example
        -------
        >>> param = pybamm.Parameter("Current function [A]")
        >>> processed = params.process_symbol(param)
        >>> processed.evaluate()
        5.0
        """
        return self._processor.process_symbol(symbol)

    def process_boundary_conditions(self, model: pybamm.BaseModel) -> dict:
        """
        Process boundary conditions for a model.

        Boundary conditions are dictionaries {"left": left bc, "right": right bc}
        in general, but may be imposed on the tabs for some variables.

        Parameters
        ----------
        model : pybamm.BaseModel
            Model whose boundary conditions to process.

        Returns
        -------
        dict
            Processed boundary conditions.

        Examples
        --------
        >>> model = pybamm.lithium_ion.SPM()
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> bcs = param.process_boundary_conditions(model)
        """
        return self._processor.process_boundary_conditions(model)

    def evaluate(self, symbol: pybamm.Symbol, inputs: dict | None = None) -> Any:
        """
        Process and evaluate a symbol.

        Parameters
        ----------
        symbol : pybamm.Symbol
            Symbol or Expression tree to evaluate.
        inputs : dict, optional
            Input parameter values for evaluation.

        Returns
        -------
        number or array
            The evaluated symbol.

        Example
        -------
        >>> param = pybamm.Parameter("Current function [A]")
        >>> params.evaluate(param)
        5.0
        """
        return self._processor.evaluate(symbol, inputs)

    # Metatdata methods
    def get_info(self, key: str) -> ParameterInfo:
        """
        Get metadata about a parameter.

        Parameters
        ----------
        key : str
            The parameter name.

        Returns
        -------
        ParameterInfo
            Metadata including value, units, category, and type information.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> info = param.get_info("Maximum concentration in negative electrode [mol.m-3]")
        >>> info.units
        'mol.m-3'
        >>> info.category
        'negative electrode'
        """
        return self._store.get_info(key)

    def list_by_category(self, category: ParameterCategory | str) -> list[str]:
        """
        Return all parameter names in a given category.

        Parameters
        ----------
        category : ParameterCategory or str
            The category to filter by. Can be a ParameterCategory enum value
            or a string like "negative electrode".

        Returns
        -------
        list[str]
            List of parameter names in the category.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> electrode_params = param.list_by_category("negative electrode")
        >>> len(electrode_params) > 0
        True
        """
        return self._store.list_by_category(category)

    def categories(self) -> dict[str, list[str]]:
        """
        Return all parameters grouped by category.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping category names to lists of parameter names.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> cats = param.categories()
        >>> "negative electrode" in cats
        True
        """
        return self._store.categories()

    def diff(self, other: ParameterValues, *, rtol: float = 0.0) -> ParameterDiff:
        """
        Compare this ParameterValues with another and return differences.

        Parameters
        ----------
        other : ParameterValues
            The other parameter values to compare against.
        rtol : float, optional
            Relative tolerance for numerical comparisons. Differences smaller
            than ``rtol * max(|a|, |b|)`` are ignored. Default is 0.0 (exact
            comparison). Set to e.g. 1e-6 to ignore tiny floating-point
            differences.

        Returns
        -------
        ParameterDiff
            Object containing added, removed, and changed parameters.

        Examples
        --------
        >>> chen = pybamm.ParameterValues("Chen2020")
        >>> marquis = pybamm.ParameterValues("Marquis2019")
        >>> diff = chen.diff(marquis)
        >>> isinstance(diff.changed, dict)
        True

        With tolerance:

        >>> params1 = pybamm.ParameterValues({"x": 1.0})
        >>> params2 = pybamm.ParameterValues({"x": 1.0 + 1e-10})
        >>> params1.diff(params2, rtol=1e-9).changed  # Empty
        {}
        """
        return self._store.diff(other._store, rtol=rtol)

    # Utility methods
    def _ipython_key_completions_(self) -> list[str]:
        """Provide key completions for IPython."""
        return list(self._store.keys())

    def print_parameters(self, parameters, output_file: str | None = None) -> dict:
        """
        Return dictionary of evaluated parameters.

        Optionally print these evaluated parameters to an output file.

        Parameters
        ----------
        parameters : class or dict containing pybamm.Parameter objects
            Class or dictionary containing all the parameters to be evaluated.
        output_file : str, optional
            The file to print parameters to. If None, the parameters are not
            printed, and this function simply acts as a test that all the
            parameters can be evaluated.

        Returns
        -------
        dict
            The evaluated parameters.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> params_dict = {"param1": pybamm.Parameter("Current function [A]")}
        >>> evaluated = param.print_parameters(params_dict)
        >>> param.print_parameters(params_dict, "output.txt")  # Save to file
        {}
        """
        # Set list of attributes to ignore
        ignore = [
            "__name__",
            "__doc__",
            "__package__",
            "__loader__",
            "__spec__",
            "__file__",
            "__cached__",
            "__builtins__",
            "absolute_import",
            "division",
            "print_function",
            "unicode_literals",
            "pybamm",
            "_options",
            "constants",
            "np",
            "geo",
            "elec",
            "therm",
            "half_cell",
            "x",
            "r",
        ]

        # If 'parameters' is a class, extract the dict
        if not isinstance(parameters, dict):
            parameters_dict = {
                k: v for k, v in parameters.__dict__.items() if k not in ignore
            }
            for domain in ["n", "s", "p"]:
                domain_param = getattr(parameters, domain)
                parameters_dict.update(
                    {
                        f"{domain}.{k}": v
                        for k, v in domain_param.__dict__.items()
                        if k not in ignore
                    }
                )
            parameters = parameters_dict

        evaluated_parameters = defaultdict(list)

        for name, symbol in parameters.items():
            if isinstance(symbol, pybamm.Symbol):
                try:
                    proc_symbol = self.process_symbol(symbol)
                except KeyError:
                    # skip parameters that don't have a value in that parameter set
                    proc_symbol = None
                if not (
                    callable(proc_symbol)
                    or proc_symbol is None
                    or proc_symbol.has_symbol_of_classes(
                        (pybamm.Concatenation, pybamm.Broadcast)
                    )
                ):
                    evaluated_parameters[name] = proc_symbol.evaluate(t=0)

        # Print the evaluated_parameters dict to output_file
        if output_file:
            self.print_evaluated_parameters(evaluated_parameters, output_file)

        return evaluated_parameters

    def print_evaluated_parameters(
        self, evaluated_parameters: dict, output_file: str
    ) -> None:
        """
        Print a dictionary of evaluated parameters to an output file.

        Parameters
        ----------
        evaluated_parameters : dict
            The evaluated parameters.
        output_file : str
            The file to print parameters to.

        Examples
        --------
        >>> param = pybamm.ParameterValues("Chen2020")
        >>> evaluated_params = {"Temperature [K]": 298.15, "Voltage [V]": 3.7}
        >>> param.print_evaluated_parameters(evaluated_params, "params.txt")
        """
        # Get column width for pretty printing
        column_width = max(len(name) for name in evaluated_parameters.keys())
        s = f"{{:>{column_width}}}"
        with open(output_file, "w") as file:
            for name, value in sorted(evaluated_parameters.items()):
                if 0.001 < abs(value) < 1000:
                    file.write((s + " : {:10.4g}\n").format(name, value))
                else:
                    file.write((s + " : {:10.3E}\n").format(name, value))


class ParameterNameParser:
    """
    Utility class for parsing and manipulating parameter names.

    Parameter names follow a grammar:
        - Base: "Parameter name"
        - With units: "Parameter name [unit]"
        - With index: "Parameter name (idx) [unit]"

    Examples
    --------
    >>> parser = ParameterNameParser()
    >>> parser.parse_units("Temperature [K]")
    'K'
    >>> parser.split("Temperature [K]")
    ('Temperature', 'K')
    >>> parser.combine("Temperature", 0, "K")
    'Temperature (0) [K]'
    """

    # Regex for indexed parameter names: "name (idx) [unit]"
    _INDEXED_RE = re.compile(
        r"""^
            (?P<base>[^\[\]]+?)     # base name (non-greedy)
            \s\((?P<idx>\d+)\)      # (index)
            (?:\s\[(?P<tag>[^\]]+)\])? # optional [unit]
        $""",
        re.VERBOSE,
    )

    # Regex for simple parameter names: "name [unit]"
    _SIMPLE_RE = re.compile(
        r"""^
            (?P<base>[^\[\]]+?)     # base name (non-greedy)
            (?:\s\[(?P<tag>[^\]]+)\])? # optional [unit]
        $""",
        re.VERBOSE,
    )

    @classmethod
    def parse_units(cls, name: str) -> str | None:
        """
        Extract units from a parameter name.

        Example
        -------
        >>> ParameterNameParser.parse_units("Temperature [K]")
        'K'
        """
        match = cls._SIMPLE_RE.match(name)
        return match["tag"] if match else None

    @classmethod
    def split(cls, name: str) -> tuple[str, str | None]:
        """
        Split a parameter name into (base, units).

        Parameters
        ----------
        name : str
            Parameter name like "Temperature [K]".

        Returns
        -------
        tuple[str, str | None]
            (base_name, units) where units may be None.

        Raises
        ------
        ValueError
            If the name doesn't match the expected grammar.

        Example
        -------
        >>> ParameterNameParser.split("Temperature [K]")
        ('Temperature', 'K')
        """
        match = cls._SIMPLE_RE.match(name)
        if not match:
            raise ValueError(f"Illegal parameter name {name!r}")
        return match["base"].rstrip(), match["tag"]

    @classmethod
    def combine(cls, base: str, idx: int, units: str | None = None) -> str:
        """
        Combine base name, index, and optional units into a parameter name.

        Parameters
        ----------
        base : str
            Base parameter name.
        idx : int
            Index (must be >= 0).
        units : str, optional
            Units to append.

        Returns
        -------
        str
            Combined name like "base (idx) [units]".

        Example
        -------
        >>> ParameterNameParser.combine("a", 0, "V")
        'a (0) [V]'
        """
        if idx < 0:
            raise ValueError("idx must be ≥ 0")
        result = f"{base} ({idx})"
        if units:
            result += f" [{units}]"
        return result

    @classmethod
    def add_units(cls, base: str, units: str | None) -> str:
        """
        Add units to a base name.

        Example
        -------
        >>> ParameterNameParser.add_units("Temperature", "K")
        'Temperature [K]'
        """
        if units:
            return f"{base} [{units}]"
        return base

    @classmethod
    def parse_indexed(cls, name: str) -> tuple[str, int, str | None] | None:
        """
        Parse an indexed parameter name.

        Parameters
        ----------
        name : str
            Parameter name like "param (0) [unit]".

        Returns
        -------
        tuple[str, int, str | None] | None
            (base, index, units) or None if not an indexed name.

        Example
        -------
        >>> ParameterNameParser.parse_indexed("a (1) [V]")
        ('a', 1, 'V')
        >>> ParameterNameParser.parse_indexed("not indexed")  # Returns None
        """
        match = cls._INDEXED_RE.match(name)
        if not match:
            return None
        return match["base"], int(match["idx"]), match["tag"]


# Dictionary Transformation Utilities
def scalarize_dict(
    params: dict[str, Any], ignored_keys: list[str] | None = None
) -> dict[str, Any]:
    """
    Expand list-valued items into scalar keys while preserving tags.

    This is useful for serialization where lists need to be flattened.

    Parameters
    ----------
    params : dict[str, Any]
        The dictionary to scalarize.
    ignored_keys : list[str], optional
        Keys to skip (not expand). Defaults to ["citations"].

    Returns
    -------
    dict[str, Any]
        The scalarized dictionary.

    Examples
    --------
    >>> scalarize_dict({'a [V]': [1, 2]})
    {'a (0) [V]': 1, 'a (1) [V]': 2}
    """
    out = {}
    ignored_keys = ignored_keys or ["citations"]

    for key, val in params.items():
        if key not in ignored_keys and isinstance(val, list):
            base, units = ParameterNameParser.split(key)
            for i, item in enumerate(val):
                indexed_key = ParameterNameParser.combine(base, i, units)
                if indexed_key in out:
                    raise ValueError(f"Duplicate key {indexed_key!r}")
                out[indexed_key] = item
        else:
            if key in out:
                raise ValueError(f"Duplicate key {key!r}")
            out[key] = val
    return out


def arrayize_dict(scalar_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Collapse indexed scalar keys back into lists.

    This is the inverse of scalarize_dict. A sequence is collapsed
    only if indices 0…N are all present.

    Parameters
    ----------
    scalar_dict : dict[str, Any]
        The dictionary with indexed keys.

    Returns
    -------
    dict[str, Any]
        The arrayized dictionary.

    Examples
    --------
    >>> arrayize_dict({'a (0) [V]': 1, 'a (1) [V]': 2})
    {'a [V]': [1, 2]}
    """
    out = {}
    processed = set()

    # Discover (base, tag) pairs that appear indexed
    pairs = set()
    for key in scalar_dict:
        parsed = ParameterNameParser.parse_indexed(key)
        if parsed:
            base, _, units = parsed
            pairs.add((base, units))

    # Rebuild each pair
    for base, units in pairs:
        idx_val = {}
        own_keys = []

        for key, val in scalar_dict.items():
            parsed = ParameterNameParser.parse_indexed(key)
            if parsed and parsed[0] == base and parsed[2] == units:
                idx = parsed[1]
                if idx in idx_val:
                    raise ValueError(f"Duplicate index {idx} for '{base}'")
                idx_val[idx] = val
                own_keys.append(key)

        if not idx_val:
            raise ValueError(
                f"No indices found for '{base}'. "
                "This should not happen. Please report this bug."
            )

        indices = set(idx_val)
        if not _is_contiguous(indices):
            missing = sorted(set(range(max(indices) + 1)) - indices)
            raise ValueError(f"Missing indices {missing} for '{base}'")

        collapsed_key = ParameterNameParser.add_units(base, units)
        if collapsed_key in out:
            raise ValueError(f"Duplicate key after rebuild: {collapsed_key!r}")

        out[collapsed_key] = [idx_val[i] for i in range(max(idx_val) + 1)]
        processed.update(own_keys)

    # Copy non-indexed entries
    for key, val in scalar_dict.items():
        if key not in processed:
            if key in out:
                raise ValueError(f"Duplicate key: {key!r}")
            out[key] = val

    return out


def _is_contiguous(indices: set[int]) -> bool:
    """Check if indices form a contiguous sequence starting from 0."""
    return bool(indices) and indices == set(range(max(indices) + 1))


def _is_iterable(val: Any) -> bool:
    """Check if a value is iterable (but not string, dict, or bytes)."""
    return hasattr(val, "__iter__") and not isinstance(val, (str, dict, bytes))


# JSON Conversion Utilities
def convert_symbols_in_dict(data_dict: dict | None = None) -> dict:
    """
    Recursively convert nested dicts using convert_symbol_from_json.

    Parameters
    ----------
    data_dict : dict, optional
        Dictionary to process. Returns empty dict if None.

    Returns
    -------
    dict
        Dictionary with symbols converted.

    Examples
    --------
    >>> data = {"Temperature [K]": "298.15", "Voltage [V]": {"type": "symbol"}}
    >>> converted = convert_symbols_in_dict(data)
    """
    if not data_dict:
        return {}

    for key, value in data_dict.items():
        if isinstance(value, dict) and "interpolator" in value:
            # Handle interpolant specification
            interpolator = value.get("interpolator", "linear")
            x = value.get("x", [])
            y = value.get("y", [])

            def interpolant_function(sto, x=x, y=y, interpolator=interpolator):
                try:
                    return pybamm.Interpolant(x, y, sto, interpolator=interpolator)
                except Exception as e:
                    print(e)
                    return pybamm.Scalar(0)

            data_dict[key] = interpolant_function
        elif isinstance(value, dict):
            data_dict[key] = convert_symbol_from_json(value)
        elif isinstance(value, list):
            data_dict[key] = [
                convert_symbol_from_json(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str):
            data_dict[key] = float(value)

    return data_dict


def convert_parameter_values_to_json(
    parameter_values: ParameterValues, filename: str | None = None
) -> dict:
    """
    Convert a ParameterValues object to a JSON-serializable dictionary.

    Optionally saves it to a file.

    Parameters
    ----------
    parameter_values : ParameterValues
        The ParameterValues object to convert.
    filename : str, optional
        The filename to save the JSON file to.

    Returns
    -------
    dict
        The JSON-serializable dictionary.

    Examples
    --------
    >>> param = pybamm.ParameterValues("Chen2020")
    >>> json_dict = convert_parameter_values_to_json(param)
    >>> convert_parameter_values_to_json(param, "params.json")  # Save to file
    {}
    """
    parameter_values_dict = {}

    for k, v in parameter_values.items():
        if callable(v):
            parameter_values_dict[k] = convert_symbol_to_json(
                convert_function_to_symbolic_expression(v, k)
            )
        else:
            parameter_values_dict[k] = convert_symbol_to_json(v)

    if filename is not None:
        with open(filename, "w") as f:
            json.dump(
                parameter_values_dict, f, indent=2, default=Serialise._json_encoder
            )

    return parameter_values_dict
