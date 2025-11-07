import json
import numbers
import re
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Any
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


class ParameterValues:
    """
    The parameter values for a simulation.

    Note that this class does not inherit directly from the python dictionary class as
    this causes issues with saving and loading simulations.

    Parameters
    ----------
    values : dict or string
        Explicit set of parameters, or reference to an inbuilt parameter set
        If string and matches one of the inbuilt parameter sets, returns that parameter
        set.

    Examples
    --------
    >>> values = {"some parameter": 1, "another parameter": 2}
    >>> param = pybamm.ParameterValues(values)
    >>> param["some parameter"]
    1
    >>> param = pybamm.ParameterValues("Marquis2019")
    >>> param["Reference temperature [K]"]
    298.15

    """

    def __init__(self, values):
        # add physical constants as default values
        self._dict_items = pybamm.FuzzyDict(
            {
                "Ideal gas constant [J.K-1.mol-1]": pybamm.constants.R.value,
                "Faraday constant [C.mol-1]": pybamm.constants.F.value,
                "Boltzmann constant [J.K-1]": pybamm.constants.k_b.value,
                "Electron charge [C]": pybamm.constants.q_e.value,
            }
        )

        if isinstance(values, dict | ParameterValues):
            # remove the "chemistry" key if it exists
            chemistry = values.pop("chemistry", None)
            self.update(values, check_already_exists=False)
        else:
            # Check if values is a named parameter set
            if isinstance(values, str) and values in pybamm.parameter_sets.keys():
                values = pybamm.parameter_sets[values]
                chemistry = values.pop("chemistry", None)
                self.update(values, check_already_exists=False)
            else:
                valid_sets = "\n".join(pybamm.parameter_sets.keys())
                raise ValueError(
                    f"'{values}' is not a valid parameter set. Parameter set must be one of:\n{valid_sets}"
                )

        if chemistry == "ecm":
            self._set_initial_state = pybamm.equivalent_circuit.set_initial_state
        else:
            self._set_initial_state = pybamm.lithium_ion.set_initial_state

        # Initialise empty _processed_symbols dict (for caching)
        self._processed_symbols = {}

        # save citations
        if "citations" in self._dict_items:
            for citation in self._dict_items["citations"]:
                pybamm.citations.register(citation)

    @staticmethod
    def _create_from_bpx(bpx, target_soc):
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

        # get initial concentrations based on SOC
        # Note: we cannot set SOC for blended electrodes,
        # see https://github.com/pybamm-team/PyBaMM/issues/2682
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

        return pybamm.ParameterValues(pybamm_dict)

    @staticmethod
    def create_from_bpx_obj(bpx_obj, target_soc: float = 1):
        """
        Parameters
        ----------
        bpx_obj: dict
            A dictionary containing the parameters in the `BPX <https://bpxstandard.com/>`_ format
        target_soc : float, optional
            Target state of charge. Must be between 0 and 1. Default is 1.

        Returns
        -------
        ParameterValues
            A parameter values object with the parameters in the bpx file

        """
        from bpx import parse_bpx_obj

        bpx = parse_bpx_obj(bpx_obj)
        return ParameterValues._create_from_bpx(bpx, target_soc)

    @staticmethod
    def create_from_bpx(filename, target_soc: float = 1):
        """
        Parameters
        ----------
        filename: str
            The filename of the `BPX <https://bpxstandard.com/>`_ file
        target_soc : float, optional
            Target state of charge. Must be between 0 and 1. Default is 1.

        Returns
        -------
        ParameterValues
            A parameter values object with the parameters in the bpx file

        """
        from bpx import parse_bpx_file

        bpx = parse_bpx_file(filename)
        return ParameterValues._create_from_bpx(bpx, target_soc)

    def __getitem__(self, key):
        try:
            return self._dict_items[key]
        except KeyError as err:
            if (
                "Exchange-current density for lithium metal electrode [A.m-2]"
                in err.args[0]
                and "Exchange-current density for plating [A.m-2]" in self._dict_items
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
                raise err

    def get(self, key, default=None):
        """Return item corresponding to key if it exists, otherwise return default"""
        try:
            return self._dict_items[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        """Call the update functionality when doing a setitem"""
        self.update({key: value})

    def __delitem__(self, key):
        del self._dict_items[key]

    def __repr__(self):
        return pformat(self._dict_items, width=1)

    def __eq__(self, other):
        return self._dict_items == other._dict_items

    def keys(self):
        """Get the keys of the dictionary"""
        return self._dict_items.keys()

    def values(self):
        """Get the values of the dictionary"""
        return self._dict_items.values()

    def items(self):
        """Get the items of the dictionary"""
        return self._dict_items.items()

    def pop(self, *args, **kwargs):
        return self._dict_items.pop(*args, **kwargs)

    def copy(self):
        """Returns a copy of the parameter values. Makes sure to copy the internal
        dictionary."""
        new_copy = ParameterValues(self._dict_items.copy())
        new_copy._set_initial_state = self._set_initial_state
        return new_copy

    def search(self, key, print_values=True):
        """
        Search dictionary for keys containing 'key'.

        See :meth:`pybamm.FuzzyDict.search()`.
        """
        return self._dict_items.search(key, print_values)

    def update(self, values, check_conflict=False, check_already_exists=True, path=""):
        """
        Update parameter dictionary, while also performing some basic checks.

        Parameters
        ----------
        values : dict
            Dictionary of parameter values to update parameter dictionary with
        check_conflict : bool, optional
            Whether to check that a parameter in `values` has not already been defined
            in the parameter class when updating it, and if so that its value does not
            change. This is set to True during initialisation, when parameters are
            combined from different sources, and is False by default otherwise
        check_already_exists : bool, optional
            Whether to check that a parameter in `values` already exists when trying to
            update it. This is to avoid cases where an intended change in the parameters
            is ignored due a typo in the parameter name, and is True by default but can
            be manually overridden.
        path : string, optional
            Path from which to load functions
        """
        # check if values is not a dictionary
        if not isinstance(values, dict):
            values = values._dict_items
        # check parameter values
        values = self.check_parameter_values(values)
        # update
        for name, value in values.items():
            # check for conflicts
            if (
                check_conflict is True
                and name in self.keys()
                and not (self[name] == float(value) or self[name] == value)
            ):
                raise ValueError(
                    f"parameter '{name}' already defined with value '{self[name]}'"
                )
            # check parameter already exists (for updating parameters)
            if check_already_exists is True:
                try:
                    self._dict_items[name]
                except KeyError as err:
                    raise KeyError(
                        f"Cannot update parameter '{name}' as it does not "
                        + f"have a default value. ({err.args[0]}). If you are "
                        + "sure you want to update this parameter, use "
                        + "param.update({name: value}, check_already_exists=False)"
                    ) from err
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
                    self._dict_items[name] = pybamm.InputParameter(name)
                # Anything else should be a converted to a float
                else:
                    self._dict_items[name] = float(value)
            elif isinstance(value, tuple) and isinstance(value[1], np.ndarray):
                # If data is provided as a 2-column array (1D data),
                # convert to two arrays for compatibility with 2D data
                # see #1805
                func_name, data = value
                data = ([data[:, 0]], data[:, 1])
                self._dict_items[name] = (func_name, data)
            else:
                self._dict_items[name] = value
        # reset processed symbols
        self._processed_symbols = {}

    def set_initial_state(
        self,
        initial_value,
        direction=None,
        param=None,
        inplace=True,
        options=None,
        inputs=None,
    ):
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

    @staticmethod
    def check_parameter_values(values):
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

    def process_model(self, unprocessed_model, inplace=True):
        """Assign parameter values to a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        unprocessed_model : :class:`pybamm.BaseModel`
            Model to assign parameter values for
        inplace: bool, optional
            If True, replace the parameters in the model in place. Otherwise, return a
            new model with parameter values set. Default is True.

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic = {}` and
            `model.variables = {}`)

        """
        pybamm.logger.info(f"Start setting parameters for {unprocessed_model.name}")

        # set up inplace vs not inplace
        if inplace:
            # any changes to unprocessed_model attributes will change model attributes
            # since they point to the same object
            model = unprocessed_model
        else:
            # create a copy of the model
            model = unprocessed_model.new_copy()

        if (
            len(unprocessed_model.rhs) == 0
            and len(unprocessed_model.algebraic) == 0
            and len(unprocessed_model.variables) == 0
        ):
            raise pybamm.ModelError("Cannot process parameters for empty model")

        new_rhs = {}
        for variable, equation in unprocessed_model.rhs.items():
            pybamm.logger.verbose(f"Processing parameters for {variable!r} (rhs)")
            new_variable = self.process_symbol(variable)
            new_rhs[new_variable] = self.process_symbol(equation)
        model.rhs = new_rhs

        new_algebraic = {}
        for variable, equation in unprocessed_model.algebraic.items():
            pybamm.logger.verbose(f"Processing parameters for {variable!r} (algebraic)")
            new_variable = self.process_symbol(variable)
            new_algebraic[new_variable] = self.process_symbol(equation)
        model.algebraic = new_algebraic

        new_initial_conditions = {}
        for variable, equation in unprocessed_model.initial_conditions.items():
            pybamm.logger.verbose(
                f"Processing parameters for {variable!r} (initial conditions)"
            )
            new_variable = self.process_symbol(variable)
            new_initial_conditions[new_variable] = self.process_symbol(equation)
        model.initial_conditions = new_initial_conditions

        model.boundary_conditions = self.process_boundary_conditions(unprocessed_model)

        new_variables = {}
        for variable, equation in unprocessed_model.variables.items():
            pybamm.logger.verbose(f"Processing parameters for {variable!r} (variables)")
            new_variables[variable] = self.process_symbol(equation)
        model.variables = new_variables

        new_events = []
        for event in unprocessed_model.events:
            pybamm.logger.verbose(f"Processing parameters for event '{event.name}''")
            new_events.append(
                pybamm.Event(
                    event.name, self.process_symbol(event.expression), event.event_type
                )
            )

        interpolant_events = self._get_interpolant_events(model)
        for event in interpolant_events:
            pybamm.logger.verbose(f"Processing parameters for event '{event.name}''")
            new_events.append(
                pybamm.Event(
                    event.name, self.process_symbol(event.expression), event.event_type
                )
            )

        model.events = new_events

        pybamm.logger.info(f"Finish setting parameters for {model.name}")

        return model

    def _get_interpolant_events(self, model):
        """Add events for functions that have been defined as parameters"""
        # Define events to catch extrapolation. In these events the sign is
        # important: it should be positive inside of the range and negative
        # outside of it
        interpolants = model._find_symbols(pybamm.Interpolant)
        interpolant_events = []
        for interpolant in interpolants:
            xs = interpolant.x
            children = interpolant.children
            for x, child in zip(xs, children, strict=False):
                interpolant_events.extend(
                    [
                        pybamm.Event(
                            f"Interpolant '{interpolant.name}' lower bound",
                            pybamm.min(child - min(x)),
                            pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
                        ),
                        pybamm.Event(
                            f"Interpolant '{interpolant.name}' upper bound",
                            pybamm.min(max(x) - child),
                            pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
                        ),
                    ]
                )
        return interpolant_events

    def process_boundary_conditions(self, model):
        """
        Process boundary conditions for a model
        Boundary conditions are dictionaries {"left": left bc, "right": right bc}
        in general, but may be imposed on the tabs (or *not* on the tab) for a
        small number of variables, e.g. {"negative tab": neg. tab bc,
        "positive tab": pos. tab bc "no tab": no tab bc}.
        """
        new_boundary_conditions = {}
        sides = [
            "left",
            "right",
            "negative tab",
            "positive tab",
            "no tab",
            "top",
            "bottom",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
            "r_min",
            "r_max",
        ]
        for variable, bcs in model.boundary_conditions.items():
            processed_variable = self.process_symbol(variable)
            new_boundary_conditions[processed_variable] = {}
            for side in sides:
                try:
                    bc, typ = bcs[side]
                    pybamm.logger.verbose(
                        f"Processing parameters for {variable!r} ({side} bc)"
                    )
                    processed_bc = (self.process_symbol(bc), typ)
                    new_boundary_conditions[processed_variable][side] = processed_bc
                except KeyError as err:
                    # don't raise error if the key error comes from the side not being
                    # found
                    if err.args[0] in side:
                        pass
                    # do raise error otherwise (e.g. can't process symbol)
                    else:
                        raise err

        return new_boundary_conditions

    def process_geometry(self, geometry):
        """
        Assign parameter values to a geometry (inplace).

        Parameters
        ----------
        geometry : dict
            Geometry specs to assign parameter values to
        """

        def process_and_check(sym):
            new_sym = self.process_symbol(sym)
            leaves = new_sym.post_order(filter=lambda node: len(node.children) == 0)
            for leaf in leaves:
                if not isinstance(leaf, pybamm.Scalar) and not isinstance(
                    leaf, pybamm.InputParameter
                ):
                    raise ValueError(
                        "Geometry parameters must be Scalars or InputParameters after parameter processing"
                    )
            return new_sym

        for domain in geometry:
            for spatial_variable, spatial_limits in geometry[domain].items():
                # process tab information if using 1 or 2D current collectors
                if spatial_variable == "tabs":
                    for tab, position_info in spatial_limits.items():
                        for position_size, sym in position_info.items():
                            geometry[domain]["tabs"][tab][position_size] = (
                                process_and_check(sym)
                            )
                else:
                    for lim, sym in spatial_limits.items():
                        geometry[domain][spatial_variable][lim] = process_and_check(sym)

    def process_symbol(self, symbol):
        """Walk through the symbol and replace any Parameter with a Value.
        If a symbol has already been processed, the stored value is returned.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol or Expression tree to set parameters for

        Returns
        -------
        symbol : :class:`pybamm.Symbol`
            Symbol with Parameter instances replaced by Value

        """
        try:
            return self._processed_symbols[symbol]
        except KeyError:
            if not isinstance(symbol, pybamm.FunctionParameter):
                processed_symbol = self._process_symbol(symbol)
            else:
                processed_symbol = self._process_function_parameter(symbol)
            self._processed_symbols[symbol] = processed_symbol

            return processed_symbol

    def _process_symbol(self, symbol):
        """See :meth:`ParameterValues.process_symbol()`."""

        if isinstance(symbol, pybamm.Parameter):
            value = self[symbol.name]
            if isinstance(value, numbers.Number):
                # Check not NaN (parameter in csv file but no value given)
                if np.isnan(value):
                    raise ValueError(f"Parameter '{symbol.name}' not found")
                # Scalar inherits name
                return pybamm.Scalar(value, name=symbol.name)
            elif isinstance(value, pybamm.Symbol):
                new_value = self.process_symbol(value)
                new_value.copy_domains(symbol)
                return new_value
            else:
                raise TypeError(f"Cannot process parameter '{value}'")

        elif isinstance(symbol, pybamm.FunctionParameter):
            function_name = self[symbol.name]
            if isinstance(
                function_name,
                numbers.Number | pybamm.Interpolant | pybamm.InputParameter,
            ) or (
                isinstance(function_name, pybamm.Symbol)
                and function_name.size_for_testing == 1
            ):
                # no need to process children, they will only be used for shape
                new_children = symbol.children
            else:
                # process children
                new_children = []
                for child in symbol.children:
                    if symbol.diff_variable is not None and any(
                        x == symbol.diff_variable for x in child.pre_order()
                    ):
                        # Wrap with NotConstant to avoid simplification,
                        # which would stop symbolic diff from working properly
                        new_child = pybamm.NotConstant(child)
                        new_children.append(self.process_symbol(new_child))
                    else:
                        new_children.append(self.process_symbol(child))

            # Create Function or Interpolant or Scalar object
            if isinstance(function_name, tuple):
                if len(function_name) == 2:  # CSV or JSON parsed data
                    # to create an Interpolant
                    name, data = function_name

                    if len(data[0]) == 1:
                        input_data = data[0][0], data[1]

                    else:
                        input_data = data

                    # For parameters provided as data we use a cubic interpolant
                    # Note: the cubic interpolant can be differentiated
                    function = pybamm.Interpolant(
                        input_data[0],
                        input_data[-1],
                        new_children,
                        name=name,
                    )

                else:  # pragma: no cover
                    raise ValueError(
                        f"Invalid function name length: {len(function_name)}"
                    )

            elif isinstance(function_name, numbers.Number):
                # Check not NaN (parameter in csv file but no value given)
                if np.isnan(function_name):
                    raise ValueError(
                        f"Parameter '{symbol.name}' (possibly a function) not found"
                    )
                # If the "function" is provided is actually a scalar, return a Scalar
                # object instead of throwing an error.
                function = pybamm.Scalar(function_name, name=symbol.name)
            elif callable(function_name):
                # otherwise evaluate the function to create a new PyBaMM object
                function = function_name(*new_children)
            elif isinstance(
                function_name, pybamm.Interpolant | pybamm.InputParameter
            ) or (
                isinstance(function_name, pybamm.Symbol)
                and function_name.size_for_testing == 1
            ):
                function = function_name
            else:
                raise TypeError(
                    f"Parameter provided for '{symbol.name}' "
                    + "is of the wrong type (should either be scalar-like or callable)"
                )
            # Differentiate if necessary
            if symbol.diff_variable is None:
                # Use ones_like so that we get the right shapes
                function_out = function * pybamm.ones_like(*new_children)
            else:
                # return differentiated function
                new_diff_variable = self.process_symbol(symbol.diff_variable)
                function_out = function.diff(new_diff_variable)
            # Process again just to be sure
            return self.process_symbol(function_out)

        # Unary operators
        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.child)
            new_symbol = symbol.create_copy(new_children=[new_child])
            # x_average can sometimes create a new symbol with electrode thickness
            # parameters, so we process again to make sure these parameters are set
            if isinstance(symbol, pybamm.XAverage) and not isinstance(
                new_symbol, pybamm.XAverage
            ):
                new_symbol = self.process_symbol(new_symbol)
            # f_a_dist in the size average needs to be processed
            if isinstance(new_symbol, pybamm.SizeAverage):
                new_symbol.f_a_dist = self.process_symbol(new_symbol.f_a_dist)
            # position in evaluate at needs to be processed, and should be a Scalar
            if isinstance(new_symbol, pybamm.EvaluateAt):
                new_symbol_position = self.process_symbol(new_symbol.position)
                if not isinstance(new_symbol_position, pybamm.Scalar):
                    raise ValueError(
                        "'position' in 'EvaluateAt' must evaluate to a scalar"
                    )
                else:
                    new_symbol.position = new_symbol_position
            return new_symbol

        # Functions, BinaryOperators & Concatenations
        elif (
            isinstance(symbol, pybamm.Function)
            or isinstance(symbol, pybamm.Concatenation)
            or isinstance(symbol, pybamm.BinaryOperator)
        ):
            new_children = [self.process_symbol(child) for child in symbol.children]
            return symbol.create_copy(new_children)

        elif isinstance(symbol, pybamm.VectorField):
            left_symbol = self.process_symbol(symbol.lr_field)
            right_symbol = self.process_symbol(symbol.tb_field)
            return symbol.create_copy(new_children=[left_symbol, right_symbol])

        # Variables: update scale
        elif isinstance(symbol, pybamm.Variable):
            new_symbol = symbol.create_copy()
            new_symbol._scale = self.process_symbol(symbol.scale)
            reference = self.process_symbol(symbol.reference)
            if isinstance(reference, pybamm.Vector):
                # address numpy 1.25 deprecation warning: array should have ndim=0
                # before conversion
                reference = pybamm.Scalar((reference.evaluate()).item())
            new_symbol._reference = reference
            new_symbol.bounds = tuple([self.process_symbol(b) for b in symbol.bounds])
            return new_symbol

        elif isinstance(symbol, numbers.Number):
            return pybamm.Scalar(symbol)

        else:
            # Backup option: return the object
            return symbol

    def _process_function_parameter(self, symbol):
        function_parameter = self[symbol.name]
        # Handle symbolic function parameter case
        if isinstance(function_parameter, pybamm.ExpressionFunctionParameter):
            # Process children
            new_children = []
            for child in symbol.children:
                if symbol.diff_variable is not None and any(
                    x == symbol.diff_variable for x in child.pre_order()
                ):
                    # Wrap with NotConstant to avoid simplification,
                    # which would stop symbolic diff from working properly
                    new_child = pybamm.NotConstant(child)
                    new_children.append(self.process_symbol(new_child))
                else:
                    new_children.append(self.process_symbol(child))

            # Get the expression and inputs for the function
            expression = function_parameter.child
            inputs = {
                arg: child
                for arg, child in zip(
                    function_parameter.func_args, symbol.children, strict=True
                )
            }

            # Set domains for function inputs in post-order traversal
            for node in expression.post_order():
                if node.name in inputs:
                    node.domains = inputs[node.name].domains
                else:
                    node.domains = node.get_children_domains(node.children)

            # Combine parameter values with inputs
            combined_params = ParameterValues({**self, **inputs})

            # Process any FunctionParameter children first to avoid recursion
            for child in expression.pre_order():
                if isinstance(child, pybamm.FunctionParameter):
                    # Build new child with parent inputs
                    new_child_children = [
                        inputs[child_child.name]
                        if isinstance(child_child, pybamm.Parameter)
                        and child_child.name in inputs
                        else child_child
                        for child_child in child.children
                    ]
                    new_child = pybamm.FunctionParameter(
                        child.name,
                        dict(zip(child.input_names, new_child_children, strict=False)),
                        diff_variable=child.diff_variable,
                        print_name=child.print_name,
                    )

                    # For this local combined parameter values, process the new child
                    # and store the result as the processed symbol for this child
                    # This means the child is evaluated with the parent inputs only when
                    # it is called from within the parent function (not elsewhere in
                    # the expression tree)
                    combined_params._processed_symbols[child] = (
                        combined_params.process_symbol(new_child)
                    )

            # Process function with combined parameter values to get a symbolic expression
            function = combined_params.process_symbol(expression)

            # Differentiate if necessary
            if symbol.diff_variable is None:
                # Use ones_like so that we get the right shapes
                function_out = function * pybamm.ones_like(*new_children)
            else:
                # return differentiated function
                new_diff_variable = self.process_symbol(symbol.diff_variable)
                function_out = function.diff(new_diff_variable)

            return function_out

        # Handle non-symbolic function_name case
        else:
            return self._process_symbol(symbol)

    def evaluate(self, symbol, inputs=None):
        """
        Process and evaluate a symbol.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol or Expression tree to evaluate

        Returns
        -------
        number or array
            The evaluated symbol
        """
        processed_symbol = self.process_symbol(symbol)
        if processed_symbol.is_constant():
            return processed_symbol.evaluate()
        else:
            # In the case that the only issue is an input parameter contained in inputs,
            # go ahead and try and evaluate it with the inputs. If it doesn't work, raise
            # the value error.
            try:
                return processed_symbol.evaluate(inputs=inputs)
            except Exception as exc:
                raise ValueError(
                    "symbol must evaluate to a constant scalar or array"
                ) from exc

    def _ipython_key_completions_(self):
        return list(self._dict_items.keys())

    def print_parameters(self, parameters, output_file=None):
        """
        Return dictionary of evaluated parameters, and optionally print these evaluated
        parameters to an output file.

        Parameters
        ----------
        parameters : class or dict containing :class:`pybamm.Parameter` objects
            Class or dictionary containing all the parameters to be evaluated
        output_file : string, optional
            The file to print parameters to. If None, the parameters are not printed,
            and this function simply acts as a test that all the parameters can be
            evaluated, and returns the dictionary of evaluated parameters.

        Returns
        -------
        evaluated_parameters : defaultdict
            The evaluated parameters, for further processing if needed

        Notes
        -----
        A C-rate of 1 C is the current required to fully discharge the battery in 1
        hour, 2 C is current to discharge the battery in 0.5 hours, etc
        """
        # Set list of attributes to ignore, for when we are evaluating parameters from
        # a class of parameters
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

        # Turn to regular dictionary for faster KeyErrors
        self._dict_items = dict(self._dict_items)

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

            # Turn back to FuzzyDict
            self._dict_items = pybamm.FuzzyDict(self._dict_items)

        # Print the evaluated_parameters dict to output_file
        if output_file:
            self.print_evaluated_parameters(evaluated_parameters, output_file)

        return evaluated_parameters

    def print_evaluated_parameters(self, evaluated_parameters, output_file):
        """
        Print a dictionary of evaluated parameters to an output file

        Parameters
        ----------
        evaluated_parameters : defaultdict
            The evaluated parameters, for further processing if needed
        output_file : string, optional
            The file to print parameters to. If None, the parameters are not printed,
            and this function simply acts as a test that all the parameters can be
            evaluated

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

    def __contains__(self, key):
        return key in self._dict_items

    def __iter__(self):
        return iter(self._dict_items)

    @staticmethod
    def from_json(filename_or_dict):
        """
        Loads a ParameterValues object from a JSON file or a dictionary.

        Parameters
        ----------
        filename_or_dict : string-like or dict
            The filename to load the JSON file from, or a dictionary.

        Returns
        -------
        ParameterValues
            The ParameterValues object
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

    def to_json(self, filename=None):
        """
        Converts the parameter values to a JSON-serializable dictionary and optionally
        saves it to a file.

        Parameters
        ----------
        filename : str, optional
            The filename to save the JSON file to. If not provided, the dictionary is
            not saved.

        Returns
        -------
        dict
            The JSON-serializable dictionary
        """
        return convert_parameter_values_to_json(self, filename)


_SCALAR_KEY_RE = re.compile(
    r""" ^
         (?P<base>[^\[\]]+?)           # foo
         \s\((?P<idx>\d+)\)            # (0)
         (?: \s\[(?P<tag>[^\]]+)\])?   # optional [unit]
         $ """,
    re.VERBOSE,
)


def convert_symbols_in_dict(
    data_dict: dict | None = None,
) -> dict:
    """Recursively converts nested dicts using convert_symbol_from_json."""
    # Create parameter values
    if data_dict:
        for key, value in data_dict.items():
            if isinstance(value, dict) and "interpolator" in value:
                # handle interpolant
                interpolator = value.get("interpolator", "linear")
                x = value.get("x", [])
                y = value.get("y", [])

                # Convert list to pybamm.Interpolant
                def interpolant_function(sto, x=x, y=y, interpolator=interpolator):
                    try:
                        return pybamm.Interpolant(x, y, sto, interpolator=interpolator)
                    except Exception as e:
                        print(e)
                        return pybamm.Scalar(0)

                data_dict[key] = interpolant_function
            elif isinstance(value, dict):
                # Handle function parameters in JSON format
                # Recursively process nested dictionaries
                data_dict[key] = convert_symbol_from_json(value)
            elif isinstance(value, list):
                data_dict[key] = [
                    convert_symbol_from_json(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, str):
                # Handle function parameters in string format
                data_dict[key] = float(value)
            # Keep other types as is
    else:
        # Return an empty dict if input is None
        data_dict = {}

    return data_dict


class _KeyMatch:
    """
    Match a parameter name against the key grammar.
    """

    __slots__ = ["base", "idx", "is_match", "name", "tag"]

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        self.name = name

        match = _SCALAR_KEY_RE.match(name)
        self.is_match = bool(match)

        if match:
            base = match["base"]
            idx = int(match["idx"])
            tag = match["tag"]
        else:
            base = ""
            idx = -1
            tag = ""

        self.base = base
        self.idx = idx
        self.tag = tag

    def __bool__(self):
        return self.is_match


def scalarize_dict(
    params: dict[str, Any], ignored_keys: list[str] | None = None
) -> dict[str, Any]:
    """
    Expand list-valued items into scalar keys while preserving tags.
    Example
    -------
    {'a [V]': [1, 2]}  →  {'a (0) [V]': 1, 'a (1) [V]': 2}

    Parameters
    ----------
    params : dict[str, Any]
        The dictionary to scalarize
    ignored_keys : list[str], optional
        The keys to ignore. Defaults to ``["citations"]``.

    Returns
    -------
    dict[str, Any]
        The scalarized dictionary
    """
    out = {}

    # Default ignored keys. Special case for citations in `pybamm.ParameterValues`
    if ignored_keys is None:
        ignored_keys = ["citations"]

    for key, val in params.items():
        if key not in ignored_keys and isinstance(val, list):
            base, tag = _split_key(key)  # accepts 'a' or 'a [V]'
            for i, item in enumerate(val):
                key_i = _combine_name(base, i, tag)
                if key_i in out:
                    raise ValueError(f"Duplicate key {key_i!r}")
                out[key_i] = item
        else:
            if key in out:
                raise ValueError(f"Duplicate key {key!r}")
            out[key] = val
    return out


def _is_iterable(val: Any) -> bool:
    return hasattr(val, "__iter__") and not isinstance(val, (str | dict | bytes))


_KEY_RE = re.compile(
    r""" ^
         (?P<base>[^\[\]]+?)           # foo
         (?: \s\[(?P<tag>[^\]]+)\])?   # optional [unit]   (for collapsed keys)
         $ """,
    re.VERBOSE,
)


def _split_key(name: str) -> tuple[str, str | None]:
    """
    Separate *name* into ``(base, tag)`` where *tag* can be ``None``.
    Works for either collapsed or scalar keys.
    """
    m = _KEY_RE.match(name)
    if not m:
        raise ValueError(f"Illegal parameter name {name!r}")
    return m["base"].rstrip(), m["tag"]


def _combine_name(base: str, idx: int, tag: str | None = None) -> str:
    """Return ``'base (idx)'`` or ``'base (idx) [tag]'``."""
    if idx < 0:
        raise ValueError("idx must be ≥ 0")
    out = f"{base} ({idx})"
    return _add_units(out, tag)


def _add_units(base: str, tag: str | None) -> str:
    """Return ``'base'`` or ``'base [tag]'``."""
    out = f"{base}"
    if tag:
        out += f" [{tag}]"
    return out


def arrayize_dict(
    scalar_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Collapse scalar keys back into lists.  Tags are kept with the base name
    (``'a [V]'``).  A sequence is collapsed only if indices 0…N are all present.

    Parameters
    ----------
    scalar_dict : dict[str, Any]
        The dictionary to arrayize

    Returns
    -------
    dict[str, Any]
        The arrayized dictionary
    """
    out = {}
    processed = set()

    # discover (base, tag) pairs that appear scalarised
    pairs = set()
    for k in scalar_dict:
        match = _KeyMatch(k)
        if match:
            pairs.add((match.base, match.tag))

    # rebuild each pair
    for base, tag in pairs:
        idx_val = {}
        own_keys = []

        for k, v in scalar_dict.items():
            match = _KeyMatch(k)
            if match and match.base == base and match.tag == tag:
                if match.idx in idx_val:
                    raise ValueError(f"Duplicate index {match.idx} for '{base}'")
                idx_val[match.idx] = v
                own_keys.append(k)

        if not idx_val:
            raise ValueError(
                f"No indices found for '{base}'. "
                "It should not be possible to reach here. Please report this bug."
            )

        indices = set(idx_val)
        if not _contiguous_and_ordered_indices(indices):
            missing = sorted(set(range(max(indices) + 1)) - indices)
            raise ValueError(f"Missing indices {missing} for '{base}'")

        collapsed_key = _add_units(base, tag)
        if collapsed_key in out:
            raise ValueError(f"Duplicate key after rebuild: {collapsed_key!r}")

        out[collapsed_key] = [idx_val[i] for i in range(max(idx_val) + 1)]
        processed.update(own_keys)

    # copy untouched scalars
    for k, v in scalar_dict.items():
        if k not in processed:
            if k in out:
                raise ValueError(f"Duplicate key: {k!r}")
            out[k] = v

    return out


def _contiguous_and_ordered_indices(indices: set[int]) -> bool:
    """
    Check if a set of indices forms a contiguous sequence starting from 0
    to max(indices).

    Parameters
    ----------
    indices : set[int]
        Set of integer indices to check for contiguity

    Returns
    -------
    bool
        True if indices form the sequence {0, 1, 2, ..., max(indices)},
        False otherwise. Returns False for empty sets.
    """
    return bool(indices) and indices == set(range(max(indices) + 1))


def convert_parameter_values_to_json(parameter_values, filename=None):
    """
    Converts a ParameterValues object to a JSON-serializable dictionary and optionally
    saves it to a file.

    Parameters
    ----------
    parameter_values : ParameterValues
        The ParameterValues object to convert

    filename : str, optional
        The filename to save the JSON file to. If not provided, the dictionary is
        not saved.
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
