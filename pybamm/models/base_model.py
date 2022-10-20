#
# Base model class
#
import copy
import casadi
import numpy as np

import pybamm
from pybamm.expression_tree.operations.latexify import Latexify

_EQUATION_ATTRIBUTES = [
    "rhs",
    "algebraic",
    "initial_conditions",
    "boundary_conditions",
    "variables",
    "external_variables",
    "events",
    "len_rhs",
    "len_alg",
    "len_rhs_and_alg",
    "bounds",
    "mass_matrix",
    "mass_matrix_inv",
    "concatenated_rhs",
    "concatenated_algebraic",
    "concatenated_initial_conditions",
]


class BaseModel:
    """
    Base model class for other models to extend.

    Attributes
    ----------
    name: str
        A string giving the name of the model.
    options: dict
        A dictionary of options to be passed to the model.
    submodels: dict
        A dictionary of submodels that the model is composed of.
    equations : :class:`pybamm.BaseEquations`
        The equations that define the model. When the model is first created, these are
        `pybamm.SymbolicEquations`. They are then converted to
        `pybamm.ParameterisedEquations` when parameter values are set
        and then `pybamm.DiscretisedEquations` when the model is discretised.
    use_jacobian : bool
        Whether to use the Jacobian when solving the model (default is True).
    convert_to_format : str
        Whether to convert the expression trees representing the rhs and
        algebraic equations, Jacobain (if using) and events into a different format:

        - None: keep PyBaMM expression tree structure.
        - "python": convert into pure python code that will calculate the result of \
        calling `evaluate(t, y)` on the given expression treeself.
        - "casadi": convert into CasADi expression tree, which then uses CasADi's \
        algorithm to calculate the Jacobian.

        Default is "casadi".
    """

    def __init__(self, name="Unnamed model"):
        self.name = name
        self._options = {"external submodels": []}

        # Initialise empty model
        self.submodels = {}
        self._equations = pybamm._SymbolicEquations()

        # Default behaviour is to use the jacobian
        self.use_jacobian = True
        self.convert_to_format = "casadi"

    @property
    def is_discretised(self):
        return isinstance(self._equations, pybamm._DiscretisedEquations)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __getattr__(self, name):
        if name in _EQUATION_ATTRIBUTES:
            return getattr(self._equations, name)
        else:
            return self.__getattribute__(name)

    def __setattr__(self, name, value):
        if name in _EQUATION_ATTRIBUTES:
            self._equations.__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def variable_names(self):
        return list(self._equations._variables.keys())

    @property
    def variables_and_events(self):
        """
        Returns variables and events in a single dictionary
        """
        return self._equations.variables_and_events

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, values):
        self._param = values

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options):
        self._options = options

    @property
    def timescale(self):
        """Timescale of model, to be used for non-dimensionalising time when solving"""
        return self._equations.timescale

    @timescale.setter
    def timescale(self, value):
        """Set the timescale"""
        self._equations.timescale = value

    @property
    def length_scales(self):
        "Length scales of model"
        return self._equations.length_scales

    @length_scales.setter
    def length_scales(self, values):
        "Set the length scale, converting any numbers to pybamm.Scalar"
        self._equations.length_scales = values

    @property
    def default_var_pts(self):
        return {}

    @property
    def default_geometry(self):
        return {}

    @property
    def default_submesh_types(self):
        return {}

    @property
    def default_spatial_methods(self):
        return {}

    @property
    def default_solver(self):
        """Return default solver based on whether model is ODE/DAE or algebraic"""
        if len(self.rhs) == 0 and len(self.algebraic) != 0:
            return pybamm.CasadiAlgebraicSolver()
        else:
            return pybamm.CasadiSolver(mode="safe")

    @property
    def default_quick_plot_variables(self):
        return None

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues({})

    @property
    def parameters(self):
        """Returns all the parameters in the model"""
        return self._equations.parameters

    @property
    def input_parameters(self):
        """Returns all the input parameters in the model"""
        return self._equations.input_parameters

    def print_parameter_info(self):
        self._equations.print_parameter_info()

    def new_copy(self, equations=None):
        """
        Creates a copy of the model, explicitly copying all the mutable attributes
        to avoid issues with shared objects.
        """
        new_model = copy.copy(self)
        if equations is None:
            equations = self._equations.copy()
        new_model._equations = equations
        return new_model

    def build_model(self):
        self._build_model()
        self._equations._built = True
        pybamm.logger.info("Finish building {}".format(self.name))

    def _build_model(self):
        # Check if already built
        if self._equations._built:
            raise pybamm.ModelError(
                """Model already built. If you are adding a new submodel, try using
                `model.update` instead."""
            )

        pybamm.logger.info("Start building {}".format(self.name))

        if self._equations._built_fundamental_and_external is False:
            self._equations.build_fundamental_and_external(self)

        self._equations.build_coupled_variables(self)

        self._equations.build_model_equations(self)

    def update(self, *submodels):
        self._equations.update(*submodels)

    def set_initial_conditions_from(self, solution, inplace=True, return_type="model"):
        """
        Update initial conditions with the final states from a Solution object or from
        a dictionary.
        This assumes that, for each variable in self.initial_conditions, there is a
        corresponding variable in the solution with the same name and size.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`, or dict
            The solution to use to initialize the model
        inplace : bool, optional
            Whether to modify the model inplace or create a new model (default True)
        return_type : str, optional
            Whether to return the model (default) or initial conditions ("ics")
        """
        initial_conditions = {}
        if isinstance(solution, pybamm.Solution):
            solution = solution.last_state
        for var in self.initial_conditions:
            if isinstance(var, pybamm.Variable):
                try:
                    final_state = solution[var.name]
                except KeyError as e:
                    raise pybamm.ModelError(
                        "To update a model from a solution, each variable in "
                        "model.initial_conditions must appear in the solution with "
                        "the same key as the variable name. In the solution provided, "
                        f"{e.args[0]}"
                    )
                if isinstance(solution, pybamm.Solution):
                    final_state = final_state.data
                if final_state.ndim == 0:
                    final_state_eval = np.array([final_state])
                elif final_state.ndim == 1:
                    final_state_eval = final_state[-1:]
                elif final_state.ndim == 2:
                    final_state_eval = final_state[:, -1]
                elif final_state.ndim == 3:
                    final_state_eval = final_state[:, :, -1].flatten(order="F")
                else:
                    raise NotImplementedError("Variable must be 0D, 1D, or 2D")
                initial_conditions[var] = pybamm.Vector(final_state_eval)
            elif isinstance(var, pybamm.Concatenation):
                children = []
                for child in var.orphans:
                    try:
                        final_state = solution[child.name]
                    except KeyError as e:
                        raise pybamm.ModelError(
                            "To update a model from a solution, each variable in "
                            "model.initial_conditions must appear in the solution with "
                            "the same key as the variable name. In the solution "
                            f"provided, {e.args[0]}"
                        )
                    if isinstance(solution, pybamm.Solution):
                        final_state = final_state.data
                    if final_state.ndim == 2:
                        final_state_eval = final_state[:, -1]
                    else:
                        raise NotImplementedError(
                            "Variable in concatenation must be 1D"
                        )
                    children.append(final_state_eval)
                initial_conditions[var] = pybamm.Vector(np.concatenate(children))

            else:
                raise NotImplementedError(
                    "Variable must have type 'Variable' or 'Concatenation'"
                )

        # Also update the concatenated initial conditions if the model is already
        # discretised
        if self.is_discretised:
            concatenated_initial_conditions = (
                self._equations._discretisation._concatenate_in_order(
                    initial_conditions
                )
            )
        else:
            concatenated_initial_conditions = None

        if return_type == "model":
            if inplace is True:
                model = self
            else:
                model = self.new_copy()

            model._equations._initial_conditions = pybamm.ReadOnlyDict(
                initial_conditions
            )
            model._equations._concatenated_initial_conditions = (
                concatenated_initial_conditions
            )
            return model
        elif return_type == "ics":
            return initial_conditions, concatenated_initial_conditions

    def check_well_posedness(self, post_discretisation=False):
        self._equations.check_well_posedness(post_discretisation)

    def info(self, symbol_name):
        """
        Provides helpful summary information for a symbol.

        Parameters
        ----------
        parameter_name : str
        """
        # Should we deprecate this? Not really sure how it's used?

        div = "-----------------------------------------"
        symbol = find_symbol_in_model(self, symbol_name)

        if symbol is None:
            return None

        print(div)
        print(symbol_name, "\n")
        print(type(symbol))

        if isinstance(symbol, pybamm.FunctionParameter):
            print("")
            print("Inputs:")
            symbol.print_input_names()

        print(div)

    def check_discretised_or_discretise_inplace_if_0D(self):
        """
        Discretise model if it isn't already discretised
        This only works with purely 0D models, as otherwise the mesh and spatial
        method should be specified by the user
        """
        if self.is_discretised is False:
            try:
                disc = pybamm.Discretisation()
                disc.process_model(self)
            except pybamm.DiscretisationError as e:
                raise pybamm.DiscretisationError(
                    "Cannot automatically discretise model, model should be "
                    "discretised before exporting casadi functions ({})".format(e)
                )

    def export_casadi_objects(self, variable_names, input_parameter_order=None):
        """
        Export the constituent parts of the model (rhs, algebraic, initial conditions,
        etc) as casadi objects.

        Parameters
        ----------
        variable_names : list
            Variables to be exported alongside the model structure
        input_parameter_order : list, optional
            Order in which the input parameters should be stacked. If None, the order
            returned by :meth:`BaseModel.input_parameters` is used

        Returns
        -------
        casadi_dict : dict
            Dictionary of {str: casadi object} pairs representing the model in casadi
            format
        """
        self.check_discretised_or_discretise_inplace_if_0D()

        # Create casadi functions for the model
        t_casadi = casadi.MX.sym("t")
        y_diff = casadi.MX.sym("y_diff", self.concatenated_rhs.size)
        y_alg = casadi.MX.sym("y_alg", self.concatenated_algebraic.size)
        y_casadi = casadi.vertcat(y_diff, y_alg)

        # Read inputs
        inputs_wrong_order = {}
        for input_param in self.input_parameters:
            name = input_param.name
            inputs_wrong_order[name] = casadi.MX.sym(name, input_param._expected_size)
        # Read external variables
        external_casadi = {}
        for external_varaiable in self.external_variables:
            name = external_varaiable.name
            ev_size = external_varaiable._evaluate_for_shape().shape[0]
            external_casadi[name] = casadi.MX.sym(name, ev_size)
        # Sort according to input_parameter_order
        if input_parameter_order is None:
            inputs = inputs_wrong_order
        else:
            inputs = {name: inputs_wrong_order[name] for name in input_parameter_order}
        # Set up external variables and inputs
        # Put external variables first like the integrator expects
        ext_and_in = {**external_casadi, **inputs}
        inputs_stacked = casadi.vertcat(*[p for p in ext_and_in.values()])

        # Convert initial conditions to casadi form
        y0 = self.concatenated_initial_conditions.to_casadi(
            t_casadi, y_casadi, inputs=inputs
        )
        x0 = y0[: self.concatenated_rhs.size]
        z0 = y0[self.concatenated_rhs.size :]

        # Convert rhs and algebraic to casadi form and calculate jacobians
        rhs = self.concatenated_rhs.to_casadi(t_casadi, y_casadi, inputs=ext_and_in)
        jac_rhs = casadi.jacobian(rhs, y_casadi)
        algebraic = self.concatenated_algebraic.to_casadi(
            t_casadi, y_casadi, inputs=inputs
        )
        jac_algebraic = casadi.jacobian(algebraic, y_casadi)

        # For specified variables, convert to casadi
        variables = {}
        for name in variable_names:
            var = self.variables[name]
            variables[name] = var.to_casadi(t_casadi, y_casadi, inputs=ext_and_in)

        casadi_dict = {
            "t": t_casadi,
            "x": y_diff,
            "z": y_alg,
            "inputs": inputs_stacked,
            "rhs": rhs,
            "algebraic": algebraic,
            "jac_rhs": jac_rhs,
            "jac_algebraic": jac_algebraic,
            "variables": variables,
            "x0": x0,
            "z0": z0,
        }

        return casadi_dict

    def generate(
        self, filename, variable_names, input_parameter_order=None, cg_options=None
    ):
        """
        Generate the model in C, using CasADi.

        Parameters
        ----------
        filename : str
            Name of the file to which to save the code
        variable_names : list
            Variables to be exported alongside the model structure
        input_parameter_order : list, optional
            Order in which the input parameters should be stacked. If None, the order
            returned by :meth:`BaseModel.input_parameters` is used
        cg_options : dict
            Options to pass to the code generator.
            See https://web.casadi.org/docs/#generating-c-code
        """
        model = self.export_casadi_objects(variable_names, input_parameter_order)

        # Read the exported objects
        t, x, z, p = model["t"], model["x"], model["z"], model["inputs"]
        x0, z0 = model["x0"], model["z0"]
        rhs, alg = model["rhs"], model["algebraic"]
        variables = model["variables"]
        jac_rhs, jac_alg = model["jac_rhs"], model["jac_algebraic"]

        # Create functions
        rhs_fn = casadi.Function("rhs_", [t, x, z, p], [rhs])
        alg_fn = casadi.Function("alg_", [t, x, z, p], [alg])
        jac_rhs_fn = casadi.Function("jac_rhs", [t, x, z, p], [jac_rhs])
        jac_alg_fn = casadi.Function("jac_alg", [t, x, z, p], [jac_alg])
        # Call these functions to initialize initial conditions
        # (initial conditions are not yet consistent at this stage)
        x0_fn = casadi.Function("x0", [p], [x0])
        z0_fn = casadi.Function("z0", [p], [z0])
        # Variables
        variables_stacked = casadi.vertcat(*variables.values())
        variables_fn = casadi.Function("variables", [t, x, z, p], [variables_stacked])

        # Write C files
        cg_options = cg_options or {}
        C = casadi.CodeGenerator(filename, cg_options)
        C.add(rhs_fn)
        C.add(alg_fn)
        C.add(jac_rhs_fn)
        C.add(jac_alg_fn)
        C.add(x0_fn)
        C.add(z0_fn)
        C.add(variables_fn)
        C.generate()

    def generate_julia_diffeq(
        self,
        input_parameter_order=None,
        get_consistent_ics_solver=None,
        dae_type="semi-explicit",
        **kwargs,
    ):
        """
        Generate a Julia representation of the model, ready to be solved by Julia's
        DifferentialEquations library.

        Parameters
        ----------
        input_parameter_order : list, optional
            Order in which input parameters will be provided when solving the model
        get_consistent_ics_solver : pybamm solver, optional
            Solver to use to get consistent initial conditions. If None, the initial
            guesses for boundary conditions (non-consistent) are used.
        dae_type : str, optional
            How to write the DAEs. Options are "semi-explicit" (default) or "implicit".

        Returns
        -------
        eqn_str : str
            The Julia-compatible equations for the model in string format,
            to be evaluated by eval(Meta.parse(...))
        ics_str : str
            The Julia-compatible initial conditions for the model in string format,
            to be evaluated by eval(Meta.parse(...))
        """
        self.check_discretised_or_discretise_inplace_if_0D()

        name = self.name.replace(" ", "_")

        if self.algebraic == {}:
            # ODE model: form dy[] = ...
            eqn_str = pybamm.get_julia_function(
                self.concatenated_rhs,
                funcname=name,
                input_parameter_order=input_parameter_order,
                **kwargs,
            )
        else:
            if dae_type == "semi-explicit":
                len_rhs = None
            else:
                len_rhs = self.concatenated_rhs.size
            # DAE model: form out[] = ... - dy[]
            eqn_str = pybamm.get_julia_function(
                pybamm.numpy_concatenation(
                    self.concatenated_rhs, self.concatenated_algebraic
                ),
                funcname=name,
                input_parameter_order=input_parameter_order,
                len_rhs=len_rhs,
                **kwargs,
            )

        if get_consistent_ics_solver is None or self.algebraic == {}:
            ics = self.concatenated_initial_conditions
        else:
            get_consistent_ics_solver.set_up(self)
            get_consistent_ics_solver._set_initial_conditions(self, {}, False)
            ics = pybamm.Vector(self.y0.full())

        ics_str = pybamm.get_julia_function(
            ics,
            funcname=name + "_u0",
            input_parameter_order=input_parameter_order,
            **kwargs,
        )
        # Change the string to a form for u0
        ics_str = ics_str.replace("(dy, y, p, t)", "(u0, p)")
        ics_str = ics_str.replace("dy", "u0")

        return eqn_str, ics_str

    def latexify(self, filename=None, newline=True):
        # For docstring, see pybamm.expression_tree.operations.latexify.Latexify
        return Latexify(self, filename, newline).latexify()

    # Set :meth:`latexify` docstring from :class:`Latexify`
    latexify.__doc__ = Latexify.__doc__

    def process_parameters_and_discretise(self, symbol, parameter_values, disc):
        """
        Process parameters and discretise a symbol using supplied parameter values
        and discretisation. Note: care should be taken if using spatial operators
        on dimensional symbols. Operators in pybamm are written in non-dimensional
        form, so may need to be scaled by the appropriate length scale. It is
        recommended to use this method on non-dimensional symbols.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol to be processed
        parameter_values : :class:`pybamm.ParameterValues`
            The parameter values to use during processing
        disc : :class:`pybamm.Discretisation`
            The discrisation to use

        Returns
        -------
        :class:`pybamm.Symbol`
            Processed symbol
        """
        # Set y slices
        if disc.y_slices == {}:
            variables = list(self.rhs.keys()) + list(self.algebraic.keys())
            disc.set_variable_slices(variables)

        # Set boundary conditions (also requires setting parameter values)
        if disc.bcs == {}:
            self.boundary_conditions = parameter_values.process_boundary_conditions(
                self
            )
            disc.bcs = disc.process_boundary_conditions(self)

        # Process
        param_symbol = parameter_values.process_symbol(symbol)
        disc_symbol = disc.process_symbol(param_symbol)

        return disc_symbol


# helper functions for finding symbols
def find_symbol_in_tree(tree, name):
    if name == tree.name:
        return tree
    elif len(tree.children) > 0:
        for child in tree.children:
            child_return = find_symbol_in_tree(child, name)
            if child_return is not None:
                return child_return


def find_symbol_in_dict(dic, name):
    for tree in dic.values():
        tree_return = find_symbol_in_tree(tree, name)
        if tree_return is not None:
            return tree_return


def find_symbol_in_model(model, name):
    dics = [model.rhs, model.algebraic, model.variables]
    for dic in dics:
        dic_return = find_symbol_in_dict(dic, name)
        if dic_return is not None:
            return dic_return
