#
# Base model class
#
import pybamm

import numbers
import os
import warnings


class BaseModel(object):
    """Base model class for other models to extend.

    Attributes
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    algebraic: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the algebraic equations. The algebraic expressions are assumed to equate
        to zero. Note that all the variables in the model must exist in the keys of
        `rhs` or `algebraic`.
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions for the state variables y. The initial conditions for
        algebraic variables are provided as initial guesses to a root finding algorithm
        that calculates consistent initial conditions.
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables
    events: list
        A list of events that should cause the solver to terminate (e.g. concentration
        goes negative)

    """

    def __init__(self):
        # Default name
        self.name = "Unnamed Model"

        # Initialise empty model
        self._rhs = {}
        self._algebraic = {}
        self._initial_conditions = {}
        self._boundary_conditions = {}
        self._variables = {}
        self._events = []
        self._concatenated_rhs = None
        self._concatenated_initial_conditions = None
        self._mass_matrix = None
        self._jacobian = None

        # Default behaviour is to use the jacobian and simplify
        self.use_jacobian = True
        self.use_simplify = True

        # Default behaviour: no capacitance in the model
        self._use_capacitance = False

    def _set_dict(self, dict, name):
        """
        Convert any scalar equations in dict to 'pybamm.Scalar'
        and check that domains are consistent
        """
        # Convert any numbers to a pybamm.Scalar
        for var, eqn in dict.items():
            if isinstance(eqn, numbers.Number):
                dict[var] = pybamm.Scalar(eqn)

        if not all(
            [
                variable.domain == equation.domain
                or variable.domain == []
                or equation.domain == []
                for variable, equation in dict.items()
            ]
        ):
            raise pybamm.DomainError(
                "variable and equation in '{}' must have the same domain".format(name)
            )

        return dict

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def rhs(self):
        return self._rhs

    @rhs.setter
    def rhs(self, rhs):
        self._rhs = self._set_dict(rhs, "rhs")

    @property
    def algebraic(self):
        return self._algebraic

    @algebraic.setter
    def algebraic(self, algebraic):
        self._algebraic = self._set_dict(algebraic, "algebraic")

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @initial_conditions.setter
    def initial_conditions(self, initial_conditions):
        self._initial_conditions = self._set_dict(
            initial_conditions, "initial_conditions"
        )

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        # Convert any numbers to a pybamm.Scalar
        for var, bcs in boundary_conditions.items():
            for side, bc in bcs.items():
                if isinstance(bc[0], numbers.Number):
                    # typ is the type of the bc, e.g. "Dirichlet" or "Neumann"
                    eqn, typ = boundary_conditions[var][side]
                    boundary_conditions[var][side] = (pybamm.Scalar(eqn), typ)
                # Check types
                if bc[1] not in ["Dirichlet", "Neumann"]:
                    raise pybamm.ModelError(
                        """
                        boundary condition types must be Dirichlet or Neumann, not '{}'
                        """.format(
                            bc[1]
                        )
                    )
        self._boundary_conditions = boundary_conditions

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        self._variables = variables

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, events):
        self._events = events

    @property
    def concatenated_rhs(self):
        return self._concatenated_rhs

    @concatenated_rhs.setter
    def concatenated_rhs(self, concatenated_rhs):
        self._concatenated_rhs = concatenated_rhs

    @property
    def concatenated_initial_conditions(self):
        return self._concatenated_initial_conditions

    @concatenated_initial_conditions.setter
    def concatenated_initial_conditions(self, concatenated_initial_conditions):
        self._concatenated_initial_conditions = concatenated_initial_conditions

    @property
    def mass_matrix(self):
        return self._mass_matrix

    @mass_matrix.setter
    def mass_matrix(self, mass_matrix):
        self._mass_matrix = mass_matrix

    @property
    def jacobian(self):
        return self._jacobian

    @jacobian.setter
    def jacobian(self, jacobian):
        self._jacobian = jacobian

    @property
    def use_capacitance(self):
        return self._use_capacitance

    @property
    def bc_options(self):
        return self._bc_options

    @property
    def set_of_parameters(self):
        return self._set_of_parameters

    def __getitem__(self, key):
        return self.rhs[key]

    def update(self, *submodels):
        """
        Update model to add new physics from submodels

        Parameters
        ----------
        submodel : iterable of :class:`pybamm.BaseModel`
            The submodels from which to create new model
        """
        for submodel in submodels:

            # check and then update dicts
            self.check_and_combine_dict(self._rhs, submodel.rhs)
            self.check_and_combine_dict(self._algebraic, submodel.algebraic)
            self.check_and_combine_dict(
                self._initial_conditions, submodel.initial_conditions
            )
            self.check_and_combine_dict(
                self._boundary_conditions, submodel.boundary_conditions
            )
            self.variables.update(submodel.variables)  # keys are strings so no check
            self._events.extend(submodel.events)

    def check_and_combine_dict(self, dict1, dict2):
        # check that the key ids are distinct
        ids1 = set(x.id for x in dict1.keys())
        ids2 = set(x.id for x in dict2.keys())
        if len(ids1.intersection(ids2)) != 0:
            raise pybamm.ModelError("Submodel incompatible: duplicate variables")
        dict1.update(dict2)

    def check_well_posedness(self, post_discretisation=False):
        """
        Check that the model is well-posed by executing the following tests:
        - Model is not over- or underdetermined, by comparing keys and equations in rhs
        and algebraic. Overdetermined if more equations than variables, underdetermined
        if more variables than equations.
        - There is an initial condition in self.initial_conditions for each
        variable/equation pair in self.rhs
        - There are appropriate boundary conditions in self.boundary_conditions for each
        variable/equation pair in self.rhs and self.algebraic

        Parameters
        ----------
        post_discretisation : boolean
            A flag indicating tests to be skipped after discretisation
        """
        self.check_well_determined(post_discretisation)
        self.check_algebraic_equations(post_discretisation)
        self.check_ics_bcs()
        self.check_variables()

    def check_well_determined(self, post_discretisation):
        """ Check that the model is not under- or over-determined. """
        # Equations (differential and algebraic)
        # Get all the variables from differential and algebraic equations
        vars_in_rhs_keys = set()
        vars_in_algebraic_keys = set()
        vars_in_eqns = set()
        # Get all variables ids from rhs and algebraic keys and equations
        # For equations we look through the whole expression tree.
        # "Variables" can be Concatenations so we also have to look in the whole
        # expression tree
        for var, eqn in self.rhs.items():
            vars_in_rhs_keys.update(
                [x.id for x in var.pre_order() if isinstance(x, pybamm.Variable)]
            )
            vars_in_eqns.update(
                [x.id for x in eqn.pre_order() if isinstance(x, pybamm.Variable)]
            )
        for var, eqn in self.algebraic.items():
            vars_in_algebraic_keys.update(
                [x.id for x in var.pre_order() if isinstance(x, pybamm.Variable)]
            )
            vars_in_eqns.update(
                [x.id for x in eqn.pre_order() if isinstance(x, pybamm.Variable)]
            )
        # If any keys are repeated between rhs and algebraic then the model is
        # overdetermined
        if not set(vars_in_rhs_keys).isdisjoint(vars_in_algebraic_keys):
            raise pybamm.ModelError("model is overdetermined (repeated keys)")
        # If any algebraic keys don't appear in the eqns then the model is
        # overdetermined (but rhs keys can be absent from the eqns, e.g. dcdt = -1 is
        # fine)
        # Skip this step after discretisation, as any variables in the equations will
        # have been discretised to slices but keys will still be variables
        extra_algebraic_keys = vars_in_algebraic_keys.difference(vars_in_eqns)
        if extra_algebraic_keys and not post_discretisation:
            raise pybamm.ModelError("model is overdetermined (extra algebraic keys)")
        # If any variables in the equations don't appear in the keys then the model is
        # underdetermined
        vars_in_keys = vars_in_rhs_keys.union(vars_in_algebraic_keys)
        extra_variables = vars_in_eqns.difference(vars_in_keys)
        if extra_variables:
            raise pybamm.ModelError("model is underdetermined (too many variables)")

    def check_algebraic_equations(self, post_discretisation):
        """
        Check that the algebraic equations are well-posed.
        Before discretisation, each algebraic equation key must appear in the equation
        After discretisation, there must be at least one StateVector in each algebraic
        equation
        """
        if not post_discretisation:
            # After the model has been defined, each algebraic equation key should
            # appear in that algebraic equation
            for var, eqn in self.algebraic.items():
                if not any(x.id == var.id for x in eqn.pre_order()):
                    raise pybamm.ModelError(
                        "each variable in the algebraic eqn keys must appear in the eqn"
                    )
        else:
            # variables in keys don't get discretised so they will no longer match
            # with the state vectors in the algebraic equations. Instead, we check
            # that each algebraic equation contains some StateVector
            for eqn in self.algebraic.values():
                if not any(isinstance(x, pybamm.StateVector) for x in eqn.pre_order()):
                    raise pybamm.ModelError(
                        "each algebraic equation must contain at least one StateVector"
                    )

    def check_ics_bcs(self):
        """ Check that the initial and boundary conditions are well-posed. """
        # Initial conditions
        for var in self.rhs.keys():
            if var not in self.initial_conditions.keys():
                raise pybamm.ModelError(
                    """no initial condition given for variable '{}'""".format(var)
                )

        # Boundary conditions
        for var, eqn in {**self.rhs, **self.algebraic}.items():
            if eqn.has_symbol_of_class((pybamm.Gradient, pybamm.Divergence)):
                # Variable must be in the boundary conditions
                if not any(
                    var.id == x.id
                    for symbol in self.boundary_conditions.keys()
                    for x in symbol.pre_order()
                ):
                    raise pybamm.ModelError(
                        """
                        no boundary condition given for variable '{}' with equation '{}'
                        """.format(
                            var, eqn
                        )
                    )

    def check_variables(self):
        """ Chec that the right variables are provided. """
        missing_vars = []
        for output, expression in self._variables.items():
            if expression is None:
                missing_vars.append(output)
        if len(missing_vars) > 0:
            warnings.warn(
                "the standard output variable(s) '{}' have not been supplied. "
                "These may be required for testing or comparison with other "
                "models.".format(missing_vars),
                pybamm.ModelWarning,
                stacklevel=2,
            )
            # Remove missing entries
            for output in missing_vars:
                del self._variables[output]


class StandardBatteryBaseModel(BaseModel):
    """
    Base model class with some default settings and required variables

    **Extends:** :class:`StandardBatteryBaseModel`
    """

    def __init__(self):
        super().__init__()
        self.set_standard_output_variables()

    @property
    def default_parameter_values(self):
        # Default parameter values, geometry, submesh, spatial methods and solver
        # Lion parameters left as default parameter set for tests
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lithium-ion")
        return pybamm.ParameterValues(
            os.path.join(
                input_path, "mcmb2528_lif6-in-ecdmc_lico2_parameters_Dualfoil.csv"
            ),
            {
                "Typical current [A]": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
                "Electrolyte diffusivity": os.path.join(
                    input_path, "electrolyte_diffusivity_Capiglia1999.py"
                ),
                "Electrolyte conductivity": os.path.join(
                    input_path, "electrolyte_conductivity_Capiglia1999.py"
                ),
                "Negative electrode OCV": os.path.join(
                    input_path, "graphite_mcmb2528_ocp_Dualfoil.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lico2_ocp_Dualfoil.py"
                ),
                "Negative electrode diffusivity": os.path.join(
                    input_path, "graphite_mcmb2528_diffusivity_Dualfoil.py"
                ),
                "Positive electrode diffusivity": os.path.join(
                    input_path, "lico2_diffusivity_Dualfoil.py"
                ),
            },
        )

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1+1D micro")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {
            var.x_n: 40,
            var.x_s: 25,
            var.x_p: 35,
            var.r_n: 10,
            var.r_p: 10,
            var.z: 10,
        }

    @property
    def default_submesh_types(self):
        return {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.Uniform1DSubMesh,
        }

    @property
    def default_spatial_methods(self):
        return {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteVolume,
        }

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        try:
            default_solver = pybamm.ScikitsOdeSolver()
        except ImportError:
            default_solver = pybamm.ScipySolver()

        return default_solver

    @property
    def default_bc_options(self):
        return {"dimensionality": 1}

    def set_standard_output_variables(self):
        # Standard output variables
        # Interfacial current
        self.variables.update(
            {
                "Negative electrode current density": None,
                "Positive electrode current density": None,
                "Electrolyte current density": None,
                "Interfacial current density": None,
                "Exchange-current density": None,
            }
        )

        self.variables.update(
            {
                "Negative electrode current density [A.m-2]": None,
                "Positive electrode current density [A.m-2]": None,
                "Electrolyte current density [A.m-2]": None,
                "Interfacial current density [A.m-2]": None,
                "Exchange-current density [A.m-2]": None,
            }
        )
        # Voltage
        self.variables.update(
            {
                "Negative electrode open circuit potential": None,
                "Positive electrode open circuit potential": None,
                "Average negative electrode open circuit potential": None,
                "Average positive electrode open circuit potential": None,
                "Average open circuit voltage": None,
                "Measured open circuit voltage": None,
                "Terminal voltage": None,
            }
        )

        self.variables.update(
            {
                "Negative electrode open circuit potential [V]": None,
                "Positive electrode open circuit potential [V]": None,
                "Average negative electrode open circuit potential [V]": None,
                "Average positive electrode open circuit potential [V]": None,
                "Average open circuit voltage [V]": None,
                "Measured open circuit voltage [V]": None,
                "Terminal voltage [V]": None,
            }
        )

        # Overpotentials
        self.variables.update(
            {
                "Negative reaction overpotential": None,
                "Positive reaction overpotential": None,
                "Average negative reaction overpotential": None,
                "Average positive reaction overpotential": None,
                "Average reaction overpotential": None,
                "Average electrolyte overpotential": None,
                "Average solid phase ohmic losses": None,
            }
        )

        self.variables.update(
            {
                "Negative reaction overpotential [V]": None,
                "Positive reaction overpotential [V]": None,
                "Average negative reaction overpotential [V]": None,
                "Average positive reaction overpotential [V]": None,
                "Average reaction overpotential [V]": None,
                "Average electrolyte overpotential [V]": None,
                "Average solid phase ohmic losses [V]": None,
            }
        )
        # Concentration
        self.variables.update(
            {
                "Electrolyte concentration": None,
                "Electrolyte concentration [mol.m-3]": None,
            }
        )

        # Potential
        self.variables.update(
            {
                "Negative electrode potential [V]": None,
                "Positive electrode potential [V]": None,
                "Electrolyte potential [V]": None,
            }
        )

        # Current
        icell = pybamm.electrical_parameters.current_with_time
        icell_dim = pybamm.electrical_parameters.dimensional_current_density_with_time
        I = pybamm.electrical_parameters.dimensional_current_with_time
        self.variables.update(
            {
                "Total current density": icell,
                "Total current density [A.m-2]": icell_dim,
                "Current [A]": I,
            }
        )
        # Time
        self.variables.update({"Time": pybamm.t})
        # x-position
        var = pybamm.standard_spatial_vars
        L_x = pybamm.geometric_parameters.L_x
        self.variables.update(
            {
                "x": var.x,
                "x [m]": var.x * L_x,
                "x_n": var.x_n,
                "x_n [m]": var.x_n * L_x,
                "x_s": var.x_s,
                "x_s [m]": var.x_s * L_x,
                "x_p": var.x_p,
                "x_p [m]": var.x_p * L_x,
            }
        )


class SubModel(StandardBatteryBaseModel):
    def __init__(self, set_of_parameters):
        super().__init__()
        self._set_of_parameters = set_of_parameters
        # Initialise empty variables (to avoid overwriting with 'None')
        self.variables = {}


class LeadAcidBaseModel(StandardBatteryBaseModel):
    """
    Overwrites default parameters from Base Model with default parameters for
    lead-acid models

    **Extends:** :class:`StandardBatteryBaseModel`

    """

    def __init__(self):
        super().__init__()

    @property
    def default_parameter_values(self):
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
        return pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current [A]": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
                "Electrolyte diffusivity": os.path.join(
                    input_path, "electrolyte_diffusivity_Gu1997.py"
                ),
                "Electrolyte conductivity": os.path.join(
                    input_path, "electrolyte_conductivity_Gu1997.py"
                ),
                "Electrolyte viscosity": os.path.join(
                    input_path, "electrolyte_viscosity_Chapman1968.py"
                ),
                "Darken thermodynamic factor": os.path.join(
                    input_path, "darken_thermodynamic_factor_Chapman1968.py"
                ),
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
            },
        )

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro")

    def set_standard_output_variables(self):
        super().set_standard_output_variables()
        # Standard time variable
        time_scale = pybamm.standard_parameters_lead_acid.tau_discharge
        I = pybamm.electrical_parameters.dimensional_current_with_time
        self.variables.update(
            {
                "Time [s]": pybamm.t * time_scale,
                "Time [min]": pybamm.t * time_scale / 60,
                "Time [h]": pybamm.t * time_scale / 3600,
                "Discharge capacity [A.h]": I * pybamm.t * time_scale / 3600,
            }
        )


class LithiumIonBaseModel(StandardBatteryBaseModel):
    """
    Overwrites default parameters from Base Model with default parameters for
    lithium-ion models

    **Extends:** :class:`StandardBatteryBaseModel`

    """

    def __init__(self):
        super().__init__()

    def set_standard_output_variables(self):
        super().set_standard_output_variables()
        # Additional standard output variables
        # Time
        time_scale = pybamm.standard_parameters_lithium_ion.tau_discharge
        I = pybamm.electrical_parameters.dimensional_current_with_time
        self.variables.update(
            {
                "Time [s]": pybamm.t * time_scale,
                "Time [min]": pybamm.t * time_scale / 60,
                "Time [h]": pybamm.t * time_scale / 3600,
                "Discharge capacity [A.h]": I * pybamm.t * time_scale / 3600,
            }
        )

        # Particle concentration and position
        self.variables.update(
            {
                "Negative particle concentration": None,
                "Positive particle concentration": None,
                "Negative particle surface concentration": None,
                "Positive particle surface concentration": None,
                "Negative particle concentration [mol.m-3]": None,
                "Positive particle concentration [mol.m-3]": None,
                "Negative particle surface concentration [mol.m-3]": None,
                "Positive particle surface concentration [mol.m-3]": None,
            }
        )
        var = pybamm.standard_spatial_vars
        param = pybamm.geometric_parameters
        self.variables.update(
            {
                "r_n": var.r_n,
                "r_n [m]": var.r_n * param.R_n,
                "r_p": var.r_p,
                "r_p [m]": var.r_p * param.R_p,
            }
        )
