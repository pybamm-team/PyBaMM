#
# Interface for discretisation
#
import functools
import itertools
from collections import OrderedDict, defaultdict
from collections.abc import Callable

import numpy as np
from scipy.sparse import block_diag, csr_matrix

import pybamm
from pybamm.models.base_model import ModelSolutionObservability


def has_bc_of_form(symbol, side, bcs, form):
    return (symbol in bcs) and (bcs[symbol][side][1] == form)


# legacy current-collector tab BC side names, converted to left/right for
# 1D meshes by Discretisation.check_tab_conditions
LEGACY_TAB_SIDES = frozenset({"negative tab", "positive tab", "no tab"})


class Discretisation:
    """The discretisation class, with methods to process a model and replace
    Spatial Operators with Matrices and Variables with StateVectors

    Parameters
    ----------
    mesh : pybamm.Mesh
        contains all submeshes to be used on each domain
    spatial_methods : dict
        a dictionary of the spatial methods to be used on each
        domain. The keys correspond to the model domains and the
        values to the spatial method.
    check_model : bool, optional
            If True, model checks are performed after discretisation. For large
            systems these checks can be slow, so can be skipped by setting this
            option to False. When developing, testing or debugging it is recommended
            to leave this option as True as it may help to identify any errors.
            Default is True.
    remove_independent_variables_from_rhs : bool, optional
        If True, model checks to see whether any variables from the RHS are used
        in any other equation. If a variable meets all of the following criteria
        (not used anywhere in the model, len(rhs)>1), then the variable
        is moved to be explicitly integrated when called by the solution object.
        Default is False.
    resolve_coupled_variables : bool, optional
        If True, resolve CoupledVariables in rhs, algebraic, initial_conditions,
        and boundary_conditions before processing. Default is False.
    """

    def __init__(
        self,
        mesh=None,
        spatial_methods=None,
        check_model=True,
        remove_independent_variables_from_rhs=False,
        resolve_coupled_variables=False,
    ):
        self._mesh = mesh
        if mesh is None:
            self._spatial_methods = {}
        else:
            # Unpack macroscale to the constituent subdomains
            if "macroscale" in spatial_methods:
                method = spatial_methods["macroscale"]
                spatial_methods["negative electrode"] = method
                spatial_methods["separator"] = method
                spatial_methods["positive electrode"] = method

            self._spatial_methods = spatial_methods
            for domain, method in self._spatial_methods.items():
                method.build(mesh)
                # Check zero-dimensional methods are only applied to zero-dimensional
                # meshes
                if isinstance(
                    method, pybamm.ZeroDimensionalSpatialMethod
                ) and not isinstance(mesh[domain], pybamm.SubMesh0D):
                    raise pybamm.DiscretisationError(
                        "Zero-dimensional spatial method for the "
                        f"{domain} domain requires a zero-dimensional submesh"
                    )

        self._bcs = {}
        self.y_slices = {}
        self._discretised_symbols = {}
        self._check_model_flag = check_model
        self._remove_independent_variables_from_rhs_flag = (
            remove_independent_variables_from_rhs
        )
        self._resolve_coupled_variables = resolve_coupled_variables

    @property
    def mesh(self):
        return self._mesh

    @property
    def y_slices(self):
        return self._y_slices

    @y_slices.setter
    def y_slices(self, value):
        if not isinstance(value, dict):
            raise TypeError(f"y_slices should be dict, not {type(value)}")

        self._y_slices = value

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @property
    def bcs(self):
        return self._bcs

    @bcs.setter
    def bcs(self, value):
        self._bcs = value
        # reset discretised_symbols
        self._discretised_symbols = {}

    def process_model(
        self,
        model,
        inplace=True,
        delayed_variable_processing=None,
    ):
        """
        Discretise a model. Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})
        inplace : bool, optional
            If True, discretise the model in place. Otherwise, return a new
            discretised model. Default is True.
        delayed_variable_processing: bool, optional
            If True, make variable processing a post-processing step.
            Default is False.

        Returns
        -------
        model_disc : :class:`pybamm.BaseModel`
            The discretised model. Note that if ``inplace`` is True, model will
            have also been discretised in place so model == model_disc. If
            ``inplace`` is False, model != model_disc

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic = {}` and
            `model.variables = {}`)

        """
        if model.is_discretised is True:
            raise pybamm.ModelError(
                "Cannot re-discretise a model. "
                "Set 'inplace=False' when first discretising a model to then be able "
                "to discretise it more times (e.g. for convergence studies)."
            )
        if delayed_variable_processing is None:
            delayed_variable_processing = False

        pybamm.logger.info(f"Start discretising {model.name}")

        # Make sure model isn't empty
        if (
            len(model.rhs) == 0
            and len(model.algebraic) == 0
            and len(model.variables) == 0
        ):
            raise pybamm.ModelError("Cannot discretise empty model")
        # Check well-posedness to avoid obscure errors
        model.check_well_posedness()

        # Prepare discretisation
        # set variables (we require the full variable not just id)

        # Search Equations for Independence
        if self._remove_independent_variables_from_rhs_flag:
            model = self.remove_independent_variables_from_rhs(model)
        # Find those RHS's that are constant
        if self.spatial_methods == {}:
            for var in itertools.chain(model.rhs, model.algebraic):
                if var.domain != []:
                    raise pybamm.DiscretisationError(
                        "Spatial method has not been given "
                        f"for variable {var.name} with domain {var.domain}"
                    )

        # Set the y split for variables
        pybamm.logger.verbose(f"Set variable slices for {model.name}")
        self.set_variable_slices(itertools.chain(model.rhs, model.algebraic))

        # set boundary conditions (only need key ids for boundary_conditions)
        pybamm.logger.verbose(f"Discretise boundary conditions for {model.name}")
        self._bcs = self.process_boundary_conditions(model)
        pybamm.logger.verbose(f"Set internal boundary conditions for {model.name}")
        self.set_internal_boundary_conditions(model)

        # set up inplace vs not inplace
        if inplace:
            # any changes to model_disc attributes will change model attributes
            # since they point to the same object
            model_disc = model
        else:
            # create a copy of the original model
            model_disc = model.new_copy()

        if self._resolve_coupled_variables:
            self._resolve_coupled_variables_in_model(model)

        # Keep a record of y_slices in the model
        model_disc.y_slices = self.y_slices_explicit
        # Keep a record of the bounds in the model
        model_disc.bounds = self.bounds

        model_disc.bcs = self.bcs

        pybamm.logger.verbose(f"Discretise initial conditions for {model.name}")
        ics, concat_ics = self.process_initial_conditions(model)
        model_disc.initial_conditions = ics
        model_disc.concatenated_initial_conditions = concat_ics

        # Discretise variables (applying boundary conditions)
        # Note that we **do not** discretise the keys of model.rhs,
        # model.initial_conditions and model.boundary_conditions
        pybamm.logger.verbose(f"Discretise variables for {model.name}")

        # pre-process variables so that all state variables are included
        # This is the ONLY place where model.variables should be modified
        pre_processed_variables = self._pre_process_variables(
            model.variables,
            model.initial_conditions,
            comparable_variables=model.variables_matching_keys(),
        )
        model_disc.variables = pybamm.FuzzyDict(pre_processed_variables)

        if not delayed_variable_processing:
            # Process variables and store in _variables_processed
            variables_to_process = model.get_processed_variables_dict()
            for name, var in pre_processed_variables.items():
                if name not in variables_to_process:
                    # New variable (e.g., added by _pre_process_variables)
                    variables_to_process[name] = var
            processed_variables = self.process_dict(variables_to_process)
            model_disc.update_processed_variables(processed_variables)

        # Process parabolic and elliptic equations
        pybamm.logger.verbose(f"Discretise model equations for {model.name}")
        rhs, concat_rhs, alg, concat_alg = self.process_rhs_and_algebraic(model)
        model_disc.rhs, model_disc.concatenated_rhs = rhs, concat_rhs
        model_disc.algebraic, model_disc.concatenated_algebraic = alg, concat_alg

        # Save length of rhs and algebraic
        model_disc.len_rhs = model_disc.concatenated_rhs.size
        model_disc.len_alg = model_disc.concatenated_algebraic.size
        model_disc.len_rhs_and_alg = model_disc.len_rhs + model_disc.len_alg

        # Process events
        processed_events = []
        pybamm.logger.verbose(f"Discretise events for {model.name}")
        for event in model.events:
            pybamm.logger.debug(f"Discretise event '{event.name}'")
            processed_event = pybamm.Event(
                event.name, self.process_symbol(event.expression), event.event_type
            )
            processed_events.append(processed_event)
        model_disc.events = processed_events

        # Create mass matrix
        pybamm.logger.verbose(f"Create mass matrix for {model.name}")
        model_disc.mass_matrix = self.create_mass_matrix(model_disc)

        # Save geometry
        pybamm.logger.verbose(f"Save geometry for {model.name}")
        model_disc._geometry = getattr(self.mesh, "_geometry", None)

        # Check that resulting model makes sense
        if self._check_model_flag:
            pybamm.logger.verbose(f"Performing model checks for {model.name}")
            self.check_model(model_disc)

        pybamm.logger.info(f"Finish discretising {model.name}")

        # Re-discretising the model means it can no longer safely process symbols.
        # Not currently reachable, but keeping the check for safety
        if model.is_discretised:
            pybamm.logger.debug(
                f"Model '{model.name}' is being re-discretised, "
                "which makes it unable to process symbols using `model.process_symbol`"
            )  # pragma: no cover
            model_disc.disable_symbol_processing(
                ModelSolutionObservability.REDISCRETISED_MODEL
            )  # pragma: no cover

        pybamm.logger.debug("Attaching the `discretisation` to the `symbol_processor`")
        model_disc.symbol_processor.discretisation = self

        model_disc.is_discretised = True
        return model_disc

    def _resolve_coupled_variables_in_model(self, model):
        """Resolve CoupledVariables in rhs, algebraic, initial_conditions, and boundary_conditions."""

        resolve_symbol = model._resolve_coupled_variables

        for var, expr in model.rhs.items():
            resolved = resolve_symbol(expr)
            if resolved is not expr:
                model.rhs[var] = resolved

        for var, expr in model.algebraic.items():
            resolved = resolve_symbol(expr)
            if resolved is not expr:
                model.algebraic[var] = resolved

        for var, expr in model.initial_conditions.items():
            resolved = resolve_symbol(expr)
            if resolved is not expr:
                model.initial_conditions[var] = resolved

        for var, bcs in model.boundary_conditions.items():
            for side, (expr, bc_type) in bcs.items():
                resolved = resolve_symbol(expr)
                if resolved is not expr:
                    model.boundary_conditions[var][side] = (resolved, bc_type)

    def set_variable_slices(self, variables):
        """
        Sets the slicing for variables.

        Parameters
        ----------
        variables : iterable of :class:`pybamm.Variables`
            The variables for which to set slices
        """
        # Set up y_slices and bounds
        y_slices = defaultdict(list)
        y_slices_explicit = defaultdict(list)
        start = 0
        end = 0
        lower_bounds = []
        upper_bounds = []

        # Iterate through unpacked variables, adding appropriate slices to y_slices
        for variable in variables:
            if variable in y_slices:
                continue
            # Add up the size of all the domains in variable.domain
            if isinstance(variable, pybamm.ConcatenationVariable):
                spatial_method = self.spatial_methods[variable.domain[0]]
                dimension = spatial_method.mesh[variable.domain[0]].dimension
                start_ = start
                children = variable.children
                meshes = OrderedDict()
                lr_points = OrderedDict()
                tb_points = OrderedDict()
                for child in children:
                    meshes[child] = [spatial_method.mesh[dom] for dom in child.domain]
                    if dimension == 2:
                        lr_points[child] = sum(
                            spatial_method.mesh[dom].npts_lr for dom in child.domain
                        )
                        tb_points[child] = sum(
                            spatial_method.mesh[dom].npts_tb for dom in child.domain
                        )
                sec_points = spatial_method._get_auxiliary_domain_repeats(
                    variable.domains
                )
                for _ in range(sec_points):
                    start_this_child = start_
                    for child, mesh in meshes.items():
                        for domain_mesh in mesh:
                            end += domain_mesh.npts_for_broadcast_to_nodes
                        # Add to slices
                        if dimension == 2:
                            other_children = set(meshes.keys()) - {child}
                            num_pts_to_skip = sum(
                                lr_points[other_child] for other_child in other_children
                            )
                            for row in range(tb_points[child]):
                                start_this_row = (
                                    start_this_child
                                    + (lr_points[child] + num_pts_to_skip) * row
                                )
                                end_this_row = start_this_row + lr_points[child]
                                y_slices[child].append(
                                    slice(start_this_row, end_this_row)
                                )
                                y_slices_explicit[child].append(
                                    slice(start_this_row, end_this_row)
                                )
                            start_this_child += lr_points[child]
                        else:
                            y_slices[child].append(slice(start_, end))
                            y_slices_explicit[child].append(slice(start_, end))
                        # Increment start_
                        start_ = end
            else:
                end += self._get_variable_size(variable)

            # Add to slices
            y_slices[variable].append(slice(start, end))
            y_slices_explicit[variable].append(slice(start, end))

            # Add to bounds
            def evaluate_bound(bound, side):
                if bound.has_symbol_of_classes(pybamm.InputParameter):
                    if side == "lower":
                        return -np.inf
                    elif side == "upper":
                        return np.inf
                else:
                    return bound.evaluate()

            # symbols without physical bounds (e.g. constants) are unbounded
            bounds = getattr(variable, "bounds", None)
            if bounds is None:
                lower, upper = -np.inf, np.inf
            else:
                lower = evaluate_bound(bounds[0], "lower")
                upper = evaluate_bound(bounds[1], "upper")
            lower_bounds.extend([lower] * (end - start))
            upper_bounds.extend([upper] * (end - start))
            # Increment start
            start = end

        # Convert y_slices back to normal dictionary
        self.y_slices = dict(y_slices)
        # Also keep a record of what the y_slices are, to be stored in the model
        self.y_slices_explicit = dict(y_slices_explicit)

        # Also keep a record of bounds
        self.bounds = (np.array(lower_bounds), np.array(upper_bounds))

        # reset discretised_symbols
        self._discretised_symbols = {}

    def _get_variable_size(self, variable):
        """Helper function to determine what size a variable should be"""
        # If domain is empty then variable has size 1
        if variable.domain == []:
            return 1
        else:
            size = 0
            spatial_method = self.spatial_methods[variable.domain[0]]
            repeats = spatial_method._get_auxiliary_domain_repeats(variable.domains)
            for dom in variable.domain:
                size += spatial_method.mesh[dom].npts_for_broadcast_to_nodes * repeats
            return size

    def set_internal_boundary_conditions(self, model):
        """
        A method to set the internal boundary conditions for the submodel.
        These are required to properly calculate the gradient.
        Note: this method modifies the state of self.boundary_conditions.
        """

        def boundary_gradient(left_symbol, right_symbol):
            pybamm.logger.debug(
                f"Calculate boundary gradient ({left_symbol} and {right_symbol})"
            )
            left_domain = left_symbol.domain[0]
            right_domain = right_symbol.domain[0]

            left_mesh = self.spatial_methods[left_domain].mesh[left_domain]
            right_mesh = self.spatial_methods[right_domain].mesh[right_domain]

            left_symbol_disc = self.process_symbol(left_symbol)
            right_symbol_disc = self.process_symbol(right_symbol)

            return self.spatial_methods[left_domain].internal_neumann_condition(
                left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
            )

        # ``self.bcs`` is only extended after this loop, so read it live.
        bc_keys = self.bcs

        internal_bcs = {}
        for var in model.boundary_conditions:
            if not isinstance(var, pybamm.Concatenation):
                continue
            children = var.orphans

            # Dispatch hook: a spatial method may own its own internal-BC
            # logic (e.g. graph-traversal for arbitrary topology); a non-None
            # return replaces the default 1D-stack pairwise routine.
            primary_method = self.spatial_methods.get(children[0].domain[0])
            if primary_method is not None:
                handled = primary_method.set_internal_bcs_for_concat(
                    self, var, children, self.bcs[var]
                )
                if handled is not None:
                    # Only adopt entries for children not already user-supplied.
                    for child, child_bcs in handled.items():
                        if child in bc_keys:
                            continue
                        if not child_bcs:
                            # adopting an empty dict would strip the child of
                            # BCs entirely; surface it instead
                            pybamm.logger.warning(
                                f"No internal or external boundary conditions "
                                f"were found for {child.name!r} in domain "
                                f"{child.domain}; it will be discretised "
                                "without boundary conditions."
                            )
                            continue
                        internal_bcs[child] = child_bcs
                    continue
                # else fall through to legacy 1D-stack pairwise logic

            first_child = children[0]
            next_child = children[1]

            if "left" not in self.bcs[var] or "right" not in self.bcs[var]:
                raise pybamm.DiscretisationError(
                    f"Boundary conditions for the concatenated variable "
                    f"{var.name!r} must include 'left' and 'right' entries "
                    f"(got {sorted(self.bcs[var])}); other sides are not "
                    "supported by the 1D-stack internal-BC routine."
                )
            lbc = self.bcs[var]["left"]
            rbc = (boundary_gradient(first_child, next_child), "Neumann")

            if first_child not in bc_keys:
                internal_bcs.update({first_child: {"left": lbc, "right": rbc}})

            for current_child, next_child in itertools.pairwise(children[1:]):
                lbc = rbc
                rbc = (boundary_gradient(current_child, next_child), "Neumann")
                if current_child not in bc_keys:
                    internal_bcs.update({current_child: {"left": lbc, "right": rbc}})

            lbc = rbc
            rbc = self.bcs[var]["right"]
            if children[-1] not in bc_keys:
                internal_bcs.update({children[-1]: {"left": lbc, "right": rbc}})

        self.bcs.update(internal_bcs)

    def process_initial_conditions(self, model):
        """Discretise model initial_conditions.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})

        Returns
        -------
        tuple
            Tuple of processed_initial_conditions (dict of initial conditions) and
            concatenated_initial_conditions (numpy array of concatenated initial
            conditions)

        """
        # Discretise initial conditions
        processed_initial_conditions = self.process_dict(
            model.initial_conditions, ics=True
        )

        # Concatenate initial conditions into a single vector
        # check that all initial conditions are set
        processed_concatenated_initial_conditions = self._concatenate_in_order(
            processed_initial_conditions, check_complete=True
        )

        return processed_initial_conditions, processed_concatenated_initial_conditions

    def process_boundary_conditions(self, model):
        """Discretise model boundary_conditions, also converting keys to ids

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})

        Returns
        -------
        dict
            Dictionary of processed boundary conditions

        """

        processed_bcs = {}

        # process and set pybamm.variables first incase required
        # in discrisation of other boundary conditions
        for key, bcs in model.boundary_conditions.items():
            processed_bcs[key] = {}

            # check if the boundary condition at the origin for sphere domains is other
            # than no flux
            for subdomain in key.domain:
                if (
                    self.mesh[subdomain].coord_sys
                    in ["spherical polar", "cylindrical polar"]
                    and next(iter(self.mesh.geometry[subdomain].values()))["min"] == 0
                    and (bcs["left"][0].value != 0 or bcs["left"][1] != "Neumann")
                ):
                    raise pybamm.ModelError(
                        "Boundary condition at r = 0 must be a homogeneous "
                        f"Neumann condition for {self.mesh[subdomain].coord_sys} coordinates"
                    )

            # Handle legacy tab boundary conditions ("negative tab", etc.)
            if LEGACY_TAB_SIDES & set(bcs.keys()):
                bcs = self.check_tab_conditions(key, bcs)

            # Process boundary conditions
            for side, bc in bcs.items():
                eqn, typ = bc
                pybamm.logger.debug(f"Discretise {key} ({side} bc)")
                processed_eqn = self.process_symbol(eqn)
                processed_bcs[key][side] = (processed_eqn, typ)

        return processed_bcs

    def check_tab_conditions(self, symbol, bcs):
        """
        Check any boundary conditions applied on "negative tab", "positive tab"
        and "no tab". For 1D current collector meshes, these conditions are
        converted into boundary conditions on "left" (tab at z=0) or "right"
        (tab at z=l_z) depending on the tab location stored in the mesh. For 2D
        current collector meshes, the boundary conditions can be applied on the
        tabs directly.

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol`
            The symbol on which the boundary conditions are applied.
        bcs : dict
            The dictionary of boundary conditions (a dict of {side: equation}).

        Returns
        -------
        dict
            The dictionary of boundary conditions, with the keys changed to
            "left" and "right" where necessary.

        """
        # Check symbol domain
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        if domain != "current collector":
            raise pybamm.ModelError(
                "Boundary conditions can only be applied on the tabs in the domain "
                f"'current collector', but {symbol} has domain {domain}"
            )
        # Replace keys with "left" and "right" as appropriate for 1D meshes
        if isinstance(mesh, pybamm.SubMesh1D):
            # send boundary conditions applied on the tabs to "left" or "right"
            # depending on the tab location stored in the mesh
            for tab in ["negative tab", "positive tab"]:
                if any(tab in side for side in bcs):
                    bcs[mesh.tabs[tab]] = bcs.pop(tab)
            # if there was a tab at either end, then the boundary conditions
            # have now been set on "left" and "right" as required by the spatial
            # method, so there is no need to further modify the bcs dict
            if "left" in bcs and "right" in bcs:
                pass
            # if both tabs are located at z=0 then the "right" boundary condition
            # (at z=1) is the condition for "no tab"
            elif "left" in bcs:
                bcs["right"] = bcs.pop("no tab")
            # else if both tabs are located at z=1, the "left" boundary condition
            # (at z=0) is the condition for "no tab"
            else:
                bcs["left"] = bcs.pop("no tab")

        return bcs

    def process_rhs_and_algebraic(self, model):
        """Discretise model equations - differential ('rhs') and algebraic.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})

        Returns
        -------
        tuple
            Tuple of processed_rhs (dict of processed differential equations),
            processed_concatenated_rhs, processed_algebraic (dict of processed algebraic
            equations) and processed_concatenated_algebraic

        """

        # Discretise right-hand sides, passing domain from variable
        processed_rhs = self.process_dict(model.rhs)

        # Concatenate rhs into a single state vector
        # Need to concatenate in order as the ordering of equations could be different
        # in processed_rhs and model.rhs
        processed_concatenated_rhs = self._concatenate_in_order(processed_rhs)

        # Discretise and concatenate algebraic equations
        processed_algebraic = self.process_dict(model.algebraic)

        # Concatenate algebraic into a single state vector
        # Need to concatenate in order as the ordering of equations could be different
        # in processed_algebraic and model.algebraic
        processed_concatenated_algebraic = self._concatenate_in_order(
            processed_algebraic
        )

        return (
            processed_rhs,
            processed_concatenated_rhs,
            processed_algebraic,
            processed_concatenated_algebraic,
        )

    def create_mass_matrix(self, model):
        """Creates mass matrix of the discretised model.
        Note that the model is assumed to be of the form M*y_dot = f(t,y), where
        M is the (possibly singular) mass matrix.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Discretised model. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})

        Returns
        -------
        :class:`pybamm.Matrix`
            The mass matrix
        """
        mass_list = []

        # get a list of model rhs variables that are sorted according to
        # where they are in the state vector
        sorted_model_variables = sorted(
            model.rhs, key=lambda var: self.y_slices[var][0]
        )

        # Process mass matrices for the differential equations
        for var in sorted_model_variables:
            if var.domain == []:
                mass = 1.0
            else:
                mass = (
                    self.spatial_methods[var.domain[0]]
                    .mass_matrix(var, self.bcs)
                    .entries
                )
            mass_list.append(mass)

        # Create lumped mass matrix (of zeros) of the correct shape for the
        # discretised algebraic equations
        if model.algebraic.keys():
            mass_algebraic_size = model.concatenated_algebraic.shape[0]
            mass_algebraic = csr_matrix((mass_algebraic_size, mass_algebraic_size))
            mass_list.append(mass_algebraic)

        # Create block diagonal (sparse) mass matrix (if model is not empty)
        N_rhs = len(model.rhs)
        N_alg = len(model.algebraic)

        if N_rhs > 0 or N_alg > 0:
            mass_matrix = pybamm.Matrix(block_diag(mass_list, format="csr"))
        else:
            mass_matrix = None

        return mass_matrix

    def _pre_process_variables(
        self,
        variables: dict[str, pybamm.Symbol],
        initial_conditions: dict[pybamm.Variable, pybamm.Symbol],
        comparable_variables: dict[str, pybamm.Symbol] | None = None,
    ):
        """
        Pre-process variables before discretisation. This involves:
        - ensuring that all the state variables are included in the variables,
          any missing are added
        - checking, by symbol identity, that state variables already present are
          the state variables themselves. ``comparable_variables`` holds the
          expressions at the same processing stage as the keys (default:
          ``variables``).

        Parameters
        ----------
        variables : dict
            Dictionary of variables to pre-process
        initial_conditions : dict
            Dictionary of initial conditions

        Returns
        -------
        dict
            Pre-processed variables (copy of input variables with any missing state)

        Raises
        ------
        :class:`pybamm.ModelError`
            If any state variable names are already included but with
            incorrect expressions
        """
        if comparable_variables is None:
            comparable_variables = variables
        new_variables = dict(variables)
        for var in initial_conditions:
            if var.name not in new_variables:
                new_variables[var.name] = var
            elif var.name in comparable_variables:
                existing_var = comparable_variables[var.name]
                if existing_var != var:
                    raise pybamm.ModelError(
                        f"Variable '{var.name}' should have expression "
                        f"'{var}', but has expression '{existing_var}'"
                    )
        return new_variables

    def process_dict(self, var_eqn_dict, ics=False):
        """Discretise a dictionary of {variable: equation}, broadcasting if necessary
        (can be model.rhs, model.algebraic, model.initial_conditions or
        model.variables).

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise
            (can be model.rhs, model.algebraic, model.initial_conditions or
            model.variables)
        ics : bool, optional
            Whether the equations are initial conditions. If True, the equations are
            scaled by the reference value of the variable, if given

        Returns
        -------
        new_var_eqn_dict : dict
            Discretised equations

        """
        return {
            k: self.process_equation(k, v, ics=ics) for k, v in var_eqn_dict.items()
        }

    def process_equation(self, name, eqn, ics=False):
        """Discretise a dictionary of {variable: equation}, broadcasting if necessary
        (can be model.rhs, model.algebraic, model.initial_conditions or
        model.variables).

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise
            (can be model.rhs, model.algebraic, model.initial_conditions or
            model.variables)
        ics : bool, optional
            Whether the equations are initial conditions. If True, the equations are
            scaled by the reference value of the variable, if given

        Returns
        -------
        processed_eqn
            Discretised equation

        """
        # Broadcast if the equation evaluates to a number (e.g. Scalar)
        if np.prod(eqn.shape_for_testing) == 1 and not isinstance(name, str):
            if name.domain == []:
                eqn = eqn * pybamm.Vector([1])
            else:
                eqn = pybamm.FullBroadcast(eqn, broadcast_domains=name.domains)

        pybamm.logger.debug(f"Discretise {name!r}")

        processed_eqn = self.process_symbol(eqn)
        if ics and (reference := getattr(name, "reference", 0)) != 0:
            processed_eqn = processed_eqn - reference

        # Calculate scale if the key has a scale
        scale = getattr(name, "scale", 1)
        if scale != 1:
            processed_eqn = processed_eqn / scale

        return processed_eqn

    def process_symbol(self, symbol):
        """Discretise operators in model equations.
        If a symbol has already been discretised, the stored value is returned.

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol`
            Symbol to discretise

        Returns
        -------
        :class:`pybamm.expression_tree.symbol.Symbol`
            Discretised symbol

        """
        return pybamm.tree_map(
            self._discretise_node,
            symbol,
            cache=self._discretised_symbols,
            is_leaf=self._discretises_own_children,
        )

    # classes whose handler discretises what it needs itself (see is_leaf)
    _self_discretising: tuple[type, ...] = ()

    @classmethod
    @functools.cache
    def _handler_for(cls, symbol_class: type) -> Callable:
        for klass in symbol_class.__mro__:
            handler = cls._handlers.get(klass)
            if handler is not None:
                return handler
        raise pybamm.DiscretisationError(  # pragma: no cover
            f"No discretisation handler registered for {symbol_class.__name__}"
        )

    @classmethod
    def _discretises_own_children(cls, symbol):
        """Symbols whose handler builds and discretises its own expressions rather
        than receiving discretised children (the ``is_leaf`` of the tree map)."""
        return isinstance(symbol, cls._self_discretising)

    def _discretise_node(self, symbol, new_leaves):
        """Discretise one node given its discretised children; the callback for
        :func:`pybamm.tree_map`."""
        if symbol.domain != [] and self.bcs:
            # If boundary conditions are provided, need to check for BCs on tabs
            key_id = next(iter(self.bcs.keys()))
            if LEGACY_TAB_SIDES & set(self.bcs[key_id].keys()):
                self.bcs[key_id] = self.check_tab_conditions(symbol, self.bcs[key_id])
        handler = self._handler_for(type(symbol))
        discretised_symbol = handler(self, symbol, new_leaves[: len(symbol.children)])
        discretised_symbol.test_shape()
        # processed variables read the meshes of the symbol's domains off the result
        return discretised_symbol.with_meshes(self.mesh, symbol.domains)

    def _spatial_method_of(self, symbol):
        """The spatial method of a symbol's primary domain (None if it has none)."""
        if symbol.domain == []:
            return None
        return self.spatial_methods[symbol.domain[0]]

    # -- handlers ---------------------------------------------------------

    def _disc_default(self, symbol, disc_children):
        # leaves such as scalars, arrays and state vectors are already discrete
        return symbol

    def _disc_children_copy(self, symbol, disc_children):
        return symbol.create_copy(list(disc_children))

    def _disc_binary(self, symbol, disc_children):
        spatial_method = self._spatial_method_of(symbol)
        left, right = symbol.children
        disc_left, disc_right = disc_children
        # A scalar diffusion coefficient becomes an identity-like vector field
        if isinstance(spatial_method, pybamm.FiniteVolume2D):
            n_components = 2
        elif isinstance(spatial_method, pybamm.FiniteVolumeUnstructured):
            n_components = self.mesh[symbol.domain[0]].dimension
        else:
            n_components = None
        if n_components is not None:
            if isinstance(left, pybamm.Scalar) and isinstance(
                right, pybamm.VectorField | pybamm.Gradient
            ):
                left = pybamm.VectorField(*[left] * n_components)
                disc_left = pybamm.VectorField(*[disc_left] * n_components)
            elif isinstance(right, pybamm.Scalar) and isinstance(
                left, pybamm.VectorField | pybamm.Gradient
            ):
                right = pybamm.VectorField(*[right] * n_components)
                disc_right = pybamm.VectorField(*[disc_right] * n_components)
        if symbol.domain == []:
            if isinstance(disc_left, pybamm.VectorField) or isinstance(
                disc_right, pybamm.VectorField
            ):
                return self._process_vector_field_binary(symbol, disc_left, disc_right)
            return pybamm.simplify_if_constant(
                symbol.create_copy(new_children=[disc_left, disc_right])
            )
        return spatial_method.process_binary_operators(
            symbol, left, right, disc_left, disc_right
        )

    def _disc_average(self, symbol, disc_children):
        # Create a new Integral operator and process it
        child = symbol.orphans[0]
        if isinstance(symbol, pybamm.SizeAverage):
            R = symbol.integration_variable[0]
            f_a_dist = symbol.f_a_dist
            # take average using Integral and distribution f_a_dist
            average = pybamm.Integral(f_a_dist * child, R) / pybamm.Integral(
                f_a_dist, R
            )
        else:
            x = symbol.integration_variable
            v = pybamm.ones_like(child)
            average = pybamm.Integral(child, x) / pybamm.Integral(v, x)
        return self.process_symbol(average)

    def _disc_unary_default(self, symbol, disc_children):
        (disc_child,) = disc_children
        if isinstance(disc_child, pybamm.VectorField):
            new_comps = [
                symbol.create_copy(new_children=[c]) for c in disc_child.components
            ]
            return pybamm.VectorField(
                *new_comps, disc_state_vector=disc_child.disc_state_vector
            )
        return symbol.create_copy(new_children=[disc_child])

    def _child_spatial_method(self, symbol):
        return self.spatial_methods[symbol.child.domain[0]]

    def _disc_gradient(self, symbol, disc_children):
        return self._child_spatial_method(symbol).gradient(
            symbol.child, disc_children[0], self.bcs
        )

    def _disc_divergence(self, symbol, disc_children):
        child = symbol.child
        if child.domain != []:
            child_spatial_method = self.spatial_methods[child.domain[0]]
            # Intercept div(grad(u)) and div(D*grad(u)) before processing
            # children, to avoid the expensive Green-Gauss gradient assembly.
            if isinstance(child_spatial_method, pybamm.FiniteVolumeUnstructured):
                grad_sym = None
                coeff_sym = None
                if isinstance(child, pybamm.Gradient):
                    grad_sym = child
                    coeff_sym = pybamm.Scalar(1)
                elif isinstance(child, pybamm.Multiplication):
                    left_c, right_c = child.children
                    if isinstance(right_c, pybamm.Gradient):
                        grad_sym, coeff_sym = right_c, left_c
                    elif isinstance(left_c, pybamm.Gradient):
                        grad_sym, coeff_sym = left_c, right_c
                if grad_sym is not None:
                    return child_spatial_method.div_D_grad(
                        symbol,
                        grad_sym.child,
                        self.process_symbol(coeff_sym),
                        self.process_symbol(grad_sym.child),
                        self.bcs,
                    )
        return self._child_spatial_method(symbol).divergence(
            child, self.process_symbol(child), self.bcs
        )

    def _disc_laplacian(self, symbol, disc_children):
        return self._child_spatial_method(symbol).laplacian(
            symbol.child, disc_children[0], self.bcs
        )

    def _disc_gradient_squared(self, symbol, disc_children):
        return self._child_spatial_method(symbol).gradient_squared(
            symbol.child, disc_children[0], self.bcs
        )

    def _disc_mass(self, symbol, disc_children):
        return self._child_spatial_method(symbol).mass_matrix(symbol.child, self.bcs)

    def _disc_boundary_mass(self, symbol, disc_children):
        return self._child_spatial_method(symbol).boundary_mass_matrix(
            symbol.child, self.bcs
        )

    def _disc_indefinite_integral(self, symbol, disc_children):
        return self._child_spatial_method(symbol).indefinite_integral(
            symbol.child, disc_children[0], "forward"
        )

    def _disc_backward_indefinite_integral(self, symbol, disc_children):
        return self._child_spatial_method(symbol).indefinite_integral(
            symbol.child, disc_children[0], "backward"
        )

    def _disc_integral(self, symbol, disc_children):
        integral_spatial_method = self.spatial_methods[
            symbol.integration_variable[0].domain[0]
        ]
        out = integral_spatial_method.integral(
            symbol.child,
            disc_children[0],
            symbol._integration_dimension,
            symbol.integration_variable,
        )
        return out.with_domains(symbol)

    def _disc_definite_integral_vector(self, symbol, disc_children):
        return self._child_spatial_method(symbol).definite_integral_matrix(
            symbol.child, vector_type=symbol.vector_type
        )

    def _disc_one_dimensional_integral(self, symbol, disc_children):
        spatial_method = self.spatial_methods[symbol.integration_domain[0]]
        return spatial_method.one_dimensional_integral(
            symbol,
            symbol.child,
            disc_children[0],
            symbol.integration_domain,
            symbol.direction,
        )

    def _disc_boundary_integral(self, symbol, disc_children):
        return self._child_spatial_method(symbol).boundary_integral(
            symbol.child, disc_children[0], symbol.region
        )

    def _disc_broadcast(self, symbol, disc_children):
        # Broadcast new_child to the domain specified by symbol.domain
        # Different discretisations may broadcast differently
        return self._spatial_method_of(symbol).broadcast(
            disc_children[0], symbol.domains, symbol.broadcast_type
        )

    def _disc_delta_function(self, symbol, disc_children):
        return self._spatial_method_of(symbol).delta_function(symbol, disc_children[0])

    def _disc_boundary_operator(self, symbol, disc_children):
        # if boundary operator applied on "negative tab" or "positive tab" *and*
        # the mesh is 1D then change side to "left" or "right" as appropriate
        if symbol.side in ["negative tab", "positive tab"]:
            mesh = self.mesh[symbol.children[0].domain[0]]
            if isinstance(mesh, pybamm.SubMesh1D):
                symbol = symbol._replace(side=mesh.tabs[symbol.side])
        return self._child_spatial_method(symbol).boundary_value_or_flux(
            symbol, disc_children[0], self.bcs
        )

    def _disc_evaluate_at(self, symbol, disc_children):
        return self._child_spatial_method(symbol).evaluate_at(
            symbol, disc_children[0], symbol.position
        )

    def _disc_upwind_downwind_2d(self, symbol, disc_children):
        return self._spatial_method_of(symbol).upwind_or_downwind(
            symbol.child,
            disc_children[0],
            self.bcs,
            symbol.lr_direction,
            symbol.tb_direction,
        )

    def _disc_node_to_edge_2d(self, symbol, disc_children):
        return self._spatial_method_of(symbol).node_to_edge(
            disc_children[0], method="arithmetic", direction=symbol.direction
        )

    def _disc_upwind_downwind(self, symbol, disc_children):
        direction = symbol.name  # upwind or downwind
        return self._spatial_method_of(symbol).upwind_or_downwind(
            symbol.child, disc_children[0], self.bcs, direction
        )

    def _disc_not_constant(self, symbol, disc_children):
        # After discretisation, we can make the symbol constant
        return disc_children[0]

    def _disc_component(self, symbol, disc_children):
        (disc_child,) = disc_children
        if not isinstance(disc_child, pybamm.VectorField):
            raise pybamm.DiscretisationError(
                "Component can only be applied to a VectorField"
            )
        if symbol.index >= disc_child.n_components:
            raise pybamm.DiscretisationError(
                f"Component index {symbol.index} is out of range for a "
                f"VectorField with {disc_child.n_components} components"
            )
        return disc_child.components[symbol.index]

    def _disc_norm(self, symbol, disc_children):
        (disc_child,) = disc_children
        if not isinstance(disc_child, pybamm.VectorField):
            raise pybamm.DiscretisationError(
                "Norm can only be applied to a VectorField"
            )
        return sum(c**2 for c in disc_child.components) ** 0.5

    def _disc_magnitude(self, symbol, disc_children):
        (disc_child,) = disc_children
        if not isinstance(disc_child, pybamm.VectorField):
            raise pybamm.DiscretisationError(
                "Magnitude can only be applied to a vector field"
            )
        direction = symbol.direction
        if direction == "lr":
            return disc_child.lr_field
        elif direction == "tb":
            return disc_child.tb_field
        else:
            raise pybamm.DiscretisationError("Invalid direction")

    def _disc_variable_dot(self, symbol, disc_children):
        # Add symbol's reference and multiply by the symbol's scale
        # so that the state vector is of order 1
        return symbol.reference + symbol.scale * pybamm.StateVectorDot(
            *self.y_slices[symbol.get_variable()],
            domains=symbol.domains,
        )

    def _disc_variable(self, symbol, disc_children):
        # check_well_posedness usually catches this, but not without debug_mode
        try:
            y_slices = self.y_slices[symbol]
        except KeyError as error:
            raise pybamm.ModelError(
                f"No key set for variable '{symbol.name}'. Make sure it is included in either "
                "model.rhs or model.algebraic in an unmodified form "
                "(e.g. not Broadcasted)"
            ) from error
        # Add symbol's reference and multiply by the symbol's scale
        # so that the state vector is of order 1
        return symbol.reference + symbol.scale * pybamm.StateVector(
            *y_slices, domains=symbol.domains
        )

    def _disc_spatial_variable(self, symbol, disc_children):
        return self._spatial_method_of(symbol).spatial_variable(symbol)

    def _disc_concatenation_variable(self, symbol, disc_children):
        # create new children without scale and reference
        # the scale and reference will be applied to the concatenation instead
        new_children = []
        old_y_slices = self.y_slices.copy()
        for child in symbol.children:
            child_no_scale = child.create_copy(scale=1, reference=0)
            self.y_slices[child_no_scale] = self.y_slices[child]
            new_children.append(self.process_symbol(child_no_scale))
        self.y_slices = old_y_slices
        new_symbol = self._spatial_method_of(symbol).concatenation(new_children)
        # apply scale to the whole concatenation
        return symbol.reference + symbol.scale * new_symbol

    def _disc_concatenation(self, symbol, disc_children):
        return self._spatial_method_of(symbol).concatenation(list(disc_children))

    def _disc_input_parameter(self, symbol, disc_children):
        if symbol.domain != []:
            expected_size = self._get_variable_size(symbol)
        else:
            expected_size = None
        if symbol._expected_size is not None:
            expected_size = symbol._expected_size
        return pybamm.InputParameter(
            symbol.name, symbol.domain, expected_size=expected_size
        )

    def _disc_coupled_variable(self, symbol, disc_children):
        raise pybamm.DiscretisationError(
            f"CoupledVariable '{symbol.name}' was not resolved before discretisation. "
            "Ensure the variable exists in model.variables."
        )

    def _disc_constant(self, symbol, disc_children):
        # after discretisation we just care about the value, not the name
        return self.process_symbol(pybamm.Scalar(symbol.value))

    def _process_vector_field_binary(self, symbol, disc_left, disc_right):
        """Broadcast a scalar side, then apply ``symbol`` component-wise."""
        left_is_vf = isinstance(disc_left, pybamm.VectorField)
        right_is_vf = isinstance(disc_right, pybamm.VectorField)
        if left_is_vf and right_is_vf:
            if disc_left.n_components != disc_right.n_components:
                raise pybamm.DiscretisationError(
                    f"Cannot combine VectorFields with {disc_left.n_components} and "
                    f"{disc_right.n_components} components"
                )
            n = disc_left.n_components
        elif left_is_vf:
            n = disc_left.n_components
            disc_right = pybamm.VectorField(*[disc_right] * n)
        else:
            n = disc_right.n_components
            disc_left = pybamm.VectorField(*[disc_left] * n)
        new_comps = [
            pybamm.simplify_if_constant(
                symbol.create_copy(
                    new_children=[disc_left.components[k], disc_right.components[k]]
                )
            )
            for k in range(n)
        ]
        if disc_left.disc_state_vector is not None:
            disc_state_vector = disc_left.disc_state_vector
        else:
            disc_state_vector = disc_right.disc_state_vector
        return pybamm.VectorField(*new_comps, disc_state_vector=disc_state_vector)

    def concatenate(self, *symbols, sparse=False):
        if sparse:
            return pybamm.SparseStack(*symbols)
        else:
            return pybamm.numpy_concatenation(*symbols)

    def _concatenate_in_order(self, var_eqn_dict, check_complete=False, sparse=False):
        """
        Concatenate a dictionary of {variable: equation} using self.y_slices

        The keys/variables in `var_eqn_dict` must be the same as the ids in
        `self.y_slices`.
        The resultant concatenation is ordered according to the ordering of the slice
        values in `self.y_slices`

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise
        check_complete : bool, optional
            Whether to check keys in var_eqn_dict against self.y_slices. Default
            is False
        sparse : bool, optional
            If True the concatenation will be a :class:`pybamm.SparseStack`. If
            False the concatenation will be a :class:`pybamm.NumpyConcatenation`.
            Default is False

        Returns
        -------
        var_eqn_dict : dict
            Discretised right-hand side equations

        """
        # Unpack symbols in variables that are concatenations of variables
        unpacked_variables = set()
        for symbol in var_eqn_dict:
            unpacked_variables.add(symbol)
            if isinstance(symbol, pybamm.ConcatenationVariable):
                unpacked_variables.update(symbol.children)

        # Check keys from the given var_eqn_dict against self.y_slices
        if check_complete and unpacked_variables != self.y_slices.keys():
            given_variable_names = [v.name for v in var_eqn_dict]
            raise pybamm.ModelError(
                "Initial conditions are insufficient. Only "
                f"provided for {given_variable_names} "
            )

        # sort equations according to slices
        return self.concatenate(
            *(
                eq
                for _, eq in sorted(
                    var_eqn_dict.items(), key=lambda item: self.y_slices[item[0]][0]
                )
            ),
            sparse=sparse,
        )

    def check_model(self, model):
        """Perform some basic checks to make sure the discretised model makes sense."""
        self.check_initial_conditions(model)

    def check_initial_conditions(self, model):
        # Check initial conditions are a numpy array
        # Individual
        for var, eqn in model.initial_conditions.items():
            ic_eval = eqn.evaluate(t=0, inputs="shape test")
            if not isinstance(ic_eval, np.ndarray):
                raise pybamm.ModelError(
                    "initial conditions must be numpy array after discretisation but "
                    f"they are {type(ic_eval)} for variable '{var}'."
                )

            # Check that the initial condition is within the bounds
            # Skip this check if there are input parameters in the initial conditions
            bounds = var.bounds
            if not eqn.has_symbol_of_classes(pybamm.InputParameter) and not (
                all(bounds[0].value <= ic_eval) and all(ic_eval <= bounds[1].value)
            ):
                raise pybamm.ModelError(
                    "initial condition is outside of variable bounds "
                    f"{bounds} for variable '{var}'."
                )

        # Check initial conditions and model equations have the same shape
        # Individual
        for var in model.rhs:
            if model.rhs[var].shape != model.initial_conditions[var].shape:
                raise pybamm.ModelError(
                    "rhs and initial conditions must have the same shape after "
                    "discretisation but rhs.shape = "
                    f"{model.rhs[var].shape} and initial_conditions.shape = {model.initial_conditions[var].shape} for variable '{var}'."
                )
        for var in model.algebraic:
            if model.algebraic[var].shape != model.initial_conditions[var].shape:
                raise pybamm.ModelError(
                    "algebraic and initial conditions must have the same shape after "
                    "discretisation but algebraic.shape = "
                    f"{model.algebraic[var].shape} and initial_conditions.shape = {model.initial_conditions[var].shape} for variable '{var}'."
                )

    def is_variable_independent(self, var, all_vars_in_eqns):
        pybamm.logger.verbose("Removing independent blocks.")
        if not isinstance(var, pybamm.Variable):
            return False

        this_var_is_independent = var not in all_vars_in_eqns
        not_in_y_slices = var not in self.y_slices
        not_in_discretised = var not in self._discretised_symbols
        is_0D = len(var.domain) == 0
        this_var_is_independent = (
            this_var_is_independent and not_in_y_slices and not_in_discretised and is_0D
        )
        return this_var_is_independent

    def remove_independent_variables_from_rhs(self, model):
        unpacker = pybamm.SymbolUnpacker(pybamm.Variable)
        eqns_to_check = itertools.chain(
            model.rhs.values(),
            model.algebraic.values(),
            (x[side][0] for x in model.boundary_conditions.values() for side in x),
            # only check children of variables, this will skip the variable itself
            # and catch any other cases
            (
                child
                for var in model.variables_matching_keys().values()
                for child in var.children
            ),
            (event.expression for event in model.events),
        )
        all_vars_in_eqns = unpacker.unpack_list_of_symbols(eqns_to_check)

        vars_to_update = {}
        for var in list(model.rhs.keys()):
            # a model needs at least one differential equation to solve
            if len(model.rhs) <= 1:
                break
            if not self.is_variable_independent(var, all_vars_in_eqns):
                continue
            pybamm.logger.info(f"removing variable {var} from rhs")
            _rhs = model.rhs.pop(var)
            _initial_condition = model.initial_conditions.pop(var)
            explicit_integral = pybamm.ExplicitTimeIntegral(_rhs, _initial_condition)
            # Collect variables to update in _variables_processed
            # Do NOT modify model.variables - only update _variables_processed
            vars_to_update[var.name] = explicit_integral
            # edge case where a variable appears
            # in variables twice under different names
            vars_to_update.update(
                {
                    key: explicit_integral
                    for key, value in model.variables_matching_keys().items()
                    if value == var
                }
            )

        model.update_processed_variables(vars_to_update)
        return model


# {class: handler(disc, symbol, disc_children)}; subclasses use their nearest
# registered base class's handler
Discretisation._handlers = {
    pybamm.Symbol: Discretisation._disc_default,
    pybamm.BinaryOperator: Discretisation._disc_binary,
    pybamm._BaseAverage: Discretisation._disc_average,
    pybamm.UnaryOperator: Discretisation._disc_unary_default,
    pybamm.Gradient: Discretisation._disc_gradient,
    pybamm.Divergence: Discretisation._disc_divergence,
    pybamm.Laplacian: Discretisation._disc_laplacian,
    pybamm.GradientSquared: Discretisation._disc_gradient_squared,
    pybamm.Mass: Discretisation._disc_mass,
    pybamm.BoundaryMass: Discretisation._disc_boundary_mass,
    pybamm.IndefiniteIntegral: Discretisation._disc_indefinite_integral,
    pybamm.BackwardIndefiniteIntegral: Discretisation._disc_backward_indefinite_integral,
    pybamm.Integral: Discretisation._disc_integral,
    pybamm.DefiniteIntegralVector: Discretisation._disc_definite_integral_vector,
    pybamm.OneDimensionalIntegral: Discretisation._disc_one_dimensional_integral,
    pybamm.BoundaryIntegral: Discretisation._disc_boundary_integral,
    pybamm.Broadcast: Discretisation._disc_broadcast,
    pybamm.DeltaFunction: Discretisation._disc_delta_function,
    pybamm.BoundaryOperator: Discretisation._disc_boundary_operator,
    pybamm.EvaluateAt: Discretisation._disc_evaluate_at,
    pybamm.UpwindDownwind2D: Discretisation._disc_upwind_downwind_2d,
    pybamm.NodeToEdge2D: Discretisation._disc_node_to_edge_2d,
    pybamm.UpwindDownwind: Discretisation._disc_upwind_downwind,
    pybamm.NotConstant: Discretisation._disc_not_constant,
    pybamm.Component: Discretisation._disc_component,
    pybamm.Norm: Discretisation._disc_norm,
    pybamm.Magnitude: Discretisation._disc_magnitude,
    pybamm.Function: Discretisation._disc_children_copy,
    pybamm.Conditional: Discretisation._disc_children_copy,
    pybamm.VariableDot: Discretisation._disc_variable_dot,
    pybamm.Variable: Discretisation._disc_variable,
    pybamm.SpatialVariable: Discretisation._disc_spatial_variable,
    pybamm.ConcatenationVariable: Discretisation._disc_concatenation_variable,
    pybamm.Concatenation: Discretisation._disc_concatenation,
    pybamm.InputParameter: Discretisation._disc_input_parameter,
    pybamm.CoupledVariable: Discretisation._disc_coupled_variable,
    pybamm.TensorField: Discretisation._disc_children_copy,
    pybamm.Constant: Discretisation._disc_constant,
}

Discretisation._self_discretising = (
    pybamm._BaseAverage,
    pybamm.ConcatenationVariable,
    pybamm.Constant,
    pybamm.Divergence,
)
