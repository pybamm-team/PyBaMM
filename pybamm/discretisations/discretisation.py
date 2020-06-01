#
# Interface for discretisation
#
import pybamm
import numpy as np
from collections import defaultdict, OrderedDict
from scipy.sparse import block_diag, csc_matrix, csr_matrix
from scipy.sparse.linalg import inv


def has_bc_of_form(symbol, side, bcs, form):

    if symbol.id in bcs:
        if bcs[symbol.id][side][1] == form:
            return True
        else:
            return False

    else:
        return False


class Discretisation(object):
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
    """

    def __init__(self, mesh=None, spatial_methods=None):
        self._mesh = mesh
        if mesh is None:
            self._spatial_methods = {}
        else:
            # Unpack macroscale to the constituent subdomains
            if "macroscale" in spatial_methods.keys():
                method = spatial_methods["macroscale"]
                spatial_methods["negative electrode"] = method
                spatial_methods["separator"] = method
                spatial_methods["positive electrode"] = method

            self._spatial_methods = spatial_methods
            for method in self._spatial_methods.values():
                method.build(mesh)

        self.bcs = {}
        self.y_slices = {}
        self._discretised_symbols = {}
        self.external_variables = {}

    @property
    def mesh(self):
        return self._mesh

    @property
    def y_slices(self):
        return self._y_slices

    @y_slices.setter
    def y_slices(self, value):
        if not isinstance(value, dict):
            raise TypeError("""y_slices should be dict, not {}""".format(type(value)))

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

    def process_model(self, model, inplace=True, check_model=True):
        """Discretise a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})
        inplace : bool, optional
            If True, discretise the model in place. Otherwise, return a new
            discretised model. Default is True.
        check_model : bool, optional
            If True, model checks are performed after discretisation. For large
            systems these checks can be slow, so can be skipped by setting this
            option to False. When developing, testing or debugging it is recommened
            to leave this option as True as it may help to identify any errors.
            Default is True.

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

        pybamm.logger.info("Start discretising {}".format(model.name))

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
        variables = list(model.rhs.keys()) + list(model.algebraic.keys())
        if self.spatial_methods == {} and any(var.domain != [] for var in variables):
            for var in variables:
                if var.domain != []:
                    raise pybamm.DiscretisationError(
                        "Spatial method has not been given "
                        "for variable {} with domain {}".format(var.name, var.domain)
                    )

        # Set the y split for variables
        pybamm.logger.info("Set variable slices for {}".format(model.name))
        self.set_variable_slices(variables)
        # Keep a record of y_slices in the model
        model.y_slices = self.y_slices_explicit

        # now add extrapolated external variables to the boundary conditions
        # if required by the spatial method
        self._preprocess_external_variables(model)
        self.set_external_variables(model)

        # set boundary conditions (only need key ids for boundary_conditions)
        pybamm.logger.info("Discretise boundary conditions for {}".format(model.name))
        self.bcs = self.process_boundary_conditions(model)
        pybamm.logger.info("Set internal boundary conditions for {}".format(model.name))
        self.set_internal_boundary_conditions(model)

        # set up inplace vs not inplace
        if inplace:
            # any changes to model_disc attributes will change model attributes
            # since they point to the same object
            model_disc = model
        else:
            # create an empty copy of the original model
            model_disc = model.new_copy()

        model_disc.bcs = self.bcs

        pybamm.logger.info("Discretise initial conditions for {}".format(model.name))
        ics, concat_ics = self.process_initial_conditions(model)
        model_disc.initial_conditions = ics
        model_disc.concatenated_initial_conditions = concat_ics

        # Discretise variables (applying boundary conditions)
        # Note that we **do not** discretise the keys of model.rhs,
        # model.initial_conditions and model.boundary_conditions
        pybamm.logger.info("Discretise variables for {}".format(model.name))
        model_disc.variables = self.process_dict(model.variables)

        # Process parabolic and elliptic equations
        pybamm.logger.info("Discretise model equations for {}".format(model.name))
        rhs, concat_rhs, alg, concat_alg = self.process_rhs_and_algebraic(model)
        model_disc.rhs, model_disc.concatenated_rhs = rhs, concat_rhs
        model_disc.algebraic, model_disc.concatenated_algebraic = alg, concat_alg

        # Process events
        processed_events = []
        pybamm.logger.info("Discretise events for {}".format(model.name))
        for event in model.events:
            pybamm.logger.debug("Discretise event '{}'".format(event.name))
            processed_event = pybamm.Event(
                event.name, self.process_symbol(event.expression), event.event_type
            )
            processed_events.append(processed_event)
        model_disc.events = processed_events

        # Create mass matrix
        pybamm.logger.info("Create mass matrix for {}".format(model.name))
        model_disc.mass_matrix, model_disc.mass_matrix_inv = self.create_mass_matrix(
            model_disc
        )

        # Check that resulting model makes sense
        if check_model:
            pybamm.logger.info("Performing model checks for {}".format(model.name))
            self.check_model(model_disc)

        pybamm.logger.info("Finish discretising {}".format(model.name))

        # Record that the model has been discretised
        model_disc.is_discretised = True

        return model_disc

    def set_variable_slices(self, variables):
        """
        Sets the slicing for variables.

        Parameters
        ----------
        variables : iterable of :class:`pybamm.Variables`
            The variables for which to set slices
        """
        # Set up y_slices
        y_slices = defaultdict(list)
        y_slices_explicit = defaultdict(list)
        start = 0
        end = 0
        # Iterate through unpacked variables, adding appropriate slices to y_slices
        for variable in variables:
            # Add up the size of all the domains in variable.domain
            if isinstance(variable, pybamm.Concatenation):
                spatial_method = self.spatial_methods[variable.domain[0]]
                children = variable.children
                meshes = OrderedDict()
                for child in children:
                    meshes[child] = [spatial_method.mesh[dom] for dom in child.domain]
                sec_points = spatial_method._get_auxiliary_domain_repeats(
                    variable.domains
                )
                for i in range(sec_points):
                    for child, mesh in meshes.items():
                        for domain_mesh in mesh:
                            end += domain_mesh.npts_for_broadcast_to_nodes
                        y_slices[child.id].append(slice(start, end))
                        y_slices_explicit[child].append(slice(start, end))
                        start = end
            else:
                end += self._get_variable_size(variable)
                y_slices[variable.id].append(slice(start, end))
                y_slices_explicit[variable].append(slice(start, end))
                start = end

        # Convert y_slices back to normal dictionary
        self.y_slices = dict(y_slices)
        # Also keep a record of what the y_slices are, to be stored in the model
        self.y_slices_explicit = dict(y_slices_explicit)

        # reset discretised_symbols
        self._discretised_symbols = {}

    def _get_variable_size(self, variable):
        "Helper function to determine what size a variable should be"
        # If domain is empty then variable has size 1
        if variable.domain == []:
            return 1
        else:
            size = 0
            spatial_method = self.spatial_methods[variable.domain[0]]
            repeats = spatial_method._get_auxiliary_domain_repeats(
                variable.auxiliary_domains
            )
            for dom in variable.domain:
                size += spatial_method.mesh[dom].npts_for_broadcast_to_nodes * repeats
            return size

    def _preprocess_external_variables(self, model):
        """
        A method to preprocess external variables so that they are
        compatible with the spatial method. For example, in finite
        volume, the user will supply a vector of values valid on the
        cell centres. However, for model processing, we also require
        the boundary edge fluxes. Therefore, we extrapolate and add
        the boundary fluxes to the boundary conditions, which are
        employed in generating the grad and div matrices.
        The processing is delegated to spatial methods as
        the preprocessing required for finite volume and finite
        element will be different.
        """

        for var in model.external_variables:
            if var.domain != []:
                new_bcs = self.spatial_methods[
                    var.domain[0]
                ].preprocess_external_variables(var)

                model.boundary_conditions.update(new_bcs)

    def set_external_variables(self, model):
        """
        Add external variables to the list of variables to account for, being careful
        about concatenations
        """
        for var in model.external_variables:
            # Find the name in the model variables
            # Look up dictionary key based on value
            try:
                idx = [x.id for x in model.variables.values()].index(var.id)
            except ValueError:
                raise ValueError(
                    """
                    Variable {} must be in the model.variables dictionary to be set
                    as an external variable
                    """.format(
                        var
                    )
                )
            name = list(model.variables.keys())[idx]
            if isinstance(var, pybamm.Variable):
                # No need to keep track of the parent
                self.external_variables[(name, None)] = var
            elif isinstance(var, pybamm.Concatenation):
                start = 0
                end = 0
                for child in var.children:
                    dom = child.domain[0]
                    if (
                        self.spatial_methods[dom]._get_auxiliary_domain_repeats(
                            child.domains
                        )
                        > 1
                    ):
                        raise NotImplementedError(
                            "Cannot create 2D external variable with concatenations"
                        )
                    end += self._get_variable_size(child)
                    # Keep a record of the parent
                    self.external_variables[(name, (var, start, end))] = child
                    start = end

    def set_internal_boundary_conditions(self, model):
        """
        A method to set the internal boundary conditions for the submodel.
        These are required to properly calculate the gradient.
        Note: this method modifies the state of self.boundary_conditions.
        """

        def boundary_gradient(left_symbol, right_symbol):

            pybamm.logger.debug(
                "Calculate boundary gradient ({} and {})".format(
                    left_symbol, right_symbol
                )
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

        # bc_key_ids = [key.id for key in list(model.boundary_conditions.keys())]
        bc_key_ids = list(self.bcs.keys())

        internal_bcs = {}
        for var in model.boundary_conditions.keys():
            if isinstance(var, pybamm.Concatenation):
                children = var.children

                first_child = children[0]
                first_orphan = first_child.new_copy()
                next_child = children[1]
                next_orphan = next_child.new_copy()

                lbc = self.bcs[var.id]["left"]
                rbc = (boundary_gradient(first_orphan, next_orphan), "Neumann")

                if first_child.id not in bc_key_ids:
                    internal_bcs.update({first_child.id: {"left": lbc, "right": rbc}})

                for i, _ in enumerate(children[1:-1]):
                    current_child = next_child
                    current_orphan = next_orphan
                    next_child = children[i + 2]
                    next_orphan = next_child.new_copy()

                    lbc = rbc
                    rbc = (boundary_gradient(current_orphan, next_orphan), "Neumann")
                    if current_child.id not in bc_key_ids:
                        internal_bcs.update(
                            {current_child.id: {"left": lbc, "right": rbc}}
                        )

                lbc = rbc
                rbc = self.bcs[var.id]["right"]
                if children[-1].id not in bc_key_ids:
                    internal_bcs.update({children[-1].id: {"left": lbc, "right": rbc}})

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
        processed_initial_conditions = self.process_dict(model.initial_conditions)

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
            processed_bcs[key.id] = {}

            # Handle any boundary conditions applied on the tabs
            if any("tab" in side for side in list(bcs.keys())):
                bcs = self.check_tab_conditions(key, bcs)

            # Process boundary conditions
            for side, bc in bcs.items():
                eqn, typ = bc
                pybamm.logger.debug("Discretise {} ({} bc)".format(key, side))
                processed_eqn = self.process_symbol(eqn)
                processed_bcs[key.id][side] = (processed_eqn, typ)

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
                """Boundary conditions can only be applied on the tabs in the domain
            'current collector', but {} has domain {}""".format(
                    symbol, domain
                )
            )
        # Replace keys with "left" and "right" as appropriate for 1D meshes
        if isinstance(mesh, pybamm.SubMesh1D):
            # send boundary conditions applied on the tabs to "left" or "right"
            # depending on the tab location stored in the mesh
            for tab in ["negative tab", "positive tab"]:
                if any(tab in side for side in list(bcs.keys())):
                    bcs[mesh.tabs[tab]] = bcs.pop(tab)
            # if there was a tab at either end, then the boundary conditions
            # have now been set on "left" and "right" as required by the spatial
            # method, so there is no need to further modify the bcs dict
            if all(side in list(bcs.keys()) for side in ["left", "right"]):
                pass
            # if both tabs are located at z=0 then the "right" boundary condition
            # (at z=1) is the condition for "no tab"
            elif "left" in list(bcs.keys()):
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
        :class:`pybamm.Matrix`
            The inverse of the ode part of the mass matrix (required by solvers
            which only accept the ODEs in explicit form)
        """
        # Create list of mass matrices for each equation to be put into block
        # diagonal mass matrix for the model
        mass_list = []
        mass_inv_list = []

        # get a list of model rhs variables that are sorted according to
        # where they are in the state vector
        model_variables = model.rhs.keys()
        model_slices = []
        for v in model_variables:
            if isinstance(v, pybamm.Concatenation):
                model_slices.append(
                    slice(
                        self.y_slices[v.children[0].id][0].start,
                        self.y_slices[v.children[-1].id][0].stop,
                    )
                )
            else:
                model_slices.append(self.y_slices[v.id][0])
        sorted_model_variables = [
            v for _, v in sorted(zip(model_slices, model_variables))
        ]

        # Process mass matrices for the differential equations
        for var in sorted_model_variables:
            if var.domain == []:
                # If variable domain empty then mass matrix is just 1
                mass_list.append(1.0)
                mass_inv_list.append(1.0)
            else:
                mass = (
                    self.spatial_methods[var.domain[0]]
                    .mass_matrix(var, self.bcs)
                    .entries
                )
                mass_list.append(mass)
                if isinstance(
                    self.spatial_methods[var.domain[0]],
                    (pybamm.ZeroDimensionalSpatialMethod, pybamm.FiniteVolume),
                ):
                    # for 0D methods the mass matrix is just a scalar 1 and for
                    # finite volumes the mass matrix is identity, so no need to
                    # compute the inverse
                    mass_inv_list.append(mass)
                else:
                    # inverse is more efficient in csc format
                    mass_inv = inv(csc_matrix(mass))
                    mass_inv_list.append(mass_inv)

        # Create lumped mass matrix (of zeros) of the correct shape for the
        # discretised algebraic equations
        if model.algebraic.keys():
            mass_algebraic_size = model.concatenated_algebraic.shape[0]
            mass_algebraic = csr_matrix((mass_algebraic_size, mass_algebraic_size))
            mass_list.append(mass_algebraic)

        # Create block diagonal (sparse) mass matrix (if model is not empty)
        # and inverse (if model has odes)
        if len(model.rhs) + len(model.algebraic) > 0:
            mass_matrix = pybamm.Matrix(block_diag(mass_list, format="csr"))
            if len(model.rhs) > 0:
                mass_matrix_inv = pybamm.Matrix(block_diag(mass_inv_list, format="csr"))
            else:
                mass_matrix_inv = None
        else:
            mass_matrix, mass_matrix_inv = None, None

        return mass_matrix, mass_matrix_inv

    def create_jacobian(self, model):
        """Creates Jacobian of the discretised model.
        Note that the model is assumed to be of the form M*y_dot = f(t,y), where
        M is the (possibly singular) mass matrix. The Jacobian is df/dy.

        Note: At present, calculation of the Jacobian is deferred until after
        simplification, since it is much faster to compute the Jacobian of the
        simplified model. However, in some use cases (e.g. running the same
        model multiple times but with different parameters) it may be more
        efficient to compute the Jacobian once, before simplification, so that
        parameters in the Jacobian can be updated (see PR #670).

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Discretised model. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})

        Returns
        -------
        :class:`pybamm.Concatenation`
            The expression trees corresponding to the Jacobian of the model
        """
        # create state vector to differentiate with respect to
        y = pybamm.StateVector(slice(0, np.size(model.concatenated_initial_conditions)))
        # set up Jacobian object, for re-use of dict
        jacobian = pybamm.Jacobian()

        # calculate Jacobian of rhs by equation
        jac_rhs_eqn_dict = {}
        for eqn_key, eqn in model.rhs.items():
            pybamm.logger.debug(
                "Calculating block of Jacobian for {!r}".format(eqn_key.name)
            )
            jac_rhs_eqn_dict[eqn_key] = jacobian.jac(eqn, y)
        jac_rhs = self._concatenate_in_order(jac_rhs_eqn_dict, sparse=True)

        # calculate Jacobian of algebraic by equation
        jac_algebraic_eqn_dict = {}
        for eqn_key, eqn in model.algebraic.items():
            pybamm.logger.debug(
                "Calculating block of Jacobian for {!r}".format(eqn_key.name)
            )
            jac_algebraic_eqn_dict[eqn_key] = jacobian.jac(eqn, y)
        jac_algebraic = self._concatenate_in_order(jac_algebraic_eqn_dict, sparse=True)

        # full Jacobian
        if model.rhs.keys() and model.algebraic.keys():
            jac = pybamm.SparseStack(jac_rhs, jac_algebraic)
        elif not model.algebraic.keys():
            jac = jac_rhs
        else:
            jac = jac_algebraic

        return jac, jac_rhs, jac_algebraic

    def process_dict(self, var_eqn_dict):
        """Discretise a dictionary of {variable: equation}, broadcasting if necessary
        (can be model.rhs, model.algebraic, model.initial_conditions or
        model.variables).

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise
            (can be model.rhs, model.algebraic, model.initial_conditions or
            model.variables)

        Returns
        -------
        new_var_eqn_dict : dict
            Discretised equations

        """
        new_var_eqn_dict = {}
        for eqn_key, eqn in var_eqn_dict.items():
            # Broadcast if the equation evaluates to a number(e.g. Scalar)
            if eqn.evaluates_to_number() and not isinstance(eqn_key, str):
                eqn = pybamm.FullBroadcast(
                    eqn, eqn_key.domain, eqn_key.auxiliary_domains
                )

            # note we are sending in the key.id here so we don't have to
            # keep calling .id
            pybamm.logger.debug("Discretise {!r}".format(eqn_key))

            processed_eqn = self.process_symbol(eqn)

            new_var_eqn_dict[eqn_key] = processed_eqn

        return new_var_eqn_dict

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
        try:
            return self._discretised_symbols[symbol.id]
        except KeyError:
            discretised_symbol = self._process_symbol(symbol)
            self._discretised_symbols[symbol.id] = discretised_symbol
            discretised_symbol.test_shape()
            # Assign mesh as an attribute to the processed variable
            if symbol.domain != []:
                discretised_symbol.mesh = self.mesh.combine_submeshes(*symbol.domain)
            else:
                discretised_symbol.mesh = None
            # Assign secondary mesh
            if "secondary" in symbol.auxiliary_domains:
                discretised_symbol.secondary_mesh = self.mesh.combine_submeshes(
                    *symbol.auxiliary_domains["secondary"]
                )
            else:
                discretised_symbol.secondary_mesh = None
            return discretised_symbol

    def _process_symbol(self, symbol):
        """ See :meth:`Discretisation.process_symbol()`. """

        if symbol.domain != []:
            spatial_method = self.spatial_methods[symbol.domain[0]]
            # If boundary conditions are provided, need to check for BCs on tabs
            if self.bcs:
                key_id = list(self.bcs.keys())[0]
                if any("tab" in side for side in list(self.bcs[key_id].keys())):
                    self.bcs[key_id] = self.check_tab_conditions(
                        symbol, self.bcs[key_id]
                    )

        if isinstance(symbol, pybamm.BinaryOperator):
            # Pre-process children
            left, right = symbol.children
            disc_left = self.process_symbol(left)
            disc_right = self.process_symbol(right)
            if symbol.domain == []:
                return symbol._binary_new_copy(disc_left, disc_right)
            else:
                return spatial_method.process_binary_operators(
                    symbol, left, right, disc_left, disc_right
                )
        elif isinstance(symbol, pybamm.UnaryOperator):
            child = symbol.child
            disc_child = self.process_symbol(child)
            if child.domain != []:
                child_spatial_method = self.spatial_methods[child.domain[0]]

            if isinstance(symbol, pybamm.Gradient):
                return child_spatial_method.gradient(child, disc_child, self.bcs)

            elif isinstance(symbol, pybamm.Divergence):
                return child_spatial_method.divergence(child, disc_child, self.bcs)

            elif isinstance(symbol, pybamm.Laplacian):
                return child_spatial_method.laplacian(child, disc_child, self.bcs)

            elif isinstance(symbol, pybamm.Gradient_Squared):
                return child_spatial_method.gradient_squared(
                    child, disc_child, self.bcs
                )

            elif isinstance(symbol, pybamm.Mass):
                return child_spatial_method.mass_matrix(child, self.bcs)

            elif isinstance(symbol, pybamm.BoundaryMass):
                return child_spatial_method.boundary_mass_matrix(child, self.bcs)

            elif isinstance(symbol, pybamm.IndefiniteIntegral):
                return child_spatial_method.indefinite_integral(
                    child, disc_child, "forward"
                )
            elif isinstance(symbol, pybamm.BackwardIndefiniteIntegral):
                return child_spatial_method.indefinite_integral(
                    child, disc_child, "backward"
                )

            elif isinstance(symbol, pybamm.Integral):
                out = child_spatial_method.integral(child, disc_child)
                out.copy_domains(symbol)
                return out

            elif isinstance(symbol, pybamm.DefiniteIntegralVector):
                return child_spatial_method.definite_integral_matrix(
                    child.domains, vector_type=symbol.vector_type
                )

            elif isinstance(symbol, pybamm.BoundaryIntegral):
                return child_spatial_method.boundary_integral(
                    child, disc_child, symbol.region
                )

            elif isinstance(symbol, pybamm.Broadcast):
                # Broadcast new_child to the domain specified by symbol.domain
                # Different discretisations may broadcast differently
                if symbol.domain == []:
                    symbol = disc_child * pybamm.Vector(np.array([1]))
                else:
                    symbol = spatial_method.broadcast(
                        disc_child,
                        symbol.domain,
                        symbol.auxiliary_domains,
                        symbol.broadcast_type,
                    )
                return symbol

            elif isinstance(symbol, pybamm.DeltaFunction):
                return spatial_method.delta_function(symbol, disc_child)

            elif isinstance(symbol, pybamm.BoundaryOperator):
                # if boundary operator applied on "negative tab" or
                # "positive tab" *and* the mesh is 1D then change side to
                # "left" or "right" as appropriate
                if symbol.side in ["negative tab", "positive tab"]:
                    mesh = self.mesh[symbol.children[0].domain[0]]
                    if isinstance(mesh, pybamm.SubMesh1D):
                        symbol.side = mesh.tabs[symbol.side]
                return child_spatial_method.boundary_value_or_flux(
                    symbol, disc_child, self.bcs
                )

            else:
                return symbol._unary_new_copy(disc_child)

        elif isinstance(symbol, pybamm.Function):
            disc_children = [self.process_symbol(child) for child in symbol.children]
            return symbol._function_new_copy(disc_children)

        elif isinstance(symbol, pybamm.VariableDot):
            return pybamm.StateVectorDot(
                *self.y_slices[symbol.get_variable().id],
                domain=symbol.domain,
                auxiliary_domains=symbol.auxiliary_domains
            )

        elif isinstance(symbol, pybamm.Variable):
            # Check if variable is a standard variable or an external variable
            if any(symbol.id == var.id for var in self.external_variables.values()):
                # Look up dictionary key based on value
                idx = [x.id for x in self.external_variables.values()].index(symbol.id)
                name, parent_and_slice = list(self.external_variables.keys())[idx]
                if parent_and_slice is None:
                    # Variable didn't come from a concatenation so we can just create a
                    # normal external variable using the symbol's name
                    return pybamm.ExternalVariable(
                        symbol.name,
                        size=self._get_variable_size(symbol),
                        domain=symbol.domain,
                        auxiliary_domains=symbol.auxiliary_domains,
                    )
                else:
                    # We have to use a special name since the concatenation doesn't have
                    # a very informative name. Needs improving
                    parent, start, end = parent_and_slice
                    ext = pybamm.ExternalVariable(
                        name,
                        size=self._get_variable_size(parent),
                        domain=parent.domain,
                        auxiliary_domains=parent.auxiliary_domains,
                    )
                    out = ext[slice(start, end)]
                    out.domain = symbol.domain
                    return out

            else:
                # add a try except block for a more informative error if a variable
                # can't be found. This should usually be caught earlier by
                # model.check_well_posedness, but won't be if debug_mode is False
                try:
                    y_slices = self.y_slices[symbol.id]
                except KeyError:
                    raise pybamm.ModelError(
                        """
                        No key set for variable '{}'. Make sure it is included in either
                        model.rhs, model.algebraic, or model.external_variables in an
                        unmodified form (e.g. not Broadcasted)
                        """.format(
                            symbol.name
                        )
                    )
                return pybamm.StateVector(
                    *y_slices,
                    domain=symbol.domain,
                    auxiliary_domains=symbol.auxiliary_domains
                )

        elif isinstance(symbol, pybamm.SpatialVariable):
            return spatial_method.spatial_variable(symbol)

        elif isinstance(symbol, pybamm.Concatenation):
            new_children = [self.process_symbol(child) for child in symbol.children]
            new_symbol = spatial_method.concatenation(new_children)

            return new_symbol

        elif isinstance(symbol, pybamm.InputParameter):
            # Return a new copy of the input parameter, but set the expected size
            # according to the domain of the input parameter
            expected_size = self._get_variable_size(symbol)
            new_input_parameter = symbol.new_copy()
            new_input_parameter.set_expected_size(expected_size)
            return new_input_parameter

        else:
            # Backup option: return new copy of the object
            try:
                return symbol.new_copy()
            except NotImplementedError:
                raise NotImplementedError(
                    "Cannot discretise symbol of type '{}'".format(type(symbol))
                )

    def concatenate(self, *symbols, sparse=False):
        if sparse:
            return pybamm.SparseStack(*symbols)
        else:
            return pybamm.NumpyConcatenation(*symbols)

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
        unpacked_variables = []
        slices = []
        for symbol in var_eqn_dict.keys():
            if isinstance(symbol, pybamm.Concatenation):
                unpacked_variables.extend([var for var in symbol.children])
                # must append the slice for the whole concatenation, so that equations
                # get sorted correctly
                slices.append(
                    slice(
                        self.y_slices[symbol.children[0].id][0].start,
                        self.y_slices[symbol.children[-1].id][0].stop,
                    )
                )
            else:
                unpacked_variables.append(symbol)
                slices.append(self.y_slices[symbol.id][0])

        if check_complete:
            # Check keys from the given var_eqn_dict against self.y_slices
            ids = {v.id for v in unpacked_variables}
            external_id = {v.id for v in self.external_variables.values()}
            for var in self.external_variables.values():
                child_ids = {child.id for child in var.children}
                external_id = external_id.union(child_ids)
            y_slices_with_external_removed = set(self.y_slices.keys()).difference(
                external_id
            )
            if ids != y_slices_with_external_removed:
                given_variable_names = [v.name for v in var_eqn_dict.keys()]
                raise pybamm.ModelError(
                    "Initial conditions are insufficient. Only "
                    "provided for {} ".format(given_variable_names)
                )

        equations = list(var_eqn_dict.values())

        # sort equations according to slices
        sorted_equations = [eq for _, eq in sorted(zip(slices, equations))]

        return self.concatenate(*sorted_equations, sparse=sparse)

    def check_model(self, model):
        """ Perform some basic checks to make sure the discretised model makes sense."""
        self.check_initial_conditions(model)
        self.check_initial_conditions_rhs(model)
        self.check_variables(model)

    def check_initial_conditions(self, model):
        """Check initial conditions are a numpy array"""
        # Individual
        for var, eqn in model.initial_conditions.items():
            assert isinstance(
                eqn.evaluate(t=0, inputs="shape test"), np.ndarray
            ), pybamm.ModelError(
                """
                initial_conditions must be numpy array after discretisation but they are
                {} for variable '{}'.
                """.format(
                    type(eqn.evaluate(t=0, inputs="shape test")), var
                )
            )
        # Concatenated
        assert (
            type(
                model.concatenated_initial_conditions.evaluate(t=0, inputs="shape test")
            )
            is np.ndarray
        ), pybamm.ModelError(
            """
            Concatenated initial_conditions must be numpy array after discretisation but
            they are {}.
            """.format(
                type(model.concatenated_initial_conditions)
            )
        )

    def check_initial_conditions_rhs(self, model):
        """Check initial conditions and rhs have the same shape"""
        y0 = model.concatenated_initial_conditions
        # Individual
        for var in model.rhs.keys():
            assert (
                model.rhs[var].shape == model.initial_conditions[var].shape
            ), pybamm.ModelError(
                """
                rhs and initial_conditions must have the same shape after discretisation
                but rhs.shape = {} and initial_conditions.shape = {} for variable '{}'.
                """.format(
                    model.rhs[var].shape, model.initial_conditions[var].shape, var
                )
            )
        # Concatenated
        assert (
            model.concatenated_rhs.shape[0] + model.concatenated_algebraic.shape[0]
            == y0.shape[0]
        ), pybamm.ModelError(
            """
            Concatenation of (rhs, algebraic) and initial_conditions must have the
            same shape after discretisation but rhs.shape = {}, algebraic.shape = {},
            and initial_conditions.shape = {}.
            """.format(
                model.concatenated_rhs.shape,
                model.concatenated_algebraic.shape,
                y0.shape,
            )
        )

    def check_variables(self, model):
        """
        Check variables in variable list against rhs
        Be lenient with size check if the variable in model.variables is broadcasted, or
        a concatenation
        (if broadcasted, variable is a multiplication with a vector of ones)
        """
        for rhs_var in model.rhs.keys():
            if rhs_var.name in model.variables.keys():
                var = model.variables[rhs_var.name]

                different_shapes = not np.array_equal(
                    model.rhs[rhs_var].shape, var.shape
                )

                not_concatenation = not isinstance(var, pybamm.Concatenation)

                not_mult_by_one_vec = not (
                    isinstance(var, pybamm.Multiplication)
                    and isinstance(var.right, pybamm.Vector)
                    and np.all(var.right.entries == 1)
                )

                if different_shapes and not_concatenation and not_mult_by_one_vec:
                    raise pybamm.ModelError(
                        """
                    variable and its eqn must have the same shape after discretisation
                    but variable.shape = {} and rhs.shape = {} for variable '{}'.
                    """.format(
                            var.shape, model.rhs[rhs_var].shape, var
                        )
                    )
