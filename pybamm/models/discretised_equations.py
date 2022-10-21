#
# Base model class
#
import pybamm
from scipy.sparse import block_diag, csc_matrix, csr_matrix
from scipy.sparse.linalg import inv

_READ_ONLY_ATTRIBUTES = [
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


class _DiscretisedEquations(pybamm._BaseProcessedEquations):
    """
    Class containing discretised equations.

    **Extends:** :class:`pybamm._BaseProcessedEquations`
    """

    def __init__(self, discretisation, *args, y_slices, bounds):
        # Save discretisation used to create this model
        self._discretisation = discretisation.copy_with_discretised_symbols()

        super().__init__(*args)
        self._y_slices = y_slices
        self._bounds = bounds

        self.create_concatenated_attributes()
        self.create_mass_matrix()

    def create_concatenated_attributes(self):
        disc = self._discretisation
        self._concatenated_rhs = disc._concatenate_in_order(self.rhs)
        self._concatenated_algebraic = disc._concatenate_in_order(self.algebraic)
        self._concatenated_initial_conditions = disc._concatenate_in_order(
            self.initial_conditions, check_complete=True
        )

        self._len_rhs = self._concatenated_rhs.size
        self._len_alg = self._concatenated_algebraic.size
        self._len_rhs_and_alg = self._len_rhs + self._len_alg

    def create_mass_matrix(self):
        """Creates mass matrix of the discretised model.
        Note that the model is assumed to be of the form M*y_dot = f(t,y), where
        M is the (possibly singular) mass matrix.
        """
        # Create list of mass matrices for each equation to be put into block
        # diagonal mass matrix for the model
        mass_list = []
        mass_inv_list = []

        # get a list of model rhs variables that are sorted according to
        # where they are in the state vector
        variables = self.rhs.keys()
        slices = []
        for v in variables:
            slices.append(self._discretisation.y_slices[v][0])
        sorted_variables = [v for _, v in sorted(zip(slices, variables))]

        # Process mass matrices for the differential equations
        for var in sorted_variables:
            if var.domain == []:
                # If variable domain empty then mass matrix is just 1
                mass_list.append(1.0)
                mass_inv_list.append(1.0)
            else:
                mass = (
                    self._discretisation.spatial_methods[var.domain[0]]
                    .mass_matrix(var, self._discretisation.bcs)
                    .entries
                )
                mass_list.append(mass)
                if isinstance(
                    self._discretisation.spatial_methods[var.domain[0]],
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
        if self._len_alg > 0:
            mass_algebraic_size = self._len_alg
            mass_algebraic = csr_matrix((mass_algebraic_size, mass_algebraic_size))
            mass_list.append(mass_algebraic)

        # Create block diagonal (sparse) mass matrix (if model is not empty)
        # and inverse (if model has odes)
        if self._len_rhs_and_alg > 0:
            mass_matrix = pybamm.Matrix(block_diag(mass_list, format="csr"))
            if len(self.rhs) > 0:
                rhs_mass_matrix_inv = pybamm.Matrix(
                    block_diag(mass_inv_list, format="csr")
                )
            else:
                rhs_mass_matrix_inv = None
        else:
            mass_matrix, rhs_mass_matrix_inv = None, None

        self._mass_matrix = mass_matrix
        self._mass_matrix_inv = rhs_mass_matrix_inv

    def __getattr__(self, name):
        if name in _READ_ONLY_ATTRIBUTES:
            return getattr(self, "_" + name)
        else:
            return self.__getattribute__(name)

    def __setattr__(self, name, value):
        if name in _READ_ONLY_ATTRIBUTES:
            raise AttributeError("can't set attribute")
        else:
            super().__setattr__(name, value)

    def variables_update_function(self, variable):
        return self._discretisation.process_symbol(variable)
