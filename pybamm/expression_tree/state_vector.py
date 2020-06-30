#
# State Vector class
#
import pybamm

import numpy as np
from scipy.sparse import csr_matrix, vstack


class StateVectorBase(pybamm.Symbol):
    """
    node in the expression tree that holds a slice to read from an external vector type

    Parameters
    ----------

    y_slice: slice
        the slice of an external y to read
    name: str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list
    auxiliary_domains : dict of str, optional
        dictionary of auxiliary domains
    evaluation_array : list, optional
        List of boolean arrays representing slices. Default is None, in which case the
        evaluation_array is computed from y_slices.

    *Extends:* :class:`pybamm.Symbol`
    """

    def __init__(
        self,
        *y_slices,
        base_name="y",
        name=None,
        domain=None,
        auxiliary_domains=None,
        evaluation_array=None,
    ):
        for y_slice in y_slices:
            if not isinstance(y_slice, slice):
                raise TypeError("all y_slices must be slice objects")
        if name is None:
            if y_slices[0].start is None:
                name = base_name + "[:{:d}]".format(y_slice.stop)
            else:
                name = base_name + "[{:d}:{:d}".format(
                    y_slices[0].start, y_slices[0].stop
                )
            if len(y_slices) > 1:
                name += ",{:d}:{:d}".format(y_slices[1].start, y_slices[1].stop)
                if len(y_slices) > 2:
                    name += ",...,{:d}:{:d}]".format(
                        y_slices[-1].start, y_slices[-1].stop
                    )
                else:
                    name += "]"
            else:
                name += "]"
        if domain is None:
            domain = []
        if auxiliary_domains is None:
            auxiliary_domains = {}
        self._y_slices = y_slices
        self._first_point = y_slices[0].start
        self._last_point = y_slices[-1].stop
        self.set_evaluation_array(y_slices, evaluation_array)
        super().__init__(name=name, domain=domain, auxiliary_domains=auxiliary_domains)

    @property
    def y_slices(self):
        return self._y_slices

    @property
    def first_point(self):
        return self._first_point

    @property
    def last_point(self):
        return self._last_point

    @property
    def evaluation_array(self):
        """Array to use for evaluating"""
        return self._evaluation_array

    @property
    def size(self):
        return self.evaluation_array.count(True)

    def set_evaluation_array(self, y_slices, evaluation_array):
        "Set evaluation array using slices"
        if evaluation_array is not None and pybamm.settings.debug_mode is False:
            self._evaluation_array = evaluation_array
        else:
            array = np.zeros(y_slices[-1].stop)
            for y_slice in y_slices:
                array[y_slice] = True
            self._evaluation_array = [bool(x) for x in array]

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()` """
        self._id = hash(
            (self.__class__, self.name, tuple(self.evaluation_array))
            + tuple(self.domain)
        )

    def _jac_diff_vector(self, variable):
        """
        Differentiate a slice of a StateVector of size m with respect to another slice
        of a different StateVector of size n. This returns a (sparse) zero matrix of
        size m x n

        Parameters
        ----------
        variable : :class:`pybamm.Symbol`
            The variable with respect to which to differentiate

        """
        if len(variable.y_slices) > 1:
            raise NotImplementedError(
                "Jacobian only implemented for a single-slice StateVector"
            )
        slices_size = self.y_slices[0].stop - self.y_slices[0].start
        variable_size = variable.last_point - variable.first_point

        # Return zeros of correct size since no entries match
        return pybamm.Matrix(csr_matrix((slices_size, variable_size)))

    def _jac_same_vector(self, variable):
        """
        Differentiate a slice of a StateVector of size m with respect to another
        slice of a StateVector of size n. This returns a (sparse) matrix of size
        m x n with ones where the y slices match, and zeros elsewhere.

        Parameters
        ----------
        variable : :class:`pybamm.Symbol`
            The variable with respect to which to differentiate

        """
        if len(variable.y_slices) > 1:
            raise NotImplementedError(
                "Jacobian only implemented for a single-slice StateVector"
            )
        variable_y_indices = np.arange(variable.first_point, variable.last_point)

        jac = csr_matrix((0, np.size(variable_y_indices)))
        for y_slice in self.y_slices:
            # Get indices of state vectors
            slice_indices = np.arange(y_slice.start, y_slice.stop)

            # Return zeros of correct size if no entries match
            if np.size(np.intersect1d(slice_indices, variable_y_indices)) == 0:
                jac = csr_matrix((np.size(slice_indices), np.size(variable_y_indices)))
            else:
                # Populate entries corresponding to matching y slices, and shift so
                # that the matrix is the correct size
                row = np.intersect1d(slice_indices, variable_y_indices) - y_slice.start
                col = (
                    np.intersect1d(slice_indices, variable_y_indices)
                    - variable.first_point
                )
                data = np.ones_like(row)
                jac = vstack(
                    [
                        jac,
                        csr_matrix(
                            (data, (row, col)),
                            shape=(np.size(slice_indices), np.size(variable_y_indices)),
                        ),
                    ]
                )
        return pybamm.Matrix(jac)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return StateVector(
            *self.y_slices,
            name=self.name,
            domain=self.domain,
            auxiliary_domains=self.auxiliary_domains,
            evaluation_array=self.evaluation_array,
        )

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a StateVector.
        The size of a StateVector is the number of True elements in its evaluation_array
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return np.nan * np.ones((self.size, 1))


class StateVector(StateVectorBase):
    """
    node in the expression tree that holds a slice to read from an external vector type

    Parameters
    ----------

    y_slice: slice
        the slice of an external y to read
    name: str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list
    auxiliary_domains : dict of str, optional
        dictionary of auxiliary domains
    evaluation_array : list, optional
        List of boolean arrays representing slices. Default is None, in which case the
        evaluation_array is computed from y_slices.

    *Extends:* :class:`pybamm.StateVectorBase`
    """

    def __init__(
        self,
        *y_slices,
        name=None,
        domain=None,
        auxiliary_domains=None,
        evaluation_array=None,
    ):
        super().__init__(
            *y_slices,
            base_name="y",
            name=name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            evaluation_array=evaluation_array,
        )

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        if y is None:
            raise TypeError("StateVector cannot evaluate input 'y=None'")
        if y.shape[0] < len(self.evaluation_array):
            raise ValueError(
                "y is too short, so value with slice is smaller than expected"
            )

        out = (y[: len(self._evaluation_array)])[self._evaluation_array]
        if isinstance(out, np.ndarray) and out.ndim == 1:
            out = out[:, np.newaxis]
        return out

    def diff(self, variable):
        if variable.id == self.id:
            return pybamm.Scalar(1)
        if variable.id == pybamm.t.id:
            return StateVectorDot(
                *self._y_slices,
                name=self.name + "'",
                domain=self.domain,
                auxiliary_domains=self.auxiliary_domains,
                evaluation_array=self.evaluation_array,
            )
        else:
            return pybamm.Scalar(0)

    def _jac(self, variable):
        if isinstance(variable, pybamm.StateVector):
            return self._jac_same_vector(variable)
        elif isinstance(variable, pybamm.StateVectorDot):
            return self._jac_diff_vector(variable)


class StateVectorDot(StateVectorBase):
    """
    node in the expression tree that holds a slice to read from the ydot

    Parameters
    ----------

    y_slice: slice
        the slice of an external ydot to read
    name: str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list
    auxiliary_domains : dict of str, optional
        dictionary of auxiliary domains
    evaluation_array : list, optional
        List of boolean arrays representing slices. Default is None, in which case the
        evaluation_array is computed from y_slices.

    *Extends:* :class:`Array`
    """

    def __init__(
        self,
        *y_slices,
        name=None,
        domain=None,
        auxiliary_domains=None,
        evaluation_array=None,
    ):
        super().__init__(
            *y_slices,
            base_name="y_dot",
            name=name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            evaluation_array=evaluation_array,
        )

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        if y_dot is None:
            raise TypeError("StateVectorDot cannot evaluate input 'y_dot=None'")
        if y_dot.shape[0] < len(self.evaluation_array):
            raise ValueError(
                "y_dot is too short, so value with slice is smaller than expected"
            )

        out = (y_dot[: len(self._evaluation_array)])[self._evaluation_array]
        if isinstance(out, np.ndarray) and out.ndim == 1:
            out = out[:, np.newaxis]
        return out

    def diff(self, variable):
        if variable.id == self.id:
            return pybamm.Scalar(1)
        elif variable.id == pybamm.t.id:
            raise pybamm.ModelError(
                "cannot take second time derivative of a state vector"
            )
        else:
            return pybamm.Scalar(0)

    def _jac(self, variable):
        if isinstance(variable, pybamm.StateVectorDot):
            return self._jac_same_vector(variable)
        elif isinstance(variable, pybamm.StateVector):
            return self._jac_diff_vector(variable)
