#
# State Vector class
#
from __future__ import annotations

import warnings

import casadi
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, vstack

import pybamm
from pybamm.type_definitions import AuxiliaryDomainType, DomainsType, DomainType


def ranges_tile(ranges, start: int, stop: int) -> bool:
    """
    Whether sorted ``(start, stop)`` ranges cover ``[start, stop)`` exactly once.

    Parameters
    ----------
    ranges : iterable of tuple of int
        Half-open index ranges, sorted by start.
    start, stop : int
        The interval to cover.

    Returns
    -------
    bool
        True if the ranges are contiguous, non-overlapping and span the interval.
    """
    position = start
    for range_start, range_stop in ranges:
        if range_start != position:
            return False
        position = range_stop
    return position == stop


class StateVectorBase(pybamm.Symbol):
    """
    Node in the expression tree that holds a slice to read from an external vector type.

    Parameters
    ----------

    *y_slices : slice
        the slices of an external y to read
    name: str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list
    auxiliary_domains : dict of str, optional
        dictionary of auxiliary domains
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    evaluation_array : list, optional
        Deprecated and ignored: the entries read are always given by ``y_slices``.

    Raises
    ------
    :class:`pybamm.ShapeError`
        If a slice has a step or a negative bound, or the slices are not in
        increasing order or overlap.
    """

    __slots__ = ("_first_point", "_index_ranges", "_last_point", "_y_slices")
    # The merged index ranges define the entries read, however the slices split them
    _id_excluded_fields = ("_first_point", "_last_point", "_y_slices")

    def __init__(
        self,
        *y_slices: slice,
        base_name="y",
        name: str | None = None,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
        evaluation_array: list[bool] | None = None,
    ):
        for y_slice in y_slices:
            if not isinstance(y_slice, slice):
                raise TypeError("all y_slices must be slice objects")
        if evaluation_array is not None:
            warnings.warn(
                "The 'evaluation_array' argument of StateVector is deprecated and "
                "ignored; the entries read are given by the y_slices.",
                DeprecationWarning,
                stacklevel=2,
            )
        if name is None:
            if y_slices[0].start is None:
                name = base_name + f"[0:{y_slice.stop:d}"
            else:
                name = base_name + f"[{y_slices[0].start:d}:{y_slices[0].stop:d}"
            if len(y_slices) > 1:
                name += f",{y_slices[1].start:d}:{y_slices[1].stop:d}"
                if len(y_slices) > 2:
                    name += f",...,{y_slices[-1].start:d}:{y_slices[-1].stop:d}]"
                else:
                    name += "]"
            else:
                name += "]"
        self._y_slices = y_slices
        self._first_point = y_slices[0].start
        self._last_point = y_slices[-1].stop
        self._index_ranges = self._merge_slices(y_slices)
        super().__init__(
            name=name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
        )

    @classmethod
    def _from_json(cls, snippet: dict):
        y_slices = [slice(s["start"], s["stop"], s["step"]) for s in snippet["y_slice"]]

        return cls(*y_slices, name=snippet["name"], domains=snippet["domains"])

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
        """Boolean mask, as long as ``last_point``, of the entries of y read."""
        array = np.zeros(self._last_point, dtype=bool)
        for start, stop in self._index_ranges:
            array[start:stop] = True
        return array.tolist()

    @property
    def size(self):
        return sum(stop - start for start, stop in self._index_ranges)

    def _read(self, vector):
        """The entries of ``vector`` this state vector reads, as a column."""
        if len(self._index_ranges) <= 1:
            start, stop = self._index_ranges[0] if self._index_ranges else (0, 0)
            out = vector[start:stop].copy()
        else:
            out = np.concatenate(
                [vector[start:stop] for start, stop in self._index_ranges]
            )
        if isinstance(out, np.ndarray) and out.ndim == 1:
            out = out[:, np.newaxis]
        return out

    @staticmethod
    def _merge_slices(y_slices) -> tuple[tuple[int, int], ...]:
        """
        The entries read by ``y_slices``, as ``(start, stop)`` ranges with adjacent
        slices merged.

        Raises
        ------
        :class:`pybamm.ShapeError`
            If a slice has a step or a negative bound, or the slices are not
            increasing or overlap.
        """
        ranges: list[list[int]] = []
        for y_slice in y_slices:
            if y_slice.step not in (None, 1):
                raise pybamm.ShapeError("y_slices must not have a step")
            start = y_slice.start or 0
            if start < 0 or y_slice.stop < 0:
                raise pybamm.ShapeError("y_slices must not have negative bounds")
            if ranges and start < ranges[-1][1]:
                raise pybamm.ShapeError(
                    "y_slices must be in increasing order and must not overlap"
                )
            if ranges and start == ranges[-1][1]:
                ranges[-1][1] = y_slice.stop
            elif y_slice.stop > start:
                ranges.append([start, y_slice.stop])
        return tuple((start, stop) for start, stop in ranges)

    def __setstate__(self, state):
        if "_evaluation_array" in state:  # pickled when the mask was stored
            state = {
                key: value for key, value in state.items() if key != "_evaluation_array"
            }
            state["_index_ranges"] = self._merge_slices(state["_y_slices"])
        super().__setstate__(state)

    def set_evaluation_array(self, y_slices, evaluation_array):
        """Set the slices to read (deprecated); ``evaluation_array`` is ignored."""
        pybamm.expression_tree.symbol._warn_mutation(
            "set_evaluation_array",
            "Construct a new StateVector with the desired slices.",
        )
        object.__setattr__(self, "_y_slices", tuple(y_slices))
        object.__setattr__(self, "_first_point", y_slices[0].start)
        object.__setattr__(self, "_last_point", y_slices[-1].stop)
        object.__setattr__(self, "_index_ranges", self._merge_slices(y_slices))

    def _jac_diff_vector(self, variable: pybamm.StateVectorBase):
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

    def _jac_same_vector(self, variable: pybamm.StateVectorBase):
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

    def create_copy(
        self,
        new_children=None,
        perform_simplifications=True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return self.__class__(
            *self.y_slices,
            name=self.name,
            domains=self._domains,
        )

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a StateVector.
        The size of a StateVector is the number of True elements in its evaluation_array
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return np.nan * np.ones((self.size, 1))

    def to_json(self):
        """
        Method to serialise a StateVector object into JSON.
        """

        json_dict = {
            "name": self.name,
            "domains": self.domains,
            "y_slice": [
                {
                    "start": y.start,
                    "stop": y.stop,
                    "step": y.step,
                }  # are there ever more than 1?
                for y in self.y_slices
            ],
            "evaluation_array": list(self.evaluation_array),
        }

        return json_dict


class StateVector(StateVectorBase):
    """
    Node in the expression tree that holds a slice to read from an external vector type.

    Parameters
    ----------

    *y_slices : slice
        the slices of an external y to read
    name: str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list
    auxiliary_domains : dict of str, optional
        dictionary of auxiliary domains
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    evaluation_array : list, optional
        List of boolean arrays representing slices. Default is None, in which case the
        evaluation_array is computed from y_slices.
    """

    __slots__ = ()

    def __init__(
        self,
        *y_slices: slice,
        name: str | None = None,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
        evaluation_array: list[bool] | None = None,
    ):
        super().__init__(
            *y_slices,
            base_name="y",
            name=name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
            evaluation_array=evaluation_array,
        )

    def _base_evaluate(
        self,
        t: float | None = None,
        y: npt.NDArray[np.float64] | None = None,
        y_dot: npt.NDArray[np.float64] | None = None,
        inputs: dict | str | None = None,
    ):
        """See :meth:`pybamm.Symbol._base_evaluate()`."""
        if y is None:
            raise TypeError("StateVector cannot evaluate input 'y=None'")
        if y.shape[0] < self._last_point:
            raise ValueError(
                "y is too short, so value with slice is smaller than expected"
            )
        return self._read(y)

    def diff(self, variable: pybamm.Symbol):
        if variable == self:
            return pybamm.Scalar(1)
        if variable == pybamm.t:
            return StateVectorDot(
                *self._y_slices,
                name=self.name + "'",
                domains=self._domains,
            )
        else:
            return pybamm.Scalar(0)

    def _to_casadi(self, t, y, y_dot, inputs, casadi_symbols):
        """See :meth:`pybamm.Symbol._to_casadi()`."""
        if y is None:
            raise ValueError("Must provide a 'y' for converting state vectors")
        return casadi.vertcat(*[y[y_slice] for y_slice in self.y_slices])

    def _jac(self, variable: pybamm.StateVector | pybamm.StateVectorDot):
        if isinstance(variable, pybamm.StateVector):
            return self._jac_same_vector(variable)
        elif isinstance(variable, pybamm.StateVectorDot):
            return self._jac_diff_vector(variable)


class StateVectorDot(StateVectorBase):
    """
    Node in the expression tree that holds a slice to read from the ydot.

    Parameters
    ----------

    *y_slices : slice
        the slices of an external ydot to read
    name: str, optional
        the name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list
    auxiliary_domains : dict of str, optional
        dictionary of auxiliary domains
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    evaluation_array : list, optional
        List of boolean arrays representing slices. Default is None, in which case the
        evaluation_array is computed from y_slices.
    """

    __slots__ = ()

    def __init__(
        self,
        *y_slices: slice,
        name: str | None = None,
        domain: DomainType = None,
        auxiliary_domains: AuxiliaryDomainType = None,
        domains: DomainsType = None,
        evaluation_array: list[bool] | None = None,
    ):
        super().__init__(
            *y_slices,
            base_name="y_dot",
            name=name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
            evaluation_array=evaluation_array,
        )

    def _base_evaluate(
        self,
        t: float | None = None,
        y: npt.NDArray[np.float64] | None = None,
        y_dot: npt.NDArray[np.float64] | None = None,
        inputs: dict | str | None = None,
    ):
        """See :meth:`pybamm.Symbol._base_evaluate()`."""
        if y_dot is None:
            raise TypeError("StateVectorDot cannot evaluate input 'y_dot=None'")
        if y_dot.shape[0] < self._last_point:
            raise ValueError(
                "y_dot is too short, so value with slice is smaller than expected"
            )
        return self._read(y_dot)

    def diff(self, variable: pybamm.Symbol):
        if variable == self:
            return pybamm.Scalar(1)
        elif variable == pybamm.t:
            raise pybamm.ModelError(
                "cannot take second time derivative of a state vector"
            )
        else:
            return pybamm.Scalar(0)

    def _to_casadi(self, t, y, y_dot, inputs, casadi_symbols):
        """See :meth:`pybamm.Symbol._to_casadi()`."""
        if y_dot is None:
            raise ValueError("Must provide a 'y_dot' for converting state vectors")
        return casadi.vertcat(*[y_dot[y_slice] for y_slice in self.y_slices])

    def _jac(self, variable: pybamm.StateVector | pybamm.StateVectorDot):
        if isinstance(variable, pybamm.StateVectorDot):
            return self._jac_same_vector(variable)
        elif isinstance(variable, pybamm.StateVector):
            return self._jac_diff_vector(variable)
