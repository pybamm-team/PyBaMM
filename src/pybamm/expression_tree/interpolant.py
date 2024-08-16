#
# Interpolating class
#
from __future__ import annotations
import numpy as np
from scipy import interpolate
from collections.abc import Sequence
import numbers

import pybamm


class Interpolant(pybamm.Function):
    """
    Interpolate data in 1D, 2D, or 3D. Interpolation in 3D requires the input data to be
    on a regular grid (as per scipy.interpolate.RegularGridInterpolator).

    Parameters
    ----------
    x : iterable of :class:`numpy.ndarray`
        The data point coordinates. If 1-D, then this is an array(s) of real values. If,
        2D or 3D interpolation, then this is to ba a tuple of 1D arrays (one for each
        dimension) which together define the coordinates of the points.
    y : :class:`numpy.ndarray`
        The values of the function to interpolate at the data points. In 2D and 3D, this
        should be a matrix of two and three dimensions respectively.
    children : iterable of :class:`pybamm.Symbol`
        Node(s) to use when evaluating the interpolant. Each child corresponds to an
        entry of x
    name : str, optional
        Name of the interpolant. Default is None, in which case the name "interpolating
        function" is given.
    interpolator : str, optional
        Which interpolator to use. Can be "linear", "cubic", or "pchip". Default is
        "linear". For 3D interpolation, only "linear" an "cubic" are currently
        supported.
    extrapolate : bool, optional
        Whether to extrapolate for points that are outside of the parametrisation
        range, or return NaN (following default behaviour from scipy). Default is True.
        Generally, it is best to set this to be False for 3D interpolation due to
        the higher potential for errors in extrapolation.
    """

    def __init__(
        self,
        x: np.ndarray | Sequence[np.ndarray],
        y: np.ndarray,
        children: Sequence[pybamm.Symbol] | pybamm.Time,
        name: str | None = None,
        interpolator: str | None = "linear",
        extrapolate: bool = True,
        entries_string: str | None = None,
        _num_derivatives: int = 0,
    ):
        # Check interpolator is valid
        if interpolator not in ["linear", "cubic", "pchip"]:
            raise ValueError(f"interpolator '{interpolator}' not recognised")

        # Perform some checks on the data
        if isinstance(x, (tuple, list)) and len(x) == 2:
            x1, x2 = x
            if y.ndim != 2:
                raise ValueError("y should be two-dimensional if len(x)=2")
            if x1.shape[0] != y.shape[0]:
                raise ValueError(
                    "len(x1) should equal y=shape[1], "
                    f"but x1.shape={x1.shape} and y.shape={y.shape}"
                )
            if x2 is not None and x2.shape[0] != y.shape[1]:
                raise ValueError(
                    "len(x2) should equal y=shape[0], "
                    f"but x2.shape={x2.shape} and y.shape={y.shape}"
                )
        elif isinstance(x, (tuple, list)) and len(x) == 3:
            x1, x2, x3 = x
            if y.ndim != 3:
                raise ValueError("y should be three-dimensional if len(x)=3")

            if x1.shape[0] != y.shape[0]:
                raise ValueError(
                    "len(x1) should equal y=shape[0], "
                    f"but x1.shape={x1.shape} and y.shape={y.shape}"
                )
            if x2 is not None and x2.shape[0] != y.shape[1]:
                raise ValueError(
                    "len(x2) should equal y=shape[1], "
                    f"but x2.shape={x2.shape} and y.shape={y.shape}"
                )
            if x3 is not None and x3.shape[0] != y.shape[2]:
                raise ValueError(
                    "len(x3) should equal y=shape[2], "
                    f"but x3.shape={x3.shape} and y.shape={y.shape}"
                )
        else:
            if isinstance(x, (tuple, list)):
                x1 = x[0]
            else:
                x1 = x
                x: list[np.ndarray] = [x]  # type: ignore[no-redef]
            x2 = None
            if x1.shape[0] != y.shape[0]:
                raise ValueError(
                    "len(x1) should equal y=shape[0], "
                    f"but x1.shape={x1.shape} and y.shape={y.shape}"
                )
        # children should be a list not a symbol or a number
        if isinstance(children, (pybamm.Symbol, numbers.Number)):
            children = [children]
        # Either a single x is provided and there is one child
        # or x is a 2-tuple and there are two children
        if len(x) != len(children):
            raise ValueError("len(x) should equal len(children)")
        # if there is only one x, y can be 2-dimensional but the child must have
        # length 1
        if len(x) == 1 and y.ndim == 2 and children[0].size != 1:
            raise ValueError(
                "child should have size 1 if y is two-dimensional and len(x)==1"
            )

        # Create interpolating function
        if len(x) == 1:
            self.dimension = 1
            if interpolator == "linear":
                if extrapolate is False:
                    fill_value_1: float | str = np.nan
                elif extrapolate is True:
                    fill_value_1 = "extrapolate"
                interpolating_function = interpolate.interp1d(
                    x1,
                    y,
                    bounds_error=False,
                    fill_value=fill_value_1,
                    axis=0,
                )
            elif interpolator == "cubic":
                interpolating_function = interpolate.CubicSpline(
                    x1, y, extrapolate=extrapolate
                )
            elif interpolator == "pchip":
                interpolating_function = interpolate.PchipInterpolator(
                    x1, y, extrapolate=extrapolate
                )
        elif len(x) == 2:
            self.dimension = 2
            if interpolator == "pchip":
                raise ValueError(
                    "interpolator should be 'linear' or 'cubic' if x is two-dimensional"
                )
            else:
                if extrapolate:
                    fill_value = None
                else:
                    fill_value = np.nan
                interpolating_function = interpolate.RegularGridInterpolator(
                    (x1, x2),
                    y,
                    method=interpolator,
                    bounds_error=False,
                    fill_value=fill_value,
                )

        elif len(x) == 3:
            self.dimension = 3

            if extrapolate:
                fill_value = None
            else:
                fill_value = np.nan

            possible_interpolators = ["linear", "cubic"]
            if interpolator not in possible_interpolators:
                raise ValueError(
                    """interpolator should be 'linear' or 'cubic'
                    for 3D interpolation"""
                )
            else:
                interpolating_function = interpolate.RegularGridInterpolator(
                    (x1, x2, x3),
                    y,
                    method=interpolator,
                    bounds_error=False,
                    fill_value=fill_value,
                )
        else:
            raise ValueError(f"Invalid dimension of x: {len(x)}")

        # Set name
        if name is None:
            name = "interpolating_function"
        self.x = x
        self.y = y
        self.entries_string = entries_string

        # Differentiate the interpolating function if necessary
        self._num_derivatives = _num_derivatives
        for _ in range(_num_derivatives):
            interpolating_function = interpolating_function.derivative()

        super().__init__(interpolating_function, *children, name=name)

        # Store information as attributes
        self.interpolator = interpolator
        self.extrapolate = extrapolate

    @classmethod
    def _from_json(cls, snippet: dict):
        """Create an Interpolant object from JSON data"""

        x1 = []

        if len(snippet["x"]) == 1:
            x1 = [np.array(x) for x in snippet["x"]]

        return cls(
            x1 if x1 else tuple(np.array(x) for x in snippet["x"]),
            np.array(snippet["y"]),
            snippet["children"],
            name=snippet["name"],
            interpolator=snippet["interpolator"],
            extrapolate=snippet["extrapolate"],
            _num_derivatives=snippet["_num_derivatives"],
        )

    @property
    def entries_string(self):
        return self._entries_string

    @entries_string.setter
    def entries_string(self, value):
        # We must include the entries in the hash, since different arrays can be
        # indistinguishable by class, name and domain alone
        # Slightly different syntax for sparse and non-sparse matrices
        if value is not None:
            self._entries_string = value
        else:
            self._entries_string = ""
            for i, x in enumerate(self.x):
                self._entries_string += "x" + str(i) + "_" + str(x.tobytes())
            self._entries_string += "y_" + str(self.y.tobytes())

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`."""
        self._id = hash(
            (
                self.__class__,
                self.name,
                self.entries_string,
                *tuple([child.id for child in self.children]),
                *tuple(self.domain),
                self._num_derivatives,
            )
        )

    def create_copy(self, new_children=None, perform_simplifications=True):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        children = self._children_for_copying(new_children)

        return pybamm.Interpolant(
            self.x,
            self.y,
            children,
            name=self.name,
            interpolator=self.interpolator,
            extrapolate=self.extrapolate,
            entries_string=self.entries_string,
            _num_derivatives=self._num_derivatives,
        )

    def _function_evaluate(self, evaluated_children):
        children_eval_flat = []
        for child in evaluated_children:
            if isinstance(child, np.ndarray):
                children_eval_flat.append(child.flatten())
            else:
                children_eval_flat.append(child)
        if self.dimension == 1:
            return self.function(*children_eval_flat).flatten()[:, np.newaxis]
        elif self.dimension in [2, 3]:
            # If the children are scalars, we need to add a dimension
            shapes = []
            for child in evaluated_children:
                if isinstance(child, (float, int)):
                    shapes.append(())
                else:
                    shapes.append(child.shape)
            shapes = set(shapes)
            shapes.discard(())

            if len(shapes) > 1:
                raise ValueError(
                    "All children must have the same shape for 3D interpolation"
                )

            if len(shapes) == 0:
                shape = (1,)
            else:
                shape = shapes.pop()
            new_evaluated_children = []
            for child in evaluated_children:
                if hasattr(child, "shape") and child.shape == shape:
                    new_evaluated_children.append(child.flatten())
                else:
                    new_evaluated_children.append(np.reshape(child, shape).flatten())

            # return nans if there are any within the children
            nans = np.isnan(new_evaluated_children)
            if np.any(nans):
                nan_children = []
                for child, interp_range in zip(
                    new_evaluated_children, self.function.grid
                ):
                    nan_children.append(np.ones_like(child) * interp_range.mean())
                nan_eval = self.function(np.transpose(nan_children))
                return np.reshape(nan_eval, shape)
            else:
                res = self.function(np.transpose(new_evaluated_children))
                return np.reshape(res, shape)

        else:  # pragma: no cover
            raise ValueError(f"Invalid dimension: {self.dimension}")

    def _function_diff(self, children: Sequence[pybamm.Symbol], idx: float):
        """
        Derivative with respect to child number 'idx'.
        See :meth:`pybamm.Symbol._diff()`.
        """
        if len(children) > 1:
            raise NotImplementedError(
                "differentiation not implemented for functions with more than one child"
            )
        else:
            # keep using "derivative" as derivative
            return Interpolant(
                self.x,
                self.y,
                children,
                name=self.name,
                interpolator=self.interpolator,
                extrapolate=self.extrapolate,
                _num_derivatives=self._num_derivatives + 1,
            )

    def to_json(self):
        """
        Method to serialise an Interpolant object into JSON.
        """

        json_dict = {
            "name": self.name,
            "id": self.id,
            "x": [x_item.tolist() for x_item in self.x],
            "y": self.y.tolist(),
            "interpolator": self.interpolator,
            "extrapolate": self.extrapolate,
            "_num_derivatives": self._num_derivatives,
        }

        return json_dict
