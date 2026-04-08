#
# Interpolating class
#
from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any

import casadi
import numpy as np
import numpy.typing as npt
from scipy import interpolate

import pybamm


def _is_uniform_grid(x: npt.NDArray[np.float64]) -> bool:
    # Try seeing if the grid was computed using np.linspace
    return bool(np.array_equal(x, np.linspace(x[0], x[-1], len(x))))


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
        x: npt.NDArray[np.float64] | Sequence[npt.NDArray[np.float64]],
        y: npt.NDArray[Any],
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

        # Normalise x to a list of arrays
        if not isinstance(x, tuple | list):
            x = [x]

        # Perform some checks on the data
        ndim = len(x)
        if ndim == 1:
            x1 = x[0]
            if x1.shape[0] != y.shape[0]:
                raise ValueError(
                    "len(x1) should equal y=shape[0], "
                    f"but x1.shape={x1.shape} and y.shape={y.shape}"
                )
            if np.any(x1[:-1] > x1[1:]):
                raise ValueError("x should be monotonically increasing")
        else:
            if y.ndim != ndim:
                raise ValueError(
                    f"y should be {ndim}-dimensional if len(x)={ndim}, "
                    f"but y.ndim={y.ndim}"
                )
            for i, xi in enumerate(x):
                if xi.shape[0] != y.shape[i]:
                    raise ValueError(
                        f"len(x{i + 1}) should equal y.shape[{i}], "
                        f"but x{i + 1}.shape={xi.shape} and y.shape={y.shape}"
                    )
            if interpolator == "pchip":
                raise ValueError(
                    "interpolator should be 'linear' or 'cubic' "
                    f"if x is {ndim}-dimensional"
                )

        # children should be a list not a symbol or a number
        if isinstance(children, pybamm.Symbol | numbers.Number):
            children = [children]
        if len(x) != len(children):
            raise ValueError("len(x) should equal len(children)")
        if ndim == 1 and y.ndim == 2 and children[0].size != 1:
            raise ValueError(
                "child should have size 1 if y is two-dimensional and len(x)==1"
            )

        # Create interpolating function
        self.dimension = ndim
        if ndim == 1:
            x1 = x[0]
            if interpolator == "linear":
                fill_value: float | str = "extrapolate" if extrapolate else np.nan
                interpolating_function = interpolate.interp1d(
                    x1,
                    y,
                    bounds_error=False,
                    fill_value=fill_value,
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
        else:
            fill_value = None if extrapolate else np.nan
            interpolating_function = interpolate.RegularGridInterpolator(
                tuple(x),
                y,
                method=interpolator,
                bounds_error=False,
                fill_value=fill_value,
            )

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
        else:
            shapes = set()
            for child in evaluated_children:
                if isinstance(child, float | int):
                    continue
                shapes.add(child.shape)

            if len(shapes) > 1:
                raise ValueError(
                    "All children must have the same shape for N-D interpolation"
                )

            shape = shapes.pop() if shapes else (1,)
            new_evaluated_children = []
            for child in evaluated_children:
                if hasattr(child, "shape") and child.shape == shape:
                    new_evaluated_children.append(child.flatten())
                else:
                    new_evaluated_children.append(np.reshape(child, shape).flatten())

            nans = np.isnan(new_evaluated_children)
            if np.any(nans):
                nan_children = []
                for child, interp_range in zip(
                    new_evaluated_children, self.function.grid, strict=True
                ):
                    nan_children.append(np.ones_like(child) * interp_range.mean())
                nan_eval = self.function(np.transpose(nan_children))
                return np.reshape(nan_eval, shape)
            else:
                res = self.function(np.transpose(new_evaluated_children))
                return np.reshape(res, shape)

    def _to_casadi(self, t, y, y_dot, inputs, casadi_symbols):
        """See :meth:`pybamm.Symbol._to_casadi()`."""
        converted_children = super()._children_to_casadi(
            t, y, y_dot, inputs, casadi_symbols
        )

        if self.interpolator == "linear":
            return self._linear_to_casadi(converted_children)
        elif self.interpolator == "pchip":
            return self._pchip_to_casadi(converted_children)
        elif self.interpolator == "cubic":
            return self._cubic_to_casadi(converted_children)
        else:
            raise NotImplementedError(  # pragma: no cover
                f"Unknown interpolator: {self.interpolator}"
            )

    def _linear_to_casadi(self, converted_children):
        if self.dimension == 1:
            v = self.y.flatten()
        else:
            v = self.y.ravel(order="F")
        lookup_modes = ["exact" if _is_uniform_grid(xi) else "binary" for xi in self.x]
        result = casadi.MX.interpn_linear(
            self.x, v, converted_children, {"lookup_mode": lookup_modes}
        )
        if result.shape[0] == 1 and result.shape[1] > 1:
            result = result.T
        return result

    def _cubic_to_casadi(self, converted_children):
        if self.dimension == 1:
            bspline = interpolate.make_interp_spline(self.x[0], self.y, k=3)
            c_flat = bspline.c.flatten()
            n_basis = len(bspline.t) - bspline.k - 1
            m = c_flat.size // n_basis
            f = casadi.Function.bspline(
                self.name, [bspline.t], c_flat.tolist(), [bspline.k], m
            )
            return f(converted_children[0])
        elif self.dimension == 2:
            bspline = interpolate.RectBivariateSpline(self.x[0], self.x[1], self.y)
            [tx, ty, c] = bspline.tck
            [kx, ky] = bspline.degrees
            f = casadi.Function.bspline(self.name, [tx, ty], c.tolist(), [kx, ky], 1)
            return f(casadi.hcat(converted_children).T).T
        else:
            LUT = casadi.interpolant("LUT", "bspline", self.x, self.y.ravel(order="F"))
            return LUT(casadi.hcat(converted_children).T).T

    def _pchip_to_casadi(self, converted_children):
        x_np = np.asarray(self.x[0], dtype=np.float64)
        y_np = np.asarray(self.y, dtype=np.float64)
        d_np = interpolate.PchipInterpolator(x_np, y_np).derivative()(x_np)

        n = len(x_np) - 1
        h = np.diff(x_np)
        uniform = _is_uniform_grid(x_np)
        is_vector_valued = y_np.ndim > 1
        y_2d = y_np if is_vector_valued else y_np[:, np.newaxis]
        d_2d = d_np if d_np.ndim > 1 else d_np[:, np.newaxis]
        m = y_2d.shape[1]

        # Precompute Hermite cubic coefficients per interval, pre-divided by h^k.
        # p(dx) = c0 + c1*dx + c2*dx² + c3*dx³,  dx = x - x_lo
        # For non-uniform grids x_lo is stored as a 5th table entry;
        # for uniform grids x_lo is computed from the index via scalar arithmetic.
        stride = 4
        if not uniform:
            stride += 1
        coeffs = np.empty((n, stride, m))

        y0 = y_2d[:-1]  # (n, m)
        y1 = y_2d[1:]
        inv_h = (1.0 / h)[:, np.newaxis]  # (n, 1) for broadcasting
        hd0 = h[:, np.newaxis] * d_2d[:-1]
        hd1 = h[:, np.newaxis] * d_2d[1:]

        coeffs[:, 0] = y0
        coeffs[:, 1] = hd0 * inv_h
        coeffs[:, 2] = (-3 * y0 - 2 * hd0 + 3 * y1 - hd1) * inv_h**2
        coeffs[:, 3] = (2 * y0 + hd0 - 2 * y1 + hd1) * inv_h**3
        if not uniform:
            coeffs[:, 4, :] = x_np[:-1, np.newaxis]

        coeffs_flat = casadi.MX(coeffs.reshape(n * stride, m))

        # 1. Single interval index lookup (clamp handles extrapolation)
        x = converted_children[0]
        lookup_mode = "exact" if uniform else "binary"
        idx = casadi.low(casadi.MX(x_np), x, {"lookup_mode": lookup_mode})
        idx = casadi.fmin(casadi.fmax(idx, 0), n - 1)
        base = idx * stride

        # 2. Compute dx = x - x_lo
        if uniform:
            dx = x - (x_np[0] + idx * h[0])
        else:
            dx = x - coeffs_flat[base + 4, 0]

        # 3. Coefficient lookup and Horner evaluation
        c0 = coeffs_flat[base, :]
        c1 = coeffs_flat[base + 1, :]
        c2 = coeffs_flat[base + 2, :]
        c3 = coeffs_flat[base + 3, :]
        result = c0 + (c1 + (c2 + c3 * dx) * dx) * dx

        if is_vector_valued:
            result = result.T
        return result

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
