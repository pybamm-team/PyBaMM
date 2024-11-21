import pybamm
from enum import Enum, auto
from typing import Union
import numbers


class CoordinateSystem(Enum):
    """
    Enumeration of supported coordinate systems. Can be CARTESIAN, CYLINDRICAL_POLAR,
    or SPHERICAL_POLAR.

    Examples
    --------
    >>> domain = Domain({"x": (0, 1)}, coord_sys=CoordinateSystem.CARTESIAN)
    >>> domain.coord_sys == CoordinateSystem.CARTESIAN
    True
    """

    CARTESIAN = auto()
    CYLINDRICAL_POLAR = auto()
    SPHERICAL_POLAR = auto()

    @classmethod
    def from_string(cls, name: str) -> "CoordinateSystem":
        """
        Create coordinate system from string name.

        Parameters
        ----------
        name : str
            Name of coordinate system (case insensitive)

        Returns
        -------
        CoordinateSystem

        Raises
        ------
        ValueError
            If name is not recognized
        """
        try:
            return {
                "cartesian": cls.CARTESIAN,
                "cylindrical polar": cls.CYLINDRICAL_POLAR,
                "spherical polar": cls.SPHERICAL_POLAR,
            }[name.lower()]
        except KeyError:
            valid = ["cartesian", "cylindrical polar", "spherical polar"]
            raise ValueError(
                f"Unknown coordinate system '{name}'. Must be one of: {valid}"
            ) from None


class Domain:
    """
    A base class for geometric domains including 0D points.

    Parameters
    ----------
    dimensions : dict[str, tuple[float, float] | float]
        Dictionary mapping dimension names to either:
        - A tuple of (min, max) for regular dimensions
        - A single float for 0D points
    properties : dict, optional
        Additional geometric properties (e.g., tabs)
    coord_sys : CoordinateSystem | str, optional
        Coordinate system type. Can be either a CoordinateSystem enum value
        or a string that will be converted to one. Defaults to CARTESIAN.

    """

    def __init__(
        self,
        dimensions: dict[str, Union[tuple[float, float], float]],
        properties: dict | None = None,
        coord_sys: Union[CoordinateSystem, str] = CoordinateSystem.CARTESIAN,
    ):
        self.dimensions = self._normalize_dimensions(dimensions)
        self.properties = properties or {}

        # Convert string to enum if needed
        if isinstance(coord_sys, str):
            coord_sys = CoordinateSystem.from_string(coord_sys)
        self.coord_sys = coord_sys

        self._validate_dimensions()

    @property
    def dimension_bounds(self):
        """Get bounds for all dimensions"""
        return tuple(self.dimensions.values())

    @property
    def dimension_names(self):
        """Get names of all dimensions"""
        return tuple(self.dimensions.keys())

    def _normalize_dimensions(self, dimensions):
        """Convert single values to (value, value) tuples for 0D points"""
        normalized = {}
        for name, bounds in dimensions.items():
            if isinstance(bounds, (int, float, pybamm.Symbol)):
                # 0D case: convert single value to (value, value)
                normalized[name] = (bounds, bounds)
            else:
                normalized[name] = bounds
        return normalized

    def _validate_dimensions(self):
        """Validate dimension consistency"""
        if not self.dimensions:
            raise ValueError("Domain must have at least one dimension")

        # Validate bounds
        for name, bounds in self.dimensions.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(
                    f"Bounds for dimension '{name}' must be a tuple of (min, max)"
                )

    @property
    def is_0D(self):
        """Check if any dimensions are 0D (point values)"""
        return any(bounds[0] == bounds[1] for bounds in self.dimensions.values())

    @property
    def zero_dimensional_dims(self):
        """Get list of dimensions that are 0D"""
        return [
            name for name, bounds in self.dimensions.items() if bounds[0] == bounds[1]
        ]

    def process_parameters(self, parameter_values):
        """Process all parameters in dimensions and properties"""
        processed_dimensions = {}
        for name, bounds in self.dimensions.items():
            # Handle both regular and 0D cases
            processed_bounds = tuple(
                self.process_and_check(x, parameter_values) for x in bounds
            )
            processed_dimensions[name] = processed_bounds

        processed_properties = self._process_nested_dict(
            self.properties, parameter_values
        )

        return Domain(processed_dimensions, processed_properties)

    def evaluate(self):
        """Evaluate all symbolic expressions in dimensions and properties"""
        evaluated_dimensions = {}
        for name, bounds in self.dimensions.items():
            evaluated_bounds = tuple(self._evaluate_bound(x) for x in bounds)
            # For 0D cases, store as single value if bounds are equal
            if evaluated_bounds[0] == evaluated_bounds[1]:
                evaluated_dimensions[name] = evaluated_bounds[0]
            else:
                evaluated_dimensions[name] = evaluated_bounds

        evaluated_properties = self._evaluate_nested_dict(self.properties)

        return Domain(evaluated_dimensions, evaluated_properties)

    @staticmethod
    def process_and_check(bound, parameter_values):
        new_bound = parameter_values.process_symbol(bound)
        if not isinstance(new_bound, pybamm.Scalar):
            raise ValueError(
                "Geometry parameters must be Scalars after parameter processing"
            )
        return new_bound

    def _process_nested_dict(self, d, parameter_values):
        """Recursively process parameters in nested dictionaries"""
        if not isinstance(d, dict):
            return self.process_and_check(d, parameter_values)
        return {k: self._process_nested_dict(v, parameter_values) for k, v in d.items()}

    def _evaluate_nested_dict(self, d):
        """Recursively evaluate expressions in nested dictionaries"""
        if not isinstance(d, dict):
            return self._evaluate_bound(d)
        return {k: self._evaluate_nested_dict(v) for k, v in d.items()}

    def _evaluate_bound(self, bound):
        """Evaluate a single bound or property value"""
        if isinstance(bound, pybamm.Symbol):
            try:
                return bound.evaluate()
            except NotImplementedError as error:
                if bound.has_symbol_of_classes(pybamm.Parameter):
                    raise pybamm.DiscretisationError(
                        "Parameter values have not yet been set for "
                        "geometry. Make sure that something like "
                        "`param.process_geometry(geometry)` has been "
                        "run."
                    ) from error
                raise error
        elif isinstance(bound, numbers.Number):
            return bound
        else:
            raise ValueError(f"Unknown type for bounds: {type(bound)}")
