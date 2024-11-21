import pybamm
import numbers


class Domain:
    def __init__(self):
        self.tabs = None

    def process_parameters(self, parameter_values):
        processed_bounds = tuple(
            self.process_and_check(x, parameter_values) for x in self.bounds
        )
        return Domain1D(processed_bounds, self.coord_sys)

    @staticmethod
    def process_and_check(bound, parameter_values):
        new_bound = parameter_values.process_symbol(bound)
        if not isinstance(new_bound, pybamm.Scalar):
            raise ValueError(
                "Geometry parameters must be Scalars after parameter processing"
            )
        return new_bound

    def evaluate(self):
        bounds_eval = []
        for bound in self.bounds:
            if isinstance(bound, pybamm.Symbol):
                try:
                    bound_eval = bound.evaluate()
                except NotImplementedError as error:
                    if bound.has_symbol_of_classes(pybamm.Parameter):
                        raise pybamm.DiscretisationError(
                            "Parameter values have not yet been set for "
                            "geometry. Make sure that something like "
                            "`param.process_geometry(geometry)` has been "
                            "run."
                        ) from error
                    else:
                        raise error
            elif isinstance(bound, numbers.Number):
                bound_eval = bound
            else:
                raise ValueError(f"Unknown type for bounds: {type(bound)}")
            bounds_eval.append(bound_eval)

        return Domain1D(bounds_eval, self.coord_sys)


class Domain1D(Domain):
    def __init__(self, bounds: tuple[float, float], coord_sys: str = "cartesian"):
        self.bounds = bounds
        self.coord_sys = coord_sys
        super().__init__()

class Domain2D(Domain):