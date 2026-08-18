#
# Exception classes
#


class DomainError(Exception):
    """Domain error: an operation was attempted on nodes with un-matched domains."""


class OptionError(Exception):
    """Option error: an unknown option was given."""


class OptionWarning(UserWarning):
    """Option warning: the chosen options may not give the desired output."""


class GeometryError(Exception):
    """Geometry error: Raised if the an unimplemented geometry is used."""


class ModelError(Exception):
    """Model error: the model is not well-posed (can be before or after processing)"""


class SolverError(Exception):
    """
    Solver error: a solution to the model could not be found with the chosen settings
    """

    def __init__(self, *args):
        self.message = args[0]


class SolverWarning(UserWarning):
    """Solver warning: the chosen solver settings may not give the desired output."""


class ShapeError(Exception):
    """Shape error: cannot evaluate an object to find its shape."""


class ModelWarning(UserWarning):
    """
    Model warning: the model does not contain all of the standard output variables
    """


class DiscretisationError(Exception):
    """A model could not be discretised."""


class InvalidModelJSONError(Exception):
    """Raised when a model JSON file is invalid or cannot be parsed."""
