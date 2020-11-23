#
# Exception classes
#


class DomainError(Exception):
    """Domain error: an operation was attempted on nodes with un-matched domains"""

    pass


class OptionError(Exception):
    """Option error: an unknown option was given"""

    pass


class GeometryError(Exception):
    """
    Geometry error: Raised if the an unimplemented geometry is used.
    """

    pass


class ModelError(Exception):
    """Model error: the model is not well-posed (can be before or after processing)"""

    pass


class SolverError(Exception):
    """
    Solver error: a solution to the model could not be found with the chosen settings
    """

    pass


class SolverWarning(UserWarning):
    """
    Solver warning: the chosen solver settings may not give the desired output
    """

    pass


class ShapeError(Exception):
    """
    Shape error: cannot evaluate an object to find its shape
    """

    pass


class ModelWarning(UserWarning):
    """
    Model warning: the model does not contain all of the standard output variables
    """

    pass


class InputError(Exception):
    """
    An external variable has been input incorrectly into PyBaMM
    """

    pass


class DiscretisationError(Exception):
    """
    A model could not be discretised
    """

    pass
