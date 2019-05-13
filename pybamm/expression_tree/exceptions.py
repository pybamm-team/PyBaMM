#
# Exception classes
#


class DomainError(Exception):
    """Domain error: an operation was attempted on nodes with un-matched domains"""

    pass


class ModelError(Exception):
    """Model error: the model is not well-posed (can be before or after processing)"""

    pass


class SolverError(Exception):
    """
    Solver error: a solution to the model could not be found with the chosen settings
    """

    pass
