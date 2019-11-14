#
# Base model for the external circuit
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Model to represent the behaviour of the external circuit. """

    def __init__(self, param):
        super().__init__(param)
