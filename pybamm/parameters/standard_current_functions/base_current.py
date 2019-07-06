#
# Base class for setting/getting current
#


class GetCurrent(object):
    """
    The base class for setting the input current for a simulation. The parameters
    dictionary holds the symbols of any paramters required to evaluate the current.
    During processing, the evaluated parameters are stored in parameters_eval.
    """

    def __init__(self):
        self.parameters = {}
        self.parameters_eval = {}

    def __str__(self):
        return "Base current"

    def __call__(self, t):
        return 1
