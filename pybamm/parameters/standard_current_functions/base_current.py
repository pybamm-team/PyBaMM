#
# Base class for setting/getting current
#


class GetCurrent(object):
    def __init__(self, **kwargs):
        # The parameters dictionary holds the symbols. During processing of
        # parameters these are evaluated and stored in parameters_eval. The
        # separate dictionary is required so that the symbol may be processed again
        # if the parameters are updated
        self.parameters = kwargs
        self.parameters_eval = kwargs

    def __call__(self, t):
        return 1
