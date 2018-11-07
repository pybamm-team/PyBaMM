class Variables:
    """
    Extracts and stores the model variables.

    Parameters
    ----------
    y : array_like
        An array containing variable values to be extracted.
    """
    def __init__(self, y, param):
        self.c = y
        # TODO: make model.variables an input of this class
