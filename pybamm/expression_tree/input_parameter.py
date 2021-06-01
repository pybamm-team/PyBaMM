#
# Parameter classes
#
import numbers
import numpy as np
import pybamm


class InputParameter(pybamm.Symbol):
    """A node in the expression tree representing an input parameter

    This node's value can be set at the point of solving, allowing parameter estimation
    and control

    Parameters
    ----------
    name : str
        name of the node
    domain : iterable of str, or str
        list of domains over which the node is valid (empty list indicates the symbol
        is valid over all domains)
    """

    def __init__(self, name, domain=None):
        # Expected shape defaults to 1
        self._expected_size = 1
        super().__init__(name, domain=domain)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        new_input_parameter = InputParameter(self.name, self.domain)
        new_input_parameter._expected_size = self._expected_size
        return new_input_parameter

    def set_expected_size(self, size):
        """Specify the size that the input parameter should be."""
        self._expected_size = size

        # We also need to update the saved size and shape
        self._saved_size = size
        self._saved_shape = (size, 1)
        self._saved_evaluate_for_shape = self._evaluate_for_shape()

    def _evaluate_for_shape(self):
        """
        Returns the scalar 'NaN' to represent the shape of a parameter.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        if self._expected_size == 1:
            return np.nan
        else:
            return np.nan * np.ones((self._expected_size, 1))

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
        return pybamm.Scalar(0)

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        # inputs should be a dictionary
        # convert 'None' to empty dictionary for more informative error
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, dict):
            # if the special input "shape test" is passed, just return NaN
            if inputs == "shape test":
                return self.evaluate_for_shape()
            raise TypeError("inputs should be a dictionary")
        try:
            input_eval = inputs[self.name]
        # raise more informative error if can't find name in dict
        except KeyError:
            raise KeyError("Input parameter '{}' not found".format(self.name))

        if isinstance(input_eval, numbers.Number):
            input_size = 1
            input_ndim = 0
        else:
            input_size = input_eval.shape[0]
            input_ndim = len(input_eval.shape)
        if input_size == self._expected_size:
            if input_ndim == 1:
                return input_eval[:, np.newaxis]
            else:
                return input_eval
        else:
            raise ValueError(
                "Input parameter '{}' was given an object of size '{}'".format(
                    self.name, input_size
                )
                + " but was expecting an object of size '{}'.".format(
                    self._expected_size
                )
            )
