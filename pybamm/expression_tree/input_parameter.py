#
# Parameter classes
#
import numbers
import numpy as np
import scipy.sparse
import pybamm


class InputParameter(pybamm.Symbol):
    """
    A node in the expression tree representing an input parameter.

    This node's value can be set at the point of solving, allowing parameter estimation
    and control

    Parameters
    ----------
    name : str
        name of the node
    domain : iterable of str, or str
        list of domains over which the node is valid (empty list indicates the symbol
        is valid over all domains)
    expected_size : int
        The size of the input parameter expected, defaults to 1 (scalar input)
    """

    def __init__(self, name, domain=None, expected_size=None):
        # Expected size defaults to 1 if no domain else None (gets set later)
        if expected_size is None:
            if domain is None:
                expected_size = 1
            else:
                expected_size = None
        self._expected_size = expected_size
        super().__init__(name, domain=domain)

    @classmethod
    def _from_json(cls, snippet: dict):
        instance = cls.__new__(cls)

        instance.__init__(
            snippet["name"],
            domain=snippet["domain"],
            expected_size=snippet["expected_size"],
        )

        return instance

    def create_copy(self):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        new_input_parameter = InputParameter(
            self.name, self.domain, expected_size=self._expected_size
        )
        return new_input_parameter

    def _evaluate_for_shape(self):
        """
        Returns the scalar 'NaN' to represent the shape of a parameter.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        if self._expected_size is None:
            return pybamm.evaluate_for_shape_using_domain(self.domains)
        elif self._expected_size == 1:
            return np.nan
        else:
            return np.nan * np.ones((self._expected_size, 1))

    def _jac(self, variable):
        """See :meth:`pybamm.Symbol._jac()`."""
        n_variable = variable.evaluation_array.count(True)
        nan_vector = self._evaluate_for_shape()
        if isinstance(nan_vector, numbers.Number):
            n_self = 1
        else:
            n_self = nan_vector.shape[0]
        zero_matrix = scipy.sparse.csr_matrix((n_self, n_variable))
        return pybamm.Matrix(zero_matrix)

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
            raise KeyError(f"Input parameter '{self.name}' not found")

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
                + f" but was expecting an object of size '{self._expected_size}'."
            )

    def to_json(self):
        """
        Method to serialise an InputParameter object into JSON.
        """

        json_dict = {
            "name": self.name,
            "id": self.id,
            "domain": self.domain,
            "expected_size": self._expected_size,
        }

        return json_dict
