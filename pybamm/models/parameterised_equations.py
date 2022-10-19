#
# Parameterized equations class
#
import numbers
import warnings
from collections import OrderedDict

import copy
import casadi
import numpy as np

import pybamm
from pybamm.expression_tree.operations.latexify import Latexify


class _ParameterisedEquations(pybamm._BaseProcessedEquations):
    """
    Class containing equations with parameters set to values.

    **Extends:** :class:`pybamm.BaseProcessedEquations`
    """

    def __init__(
        self,
        parameter_values,
        *args,
    ):
        # Save parameter values used to create this model
        self._parameter_values = parameter_values

        super().__init__(*args)

    def variables_update_function(self, key):
        return self._parameter_values.process_symbol(self._unprocessed_variables[key])
    