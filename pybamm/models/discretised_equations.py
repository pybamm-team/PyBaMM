#
# Base model class
#
import numbers
import warnings
from collections import OrderedDict

import copy
import casadi
import numpy as np

import pybamm
from pybamm.expression_tree.operations.latexify import Latexify


class _DiscretisedEquations(pybamm._BaseProcessedEquations):
    """
    Class containing discretised equations.

    **Extends:** :class:`pybamm._BaseProcessedEquations`
    """

    def __init__(self, discretisation, *args, y_slices, bounds):
        # Save discretisation used to create this model
        self._discretisation = discretisation

        super().__init__(*args)
        self.y_slices = y_slices
        self.bounds = bounds

        self.set_concatenated_attributes()

    def set_concatenated_attributes(self):
        disc = self._discretisation
        self._concatenated_rhs = disc._concatenate_in_order(self.rhs)
        self._concatenated_algebraic = disc._concatenate_in_order(self.algebraic)
        self._concatenated_initial_conditions = disc._concatenate_in_order(
            self.initial_conditions, check_complete=True
        )

        self.len_rhs = self._concatenated_rhs.size
        self.len_algebraic = self._concatenated_algebraic.size
        self.len_rhs_and_alg = self.len_rhs + self.len_algebraic

    def variables_update_function(self, key):
        return self._discretisation.process_symbol(self._unprocessed_variables[key])
