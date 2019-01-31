#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


def homogeneous_reaction():
    current_neg = (
        pybamm.Scalar(1, domain=["negative electrode"]) / pybamm.standard_parameters.ln
    )
    current_sep = pybamm.Scalar(0, domain=["separator"])
    current_pos = (
        -pybamm.Scalar(1, domain=["positive electrode"]) / pybamm.standard_parameters.lp
    )
    return pybamm.Concatenation(current_neg, current_sep, current_pos)
