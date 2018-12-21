#
# Standard parameters for battery models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from pybamm import Parameter

# Lengths
Ln = Parameter("Ln")
Ls = Parameter("Ls")
Lp = Parameter("Lp")
L = Ln + Ls + Lp

# Dimensionless lengths
ln = Ln / L
ls = Ls / L
lp = Lp / L
