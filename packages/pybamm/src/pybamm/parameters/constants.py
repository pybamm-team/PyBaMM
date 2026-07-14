#
# Standard constants
#
from scipy import constants

from pybamm import Constant

R = Constant(constants.R, "R")
F = Constant(constants.physical_constants["Faraday constant"][0], "F")
k_b = Constant(constants.physical_constants["Boltzmann constant"][0], "k_b")
q_e = Constant(constants.physical_constants["electron volt"][0], "q_e")
