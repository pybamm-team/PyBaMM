#
# Standard constants
#
from scipy import constants
from pybamm import Scalar

R = Scalar(constants.R, "J.mol-1.K-1")
F = Scalar(constants.physical_constants["Faraday constant"][0], "C.mol-1")
k_b = Scalar(constants.physical_constants["Boltzmann constant"][0], "J.K-1")
q_e = Scalar(constants.physical_constants["electron volt"][0], "J")
