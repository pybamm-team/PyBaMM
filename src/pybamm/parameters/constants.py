#
# Standard constants
#
from scipy import constants
from pybamm import Scalar

R = Scalar(constants.R)
F = Scalar(constants.physical_constants["Faraday constant"][0])
k_b = Scalar(constants.physical_constants["Boltzmann constant"][0])
q_e = Scalar(constants.physical_constants["electron volt"][0])
