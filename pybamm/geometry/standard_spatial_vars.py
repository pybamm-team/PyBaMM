import pybamm

KNOWN_COORD_SYS = ["cartesian", "spherical polar"]

whole_cell = ["negative electrode", "separator", "positive electrode"]

x_n = pybamm.SpatialVariable(
    "x_n", domain=["negative electrode"], coord_sys="cartesian"
)
x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
x_p = pybamm.SpatialVariable(
    "x_p", domain=["positive electrode"], coord_sys="cartesian"
)

y = pybamm.SpatialVariable("y", domain=whole_cell, coord_sys="cartesian")
z = pybamm.SpatialVariable("z", domain=whole_cell, coord_sys="cartesian")

r_n = pybamm.SpatialVariable(
    "r_n", domain=["negative particle"], coord_sys="spherical polar"
)
r_p = pybamm.SpatialVariable(
    "r_p", domain=["positive particle"], coord_sys="spherical polar"
)
