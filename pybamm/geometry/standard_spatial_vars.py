import pybamm

whole_cell = ["negative electrode", "separator", "positive electrode"]

# Domains at cell centres
x_n = pybamm.SpatialVariable(
    "x_n",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_s = pybamm.SpatialVariable(
    "x_s",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_p = pybamm.SpatialVariable(
    "x_p",
    domain=["positive electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x = pybamm.SpatialVariable(
    "x",
    domain=whole_cell,
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)

y = pybamm.SpatialVariable("y", domain="current collector", coord_sys="cartesian")
z = pybamm.SpatialVariable("z", domain="current collector", coord_sys="cartesian")

r_n = pybamm.SpatialVariable(
    "r_n",
    domain=["negative particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p = pybamm.SpatialVariable(
    "r_p",
    domain=["positive particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)

# Domains at cell edges
x_n_edge = pybamm.SpatialVariable(
    "x_n_edge",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_s_edge = pybamm.SpatialVariable(
    "x_s_edge",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_p_edge = pybamm.SpatialVariable(
    "x_p_edge",
    domain=["positive electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_edge = pybamm.SpatialVariable(
    "x_edge",
    domain=whole_cell,
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)

y_edge = pybamm.SpatialVariable(
    "y_edge", domain="current collector", coord_sys="cartesian"
)
z_edge = pybamm.SpatialVariable(
    "z_edge", domain="current collector", coord_sys="cartesian"
)

r_n_edge = pybamm.SpatialVariable(
    "r_n_edge",
    domain=["negative particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p_edge = pybamm.SpatialVariable(
    "r_p_edge",
    domain=["positive particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
