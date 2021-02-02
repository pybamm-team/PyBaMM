import pybamm

whole_cell = ["separator", "working electrode"]

# Domains at cell centres
x_Li = pybamm.SpatialVariable(
    "x_Li",
    domain=["lithium counter electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_s = pybamm.SpatialVariable(
    "x_s",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_w = pybamm.SpatialVariable(
    "x_w",
    domain=["working electrode"],
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

r_w = pybamm.SpatialVariable(
    "r_w",
    domain=["working particle"],
    auxiliary_domains={
        "secondary": "working electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)

# Domains at cell edges
x_Li_edge = pybamm.SpatialVariableEdge(
    "x_Li",
    domain=["lithium counter electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_s_edge = pybamm.SpatialVariableEdge(
    "x_s",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_w_edge = pybamm.SpatialVariableEdge(
    "x_w",
    domain=["working electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_edge = pybamm.SpatialVariableEdge(
    "x",
    domain=whole_cell,
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)

y_edge = pybamm.SpatialVariableEdge(
    "y", domain="current collector", coord_sys="cartesian"
)
z_edge = pybamm.SpatialVariableEdge(
    "z", domain="current collector", coord_sys="cartesian"
)

r_w_edge = pybamm.SpatialVariableEdge(
    "r_w",
    domain=["working particle"],
    auxiliary_domains={
        "secondary": "working electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
