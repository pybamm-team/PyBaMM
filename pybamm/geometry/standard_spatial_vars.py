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
r_macro = pybamm.SpatialVariable(
    "r_macro", domain="current collector", coord_sys="cylindrical polar"
)

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
r_n_prim = pybamm.SpatialVariable(
    "r_n_prim",
    domain=["negative primary particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p_prim = pybamm.SpatialVariable(
    "r_p_prim",
    domain=["positive primary particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_n_sec = pybamm.SpatialVariable(
    "r_n_sec",
    domain=["negative secondary particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p_sec = pybamm.SpatialVariable(
    "r_p_sec",
    domain=["positive secondary particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)

R_n = pybamm.SpatialVariable(
    "R_n",
    domain=["negative particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)
R_p = pybamm.SpatialVariable(
    "R_p",
    domain=["positive particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

# Domains at cell edges
x_n_edge = pybamm.SpatialVariableEdge(
    "x_n",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_s_edge = pybamm.SpatialVariableEdge(
    "x_s",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_p_edge = pybamm.SpatialVariableEdge(
    "x_p",
    domain=["positive electrode"],
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
r_macro_edge = pybamm.SpatialVariableEdge(
    "r_macro", domain="current collector", coord_sys="cylindrical polar"
)


r_n_edge = pybamm.SpatialVariableEdge(
    "r_n",
    domain=["negative particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p_edge = pybamm.SpatialVariableEdge(
    "r_p",
    domain=["positive particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)

R_n_edge = pybamm.SpatialVariableEdge(
    "R_n",
    domain=["negative particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)
R_p_edge = pybamm.SpatialVariableEdge(
    "R_p",
    domain=["positive particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

# Spatial variables for the core and shell phases in the core-shell model
# for PE phase transition caused degradation
# the core domain (0, s) and shell domain (s, R) have been mapped to (0, R_typ)
# core variable  r_co = r / s * R_typ, r goes from 0 to s
# shell variable r_sh = (r - s) / (R - s) * R_typ, r goes from s to R
r_co = pybamm.SpatialVariable(
    "r_co",
    domain=["positive core"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar"
)
r_sh = pybamm.SpatialVariable(
    "r_sh",
    domain=["positive shell"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    # note the coordinate system is different from that of r_co
    # thus the rhs of eqs residing in the core and shell differs
    coord_sys="cartesian"
)
r_co_prim = pybamm.SpatialVariable(
    "r_co_prim",
    domain=["positive primary core"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar"
)
r_sh_prim = pybamm.SpatialVariable(
    "r_sh_prim",
    domain=["positive primary shell"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian"
)
r_co_sec = pybamm.SpatialVariable(
    "r_co_sec",
    domain=["positive secondary core"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar"
)
r_sh_sec = pybamm.SpatialVariable(
    "r_sh_sec",
    domain=["positive secondary shell"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian"
)