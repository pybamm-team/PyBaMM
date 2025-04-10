import pybamm

whole_cell = ["negative electrode", "separator", "positive electrode"]

# Domains at cell centres
x_n = pybamm.SpatialVariable(
    "x_n",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
)
x_s = pybamm.SpatialVariable(
    "x_s",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
)
x_p = pybamm.SpatialVariable(
    "x_p",
    domain=["positive electrode"],
    auxiliary_domains={"secondary": "current collector"},
)
x = pybamm.SpatialVariable(
    "x",
    domain=whole_cell,
    auxiliary_domains={"secondary": "current collector"},
)

y = pybamm.SpatialVariable("y", domain="current collector")
z = pybamm.SpatialVariable("z", domain="current collector")
r_macro = pybamm.SpatialVariable("r_macro", domain="current collector")

r_n = pybamm.SpatialVariable(
    "r_n",
    domain=["negative particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)
r_p = pybamm.SpatialVariable(
    "r_p",
    domain=["positive particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)
r_n_prim = pybamm.SpatialVariable(
    "r_n_prim",
    domain=["negative primary particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)
r_p_prim = pybamm.SpatialVariable(
    "r_p_prim",
    domain=["positive primary particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)
r_n_sec = pybamm.SpatialVariable(
    "r_n_sec",
    domain=["negative secondary particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)
r_p_sec = pybamm.SpatialVariable(
    "r_p_sec",
    domain=["positive secondary particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)

R_n = pybamm.SpatialVariable(
    "R_n",
    domain=["negative particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)

R_p = pybamm.SpatialVariable(
    "R_p",
    domain=["positive particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)

R_n_prim = pybamm.SpatialVariable(
    "R_n_prim",
    domain=["negative primary particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)
R_p_prim = pybamm.SpatialVariable(
    "R_p_prim",
    domain=["positive primary particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)

R_n_sec = pybamm.SpatialVariable(
    "R_n_sec",
    domain=["negative secondary particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)
R_p_sec = pybamm.SpatialVariable(
    "R_p_sec",
    domain=["positive secondary particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)

# Domains at cell edges
x_n_edge = pybamm.SpatialVariableEdge(
    "x_n",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
)
x_s_edge = pybamm.SpatialVariableEdge(
    "x_s",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
)
x_p_edge = pybamm.SpatialVariableEdge(
    "x_p",
    domain=["positive electrode"],
    auxiliary_domains={"secondary": "current collector"},
)
x_edge = pybamm.SpatialVariableEdge(
    "x",
    domain=whole_cell,
    auxiliary_domains={"secondary": "current collector"},
)

y_edge = pybamm.SpatialVariableEdge("y", domain="current collector")
z_edge = pybamm.SpatialVariableEdge("z", domain="current collector")
r_macro_edge = pybamm.SpatialVariableEdge("r_macro", domain="current collector")


r_n_edge = pybamm.SpatialVariableEdge(
    "r_n",
    domain=["negative particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)
r_p_edge = pybamm.SpatialVariableEdge(
    "r_p",
    domain=["positive particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)

R_n_edge = pybamm.SpatialVariableEdge(
    "R_n",
    domain=["negative particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)
R_p_edge = pybamm.SpatialVariableEdge(
    "R_p",
    domain=["positive particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)
