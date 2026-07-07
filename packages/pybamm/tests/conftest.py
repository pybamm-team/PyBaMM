import pytest

import pybamm


@pytest.fixture
def mesh_2d():
    param = pybamm.ParameterValues(
        values={
            "Electrode height [m]": 1,
            "Negative electrode thickness [m]": 0.3333333333333333,
            "Separator thickness [m]": 0.3333333333333333,
            "Positive electrode thickness [m]": 0.3333333333333334,
            "Negative particle radius [m]": 0.5,
            "Positive particle radius [m]": 0.5,
        }
    )

    x = pybamm.SpatialVariable(
        "x", ["negative electrode", "separator", "positive electrode"], direction="lr"
    )
    z = pybamm.SpatialVariable(
        "z", ["negative electrode", "separator", "positive electrode"], direction="tb"
    )
    r_n = pybamm.SpatialVariable("r_n", ["negative particle"])
    r_p = pybamm.SpatialVariable("r_p", ["positive particle"])

    geometry = {
        "negative electrode": {
            x: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Negative electrode thickness [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
        "separator": {
            x: {
                "min": pybamm.Parameter("Negative electrode thickness [m]"),
                "max": pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
        "positive electrode": {
            x: {
                "min": pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
                "max": pybamm.Parameter("Positive electrode thickness [m]")
                + pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
        "negative particle": {
            r_n: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Scalar(1),
            },
        },
        "positive particle": {
            r_p: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Scalar(1),
            },
        },
    }

    param.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.Uniform2DSubMesh,
        "separator": pybamm.Uniform2DSubMesh,
        "positive electrode": pybamm.Uniform2DSubMesh,
        "negative particle": pybamm.Uniform1DSubMesh,
        "positive particle": pybamm.Uniform1DSubMesh,
    }

    var_pts = {
        x: 40,
        z: 15,
        r_n: 10,
        r_p: 10,
    }

    return pybamm.Mesh(geometry, submesh_types, var_pts)
