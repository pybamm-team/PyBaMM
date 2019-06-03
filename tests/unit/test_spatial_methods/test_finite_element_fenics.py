#
# Test for the operator class
#
import pybamm
from tests import get_2p1d_mesh_for_testing
import numpy as np
import unittest
from pybamm.spatial_methods.finite_element_fenics import dolfin_spec


@unittest.skipIf(dolfin_spec is None, "dolfin not installed")
class TestFiniteElementFenics(unittest.TestCase):
    def test_not_implemented(self):
        mesh = get_2p1d_mesh_for_testing()
        spatial_method = pybamm.FiniteElementFenics(mesh)
        self.assertEqual(spatial_method.mesh, mesh)
        with self.assertRaises(NotImplementedError):
            spatial_method.gradient(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.divergence(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.indefinite_integral(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.boundary_value_or_flux(None, None)

    def test(self):
        # get mesh
        mesh = get_2p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteElementFenics,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # discretise some equations
        var = pybamm.Variable("var", domain="current collector")
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        disc.set_variable_slices([var])
        y_test = np.ones(mesh["current collector"][0].npts)
        unit_source = pybamm.Broadcast(1, "current collector")
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

        for eqn in [
            pybamm.laplacian(var),
            pybamm.source(unit_source, var),
            pybamm.laplacian(var) - pybamm.source(unit_source, var),
            pybamm.source(var, var),
            pybamm.laplacian(var) - pybamm.source(2 * var, var),
            pybamm.Integral(var, [y, z]) - 1,
        ]:
            # Check that equation can be evaluated in each case
            # Dirichlet
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # Neumann
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # One of each
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # One of each
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)

        # check  ValueError raised for non Dirichlet or Neumann BCs
        eqn = pybamm.laplacian(var) - pybamm.source(unit_source, var)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Other BC"),
            }
        }
        with self.assertRaises(ValueError):
            eqn_disc = disc.process_symbol(eqn)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(0), "Other BC"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        with self.assertRaises(ValueError):
            eqn_disc = disc.process_symbol(eqn)

    def test_definite_integral(self):
        mesh = get_2p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteElementFenics,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="current collector")
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        integral_eqn = pybamm.Integral(var, [y, z])
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        y_test = 6 * np.ones(mesh["current collector"][0].npts)
        fem_mesh = mesh["current collector"][0]
        ly = fem_mesh.y_lims["max"]
        lz = fem_mesh.z_lims["max"]
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, y_test), 6 * ly * lz
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
