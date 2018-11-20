#
# Tests for the electrolyte class
#
import pybamm

import numpy as np
from numpy.linalg import norm

import unittest


class TestElectrolyte(unittest.TestCase):
    def test_cation_conservation_finite_volumes_convergence(self):
        electrolyte = pybamm.Electrolyte()

        # Finite volume only has h**2 convergence if the mesh is uniform?
        uniform_lengths = {"Ln": 1e-3, "Ls": 1e-3, "Lp": 1e-3}
        param = pybamm.Parameters(
            optional_parameters=uniform_lengths, tests="convergence"
        )

        # Test convergence
        ns = [100, 200, 400]
        errs = [0] * len(ns)
        for i, n in enumerate(ns):
            # Set up
            mesh = pybamm.Mesh(param, n)
            param.set_mesh_dependent_parameters(mesh)
            c = np.cos(2 * np.pi * mesh.xc)
            operators = {"xc": pybamm.Operators("Finite Volumes", "xc", mesh)}
            lbc = np.array([0])
            rbc = np.array([0])
            dcdt_exact = -4 * np.pi ** 2 * c

            # Calculate solution and errors
            electrolyte.set_simulation(param, operators, mesh)

            dcdt = electrolyte.cation_conservation(c, 0, (lbc, rbc))
            errs[i] = norm(dcdt - dcdt_exact) / norm(dcdt_exact)

        # Expect h**2 convergence
        [
            self.assertLess(errs[i + 1] / errs[i], 0.26)
            for i in range(len(errs) - 1)
        ]

    def test_macinnes_finite_volumes_convergence(self):
        electrolyte = pybamm.Electrolyte()

        # Finite volume only has h**2 convergence if the mesh is uniform?
        uniform_lengths = {"Ln": 1e-3, "Ls": 1e-3, "Lp": 1e-3}
        param = pybamm.Parameters(
            optional_parameters=uniform_lengths, tests="convergence"
        )

        # Test convergence
        ns = [100, 200, 400]
        errn = [0] * len(ns)
        errp = [0] * len(ns)
        for i, n in enumerate(ns):
            # Set up
            mesh = pybamm.Mesh(param, n)
            param.set_mesh_dependent_parameters(mesh)
            cn = np.cos(2 * np.pi * mesh.xcn)
            cp = np.sin(2 * np.pi * mesh.xcp)
            en = mesh.xcn ** 2
            ep = mesh.xcp ** 2
            operators = {
                "xcn": pybamm.Operators("Finite Volumes", "xcn", mesh),
                "xcp": pybamm.Operators("Finite Volumes", "xcp", mesh),
            }
            in_exact = -2 * np.pi * np.sin(2 * np.pi * mesh.xn) + 2 * mesh.xn
            ip_exact = 2 * np.pi * np.cos(2 * np.pi * mesh.xp) + 2 * mesh.xp
            current_bcs_n = (in_exact[0, None], in_exact[-1, None])
            current_bcs_p = (ip_exact[0, None], ip_exact[-1, None])

            # Calculate solution and errors
            electrolyte.set_simulation(param, operators, mesh)

            i_n = electrolyte.macinnes("xcn", cn, en, current_bcs_n)
            i_p = electrolyte.macinnes("xcp", cp, ep, current_bcs_p)
            errn[i] = norm(i_n - in_exact) / norm(in_exact)
            errp[i] = norm(i_p - ip_exact) / norm(ip_exact)

        # Expect h**2 convergence
        for i in range(len(errn) - 1):
            self.assertLess(errn[i + 1] / errn[i], 0.26)
            self.assertLess(errp[i + 1] / errp[i], 0.26)

    def test_current_conservation_finite_volumes_convergence(self):
        electrolyte = pybamm.Electrolyte()

        # Finite volume only has h**2 convergence if the mesh is uniform?
        uniform_lengths = {"Ln": 1e-3, "Ls": 1e-3, "Lp": 1e-3}
        param = pybamm.Parameters(
            optional_parameters=uniform_lengths, tests="convergence"
        )

        # Test convergence
        ns = [100, 200, 400]
        errn = [0] * len(ns)
        errp = [0] * len(ns)
        for i, n in enumerate(ns):
            # Set up
            mesh = pybamm.Mesh(param, n)
            param.set_mesh_dependent_parameters(mesh)
            cn = np.cos(2 * np.pi * mesh.xcn)
            cp = np.cos(2 * np.pi * mesh.xcp)
            en = mesh.xcn ** 2
            ep = mesh.xcp ** 2
            operators = {
                "xcn": pybamm.Operators("Finite Volumes", "xcn", mesh),
                "xcp": pybamm.Operators("Finite Volumes", "xcp", mesh),
            }
            in_exact = (-2 * np.pi * np.sin(2 * np.pi * mesh.xn)) + 2 * mesh.xn
            ip_exact = (
                -2 * np.pi * np.sin(2 * np.pi * mesh.xp / param.lp)
            ) + 2 * mesh.xp
            current_bcs_n = (in_exact[0, None], in_exact[-1, None])
            current_bcs_p = (ip_exact[0, None], ip_exact[-1, None])

            # Exact solutions
            dendt_exact = 1 / param.gamma_dl_n * (-4 * np.pi ** 2 * cn + 2)
            depdt_exact = 1 / param.gamma_dl_p * (-4 * np.pi ** 2 * cp + 2)

            # Calculate solution and errors
            electrolyte.set_simulation(param, operators, mesh)
            dendt = electrolyte.current_conservation(
                "xcn", cn, en, 0, current_bcs_n
            )
            depdt = electrolyte.current_conservation(
                "xcp", cp, ep, 0, current_bcs_p
            )
            errn[i] = norm((dendt - dendt_exact)[1:-1]) / norm(
                dendt_exact[1:-1]
            )
            errp[i] = norm((depdt - depdt_exact)[1:-1]) / norm(
                depdt_exact[1:-1]
            )
        # Expect h**2 convergence
        for i in range(len(errn) - 1):
            self.assertLess(errn[i + 1] / errn[i], 0.26)
            self.assertLess(errp[i + 1] / errp[i], 0.26)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
