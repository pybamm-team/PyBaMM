#
# Tests for the Finite Volume Method
#

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import eye, kron

import pybamm
from tests import (
    assert_symbolic_mesh_matches_numeric,
    get_1p1d_mesh_for_testing,
    get_mesh_for_testing,
    get_mesh_for_testing_symbolic,
    get_p2d_mesh_for_testing,
)


class TestFiniteVolume:
    def test_node_to_edge_to_node(self):
        # Create discretisation
        mesh = get_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)
        n = mesh["negative electrode"].npts

        # node to edge
        c = pybamm.StateVector(slice(0, n), domain=["negative electrode"])
        y_test = np.ones(n)
        diffusivity_c_ari = fin_vol.node_to_edge(c, method="arithmetic")
        np.testing.assert_array_equal(
            diffusivity_c_ari.evaluate(None, y_test), np.ones((n + 1, 1))
        )
        diffusivity_c_har = fin_vol.node_to_edge(c, method="harmonic")
        np.testing.assert_array_equal(
            diffusivity_c_har.evaluate(None, y_test), np.ones((n + 1, 1))
        )

        # edge to node
        d = pybamm.StateVector(slice(0, n + 1), domain=["negative electrode"])
        y_test = np.ones(n + 1)
        diffusivity_d_ari = fin_vol.edge_to_node(d, method="arithmetic")
        np.testing.assert_array_equal(
            diffusivity_d_ari.evaluate(None, y_test), np.ones((n, 1))
        )
        diffusivity_d_har = fin_vol.edge_to_node(d, method="harmonic")
        np.testing.assert_array_equal(
            diffusivity_d_har.evaluate(None, y_test), np.ones((n, 1))
        )

        # bad shift key
        with pytest.raises(ValueError, match=r"shift key"):
            fin_vol.shift(c, "bad shift key", "arithmetic")

        with pytest.raises(ValueError, match=r"shift key"):
            fin_vol.shift(c, "bad shift key", "harmonic")

        # bad method
        with pytest.raises(ValueError, match=r"method"):
            fin_vol.shift(c, "shift key", "bad method")

    @pytest.mark.parametrize("method", ["arithmetic", "harmonic"])
    def test_node_to_edge_exterior_matches_boundary_value(self, method):
        # On a non-uniform mesh the exterior edge values must agree with
        # boundary_value. A flux written as `D * grad(c)` takes D to the edges
        # through node_to_edge, while a Neumann condition written as
        # `-j / surf(D)` uses boundary_value; if the two disagree the imposed
        # flux is wrong and mass is not conserved.
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )
        geometry = {
            "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
            )
        }
        n = 16
        mesh = pybamm.Mesh(geometry, submesh_types, {r: n})
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)

        submesh = mesh["negative particle"]
        c = pybamm.StateVector(slice(0, n), domain=["negative particle"])
        # Linear field, so the exact edge values are known
        y_test = 1 + 2 * submesh.nodes

        edges = np.ravel(fin_vol.node_to_edge(c, method=method).evaluate(None, y_test))
        for side, index in [("left", 0), ("right", -1)]:
            boundary = np.ravel(
                fin_vol.boundary_value_or_flux(
                    pybamm.BoundaryValue(c, side), c
                ).evaluate(None, y_test)
            )
            np.testing.assert_allclose(edges[index], boundary, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                edges[index], 1 + 2 * submesh.edges[index], rtol=1e-12, atol=1e-12
            )

        if method == "arithmetic":
            # Linear interpolation of a linear field is exact everywhere, which
            # 0.5/0.5 interior weights are not on a non-uniform mesh
            np.testing.assert_allclose(
                edges, 1 + 2 * submesh.edges, rtol=1e-12, atol=1e-12
            )

    @pytest.mark.parametrize("method", ["arithmetic", "harmonic"])
    @pytest.mark.parametrize("shift_key", ["node to edge", "edge to node"])
    def test_shift_secondary_domain(self, shift_key, method):
        mesh = get_p2d_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)
        n_x = mesh["negative electrode"].npts
        size = mesh["negative particle"].npts + (shift_key == "edge to node")
        particle = {"primary": ["negative particle"]}
        single = pybamm.StateVector(slice(0, size), domains=particle)
        repeated = pybamm.StateVector(
            slice(0, n_x * size),
            domains={**particle, "secondary": ["negative electrode"]},
        )
        # positive, as the harmonic mean divides by the values
        values = 1 + np.random.default_rng(0).random((n_x, size))

        # each electrode point is shifted as a single particle would be
        expected = np.concatenate(
            [
                fin_vol.shift(single, shift_key, method).evaluate(y=row[:, None])
                for row in values
            ]
        )
        np.testing.assert_allclose(
            fin_vol.shift(repeated, shift_key, method).evaluate(
                y=values.reshape(-1, 1)
            ),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_node_to_edge_to_node_symbolic(self):
        # Create discretisation
        mesh = get_mesh_for_testing_symbolic()
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)
        n = mesh["domain"].npts

        # node to edge
        c = pybamm.StateVector(slice(0, n), domain=["domain"])
        y_test = np.ones(n)
        diffusivity_c_ari = fin_vol.node_to_edge(c, method="arithmetic")
        np.testing.assert_array_equal(
            diffusivity_c_ari.evaluate(None, y_test), np.ones((n + 1, 1))
        )
        diffusivity_c_har = fin_vol.node_to_edge(c, method="harmonic")
        np.testing.assert_array_equal(
            diffusivity_c_har.evaluate(None, y_test), np.ones((n + 1, 1))
        )

        # edge to node
        d = pybamm.StateVector(slice(0, n + 1), domain=["domain"])
        y_test = np.ones(n + 1)
        diffusivity_d_ari = fin_vol.edge_to_node(d, method="arithmetic")
        np.testing.assert_array_equal(
            diffusivity_d_ari.evaluate(None, y_test), np.ones((n, 1))
        )
        diffusivity_d_har = fin_vol.edge_to_node(d, method="harmonic")
        np.testing.assert_array_equal(
            diffusivity_d_har.evaluate(None, y_test), np.ones((n, 1))
        )

    def test_concatenation(self):
        mesh = get_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        edges = [pybamm.Vector(mesh[dom].edges, domain=dom) for dom in whole_cell]
        # Concatenation of edges should get averaged to nodes first, using edge_to_node
        v_disc = fin_vol.concatenation(edges)
        np.testing.assert_array_equal(v_disc.evaluate()[:, 0], mesh[whole_cell].nodes)

        # test for bad shape
        edges = [
            pybamm.Vector(np.ones(mesh[dom].npts + 2), domain=dom) for dom in whole_cell
        ]
        with pytest.raises(pybamm.ShapeError, match=r"child must have size n_nodes"):
            fin_vol.concatenation(edges)

    def test_discretise_diffusivity_times_spatial_operator(self):
        # Setup mesh and discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh[whole_cell]

        # Discretise some equations where averaging is needed
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y_test = np.ones_like(submesh.nodes[:, np.newaxis])
        for eqn in [
            var * pybamm.grad(var),
            var**2 * pybamm.grad(var),
            var * pybamm.grad(var) ** 2,
            var * (pybamm.grad(var) + 2),
            (pybamm.grad(var) + 2) * (-var),
            (pybamm.grad(var) + 2) * (2 * var),
            pybamm.grad(var) * pybamm.grad(var),
            (pybamm.grad(var) + 2) * pybamm.grad(var) ** 2,
            pybamm.div(pybamm.grad(var)),
            pybamm.div(pybamm.grad(var)) + 2,
            pybamm.div(pybamm.grad(var)) + var,
            pybamm.div(2 * pybamm.grad(var)),
            pybamm.div(2 * pybamm.grad(var)) + 3 * var,
            -2 * pybamm.div(var * pybamm.grad(var) + 2 * pybamm.grad(var)),
            pybamm.laplacian(var),
        ]:
            # Check that the equation can be evaluated for different combinations
            # of boundary conditions
            # Dirichlet
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # Neumann
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # One of each
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            disc.bcs = {
                var: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)

    def test_discretise_spatial_variable(self):
        # Create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # macroscale
        x1 = pybamm.SpatialVariable("x", ["negative electrode"])
        x1_disc = disc.process_symbol(x1)
        assert isinstance(x1_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x1_disc.evaluate(), disc.mesh["negative electrode"].nodes[:, np.newaxis]
        )
        # macroscale with concatenation
        x2 = pybamm.SpatialVariable("x", ["negative electrode", "separator"])
        x2_disc = disc.process_symbol(x2)
        assert isinstance(x2_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x2_disc.evaluate(),
            disc.mesh[("negative electrode", "separator")].nodes[:, np.newaxis],
        )
        # microscale
        r = 3 * pybamm.SpatialVariable("r", ["negative particle"])
        r_disc = disc.process_symbol(r)
        assert isinstance(r_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            r_disc.evaluate(), 3 * disc.mesh["negative particle"].nodes[:, np.newaxis]
        )

    def test_spatial_variable_symbolic(self):
        mesh = get_mesh_for_testing_symbolic()
        spatial_methods = {"domain": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        x = pybamm.SpatialVariable("x", ["domain"])
        x_disc = disc.process_symbol(x)
        assert isinstance(x_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x_disc.evaluate(),
            mesh["domain"].nodes[:, np.newaxis] * mesh["domain"].length,
        )

        # test spatial variable on edges
        x = pybamm.SpatialVariable("x", ["domain"])
        disc = pybamm.Discretisation(mesh, spatial_methods)
        x._saved_evaluates_on_edges = dict.fromkeys(
            pybamm.expression_tree.symbol.DOMAIN_LEVELS, True
        )
        x_edges_disc = disc.process_symbol(x)
        assert isinstance(x_edges_disc, pybamm.Vector)
        np.testing.assert_array_equal(
            x_edges_disc.evaluate(),
            mesh["domain"].edges[:, np.newaxis] * mesh["domain"].length,
        )

    def test_internal_neumann_condition_symbolic_length(self):
        x_left = pybamm.SpatialVariable("x_left", ["left"], coord_sys="cartesian")
        x_right = pybamm.SpatialVariable("x_right", ["right"], coord_sys="cartesian")
        a = pybamm.Variable("a", "left")
        b = pybamm.Variable("b", "right")

        def condition(middle, end, submesh):
            geometry = {
                "left": {x_left: {"min": pybamm.Scalar(0), "max": middle}},
                "right": {x_right: {"min": middle, "max": end}},
            }
            mesh = pybamm.Mesh(
                geometry, {"left": submesh, "right": submesh}, {x_left: 5, x_right: 5}
            )
            spatial_methods = {
                "left": pybamm.FiniteVolume(),
                "right": pybamm.FiniteVolume(),
            }
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc.set_variable_slices([a, b])
            return spatial_methods["left"].internal_neumann_condition(
                disc.process_symbol(a),
                disc.process_symbol(b),
                mesh["left"],
                mesh["right"],
            )

        length = pybamm.InputParameter("L")
        symbolic = condition(length, 2 * length, pybamm.SymbolicUniform1DSubMesh)
        numeric = condition(pybamm.Scalar(2), pybamm.Scalar(4), pybamm.Uniform1DSubMesh)
        y = np.linspace(0, 1, 10)[:, np.newaxis] ** 2
        assert_symbolic_mesh_matches_numeric(symbolic, numeric, y, {"L": 2})

    def test_internal_neumann_condition_secondary_domain(self):
        mesh = get_1p1d_mesh_for_testing(zpts=4)
        fin_vol = pybamm.FiniteVolume()
        disc = pybamm.Discretisation(
            mesh, {"macroscale": fin_vol, "current collector": pybamm.FiniteVolume()}
        )
        secondary = {"secondary": "current collector"}
        a = pybamm.Variable("a", "negative electrode", auxiliary_domains=secondary)
        b = pybamm.Variable("b", "separator", auxiliary_domains=secondary)
        disc.set_variable_slices([a, b])
        left, right = mesh["negative electrode"], mesh["separator"]
        condition = fin_vol.internal_neumann_condition(
            disc.process_symbol(a), disc.process_symbol(b), left, right
        )

        n_z = mesh["current collector"].npts
        a_values = np.linspace(0, 1, n_z * left.npts).reshape(n_z, left.npts) ** 2
        b_values = np.linspace(2, 3, n_z * right.npts).reshape(n_z, right.npts) ** 2
        y = np.concatenate([a_values.ravel(), b_values.ravel()])[:, np.newaxis]
        # one gradient across the interface for every current collector point
        expected = (b_values[:, 0] - a_values[:, -1]) / (
            right.nodes[0] - left.nodes[-1]
        )
        np.testing.assert_allclose(
            condition.evaluate(y=y).ravel(), expected, rtol=1e-12, atol=1e-12
        )

    def test_mass_matrix_shape(self):
        # Create model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.variables = {"c": c, "N": N}

        # Create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        disc.process_model(model)

        # Mass matrix
        mass = np.eye(submesh.npts)
        np.testing.assert_array_equal(mass, model.mass_matrix.entries.toarray())

    def test_p2d_mass_matrix_shape(self):
        # Create model
        c = pybamm.Variable(
            "c",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        N = pybamm.grad(c)
        model = pybamm.BaseModel()
        model.rhs = {c: pybamm.div(N)}
        model.initial_conditions = {c: pybamm.Scalar(0)}
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Dirichlet")}
        }
        model.variables = {"c": c, "N": N}

        # Create discretisation
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Mass matrix
        prim_pts = mesh["negative particle"].npts
        sec_pts = mesh["negative electrode"].npts
        mass_local = eye(prim_pts)
        mass = kron(eye(sec_pts), mass_local)
        np.testing.assert_array_equal(
            mass.toarray(), model.mass_matrix.entries.toarray()
        )

    @pytest.mark.parametrize(
        "operator",
        [
            "gradient",
            "divergence",
            "definite integral row",
            "definite integral column",
            "indefinite integral edges forward",
            "indefinite integral edges backward",
            "indefinite integral nodes forward",
            "indefinite integral nodes backward",
        ],
    )
    def test_operator_matrix_repeats_per_secondary_point(self, operator):
        mesh = get_p2d_mesh_for_testing()
        fin_vol = pybamm.FiniteVolume()
        fin_vol.build(mesh)

        def operator_matrix(domains):
            if operator == "gradient":
                matrix = fin_vol.gradient_matrix(domains["primary"], domains)
            elif operator == "divergence":
                matrix = fin_vol.divergence_matrix(domains)
            elif operator.startswith("definite integral"):
                child = pybamm.Variable("c", domains=domains)
                matrix = fin_vol.definite_integral_matrix(
                    child, vector_type=operator.split()[-1]
                )
            else:
                _, _, location, direction = operator.split()
                build = getattr(fin_vol, f"indefinite_integral_matrix_{location}")
                matrix = build(domains, direction)
            return matrix.evaluate().toarray()

        particle = {"primary": ["negative particle"]}
        single = operator_matrix(particle)
        repeated = operator_matrix({**particle, "secondary": ["negative electrode"]})
        np.testing.assert_allclose(
            repeated,
            block_diag(*[single] * mesh["negative electrode"].npts),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_jacobian(self):
        # Create discretisation
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        submesh = mesh[whole_cell]
        spatial_method = pybamm.FiniteVolume()
        spatial_method.build(mesh)

        # Setup variable
        var = pybamm.Variable("var", domain=whole_cell)
        disc.set_variable_slices([var])
        y = pybamm.StateVector(slice(0, submesh.npts))
        y_test = np.ones_like(submesh.nodes[:, np.newaxis])

        # grad
        eqn = pybamm.grad(var)
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(1), "Dirichlet"),
                "right": (pybamm.Scalar(2), "Dirichlet"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        jacobian = eqn_jac.evaluate(y=y_test)
        grad_matrix = spatial_method.gradient_matrix(
            whole_cell, {"primary": whole_cell}
        ).evaluate()
        np.testing.assert_allclose(jacobian.toarray()[1:-1], grad_matrix.toarray())
        np.testing.assert_allclose(
            jacobian.toarray()[0, 0], grad_matrix.toarray()[0][0] * -2
        )
        np.testing.assert_allclose(
            jacobian.toarray()[-1, -1], grad_matrix.toarray()[-1][-1] * -2
        )

        # grad with averaging
        eqn = var * pybamm.grad(var)
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

        # div(grad)
        flux = pybamm.grad(var)
        eqn = pybamm.div(flux)
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

        # div(grad) with averaging
        flux = var * pybamm.grad(var)
        eqn = pybamm.div(flux)
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(1), "Neumann"),
                "right": (pybamm.Scalar(2), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_jac = eqn_disc.jac(y)
        eqn_jac.evaluate(y=y_test)

    def test_boundary_value_domain(self):
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        c_s_n = pybamm.Variable(
            "c_s_n",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        c_s_p = pybamm.Variable(
            "c_s_p",
            domain=["positive particle"],
            auxiliary_domains={"secondary": ["positive electrode"]},
        )

        disc.set_variable_slices([c_s_n, c_s_p])

        # surface values
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)
        c_s_n_surf_disc = disc.process_symbol(c_s_n_surf)
        c_s_p_surf_disc = disc.process_symbol(c_s_p_surf)
        assert c_s_n_surf_disc.domain == ["negative electrode"]
        assert c_s_p_surf_disc.domain == ["positive electrode"]

    def test_delta_function(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var")
        delta_fn_left = pybamm.DeltaFunction(var, "left", "negative electrode")
        delta_fn_right = pybamm.DeltaFunction(var, "right", "negative electrode")
        disc.set_variable_slices([var])
        delta_fn_left_disc = disc.process_symbol(delta_fn_left)
        delta_fn_right_disc = disc.process_symbol(delta_fn_right)

        # Basic shape and type tests
        y = np.ones_like(mesh["negative electrode"].nodes[:, np.newaxis])
        # Left
        assert delta_fn_left_disc.domains == delta_fn_left.domains
        assert isinstance(delta_fn_left_disc, pybamm.Multiplication)
        assert isinstance(delta_fn_left_disc.left, pybamm.Symbol)
        np.testing.assert_allclose(delta_fn_left_disc.left.evaluate()[:, 1:], 0)
        assert delta_fn_left_disc.shape == y.shape
        # Right
        assert delta_fn_right_disc.domains == delta_fn_right.domains
        assert isinstance(delta_fn_right_disc, pybamm.Multiplication)
        assert isinstance(delta_fn_right_disc.left, pybamm.Symbol)
        np.testing.assert_array_equal(delta_fn_right_disc.left.evaluate()[:, :-1], 0)
        assert delta_fn_right_disc.shape == y.shape

        # Value tests
        # Delta function should integrate to the same thing as variable
        var_disc = disc.process_symbol(var)
        x = pybamm.standard_spatial_vars.x_n
        delta_fn_int_disc = disc.process_symbol(pybamm.Integral(delta_fn_left, x))
        np.testing.assert_allclose(
            var_disc.evaluate(y=y) * mesh["negative electrode"].edges[-1],
            np.sum(delta_fn_int_disc.evaluate(y=y)),
        )

    def test_delta_function_symbolic(self):
        mesh = get_mesh_for_testing_symbolic()
        spatial_methods = {"domain": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var")
        delta_fn_left = pybamm.DeltaFunction(var, "left", "domain")
        delta_fn_right = pybamm.DeltaFunction(var, "right", "domain")
        disc.set_variable_slices([var])
        delta_fn_left_disc = disc.process_symbol(delta_fn_left)
        delta_fn_right_disc = disc.process_symbol(delta_fn_right)

        # Basic shape and type tests
        y = np.ones_like(mesh["domain"].nodes[:, np.newaxis])
        # Left
        assert delta_fn_left_disc.domains == delta_fn_left.domains
        assert isinstance(delta_fn_left_disc, pybamm.Multiplication)
        assert isinstance(delta_fn_left_disc.left, pybamm.Symbol)
        np.testing.assert_allclose(delta_fn_left_disc.left.evaluate()[:, 1:], 0)
        assert delta_fn_left_disc.shape == y.shape
        # Right
        assert delta_fn_right_disc.domains == delta_fn_right.domains
        assert isinstance(delta_fn_right_disc, pybamm.Multiplication)
        assert isinstance(delta_fn_right_disc.left, pybamm.Symbol)
        np.testing.assert_array_equal(delta_fn_right_disc.left.evaluate()[:, :-1], 0)
        assert delta_fn_right_disc.shape == y.shape

        # Value tests
        # Delta function should integrate to the same thing as variable
        var_disc = disc.process_symbol(var)
        x = pybamm.SpatialVariable("x", ["domain"])
        delta_fn_int_disc = disc.process_symbol(pybamm.Integral(delta_fn_left, x))
        np.testing.assert_allclose(
            var_disc.evaluate(y=y) * mesh["domain"].edges[-1] * mesh["domain"].length
            + mesh["domain"].min,
            np.sum(delta_fn_int_disc.evaluate(y=y)),
        )

    def test_heaviside(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        var = pybamm.Variable("var", domain="negative electrode")
        heav = var > 1

        disc.set_variable_slices([var])
        # process_binary_operators should work with heaviside
        disc_heav = disc.process_symbol(heav * var)
        nodes = mesh["negative electrode"].nodes
        assert disc_heav.size == nodes.size
        np.testing.assert_array_equal(disc_heav.evaluate(y=2 * np.ones_like(nodes)), 2)
        np.testing.assert_array_equal(disc_heav.evaluate(y=-2 * np.ones_like(nodes)), 0)

    def test_upwind_downwind(self):
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n = mesh["negative electrode"].npts
        var = pybamm.StateVector(slice(0, n), domain="negative electrode")
        upwind = pybamm.upwind(var)
        downwind = pybamm.downwind(var)

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(5), "Dirichlet"),
                "right": (pybamm.Scalar(3), "Dirichlet"),
            }
        }

        disc_upwind = disc.process_symbol(upwind)
        disc_downwind = disc.process_symbol(downwind)

        nodes = mesh["negative electrode"].nodes
        n = mesh["negative electrode"].npts
        assert disc_upwind.size == nodes.size + 1
        assert disc_downwind.size == nodes.size + 1

        # The inflow face takes the Dirichlet value itself
        y_test = nodes**2
        np.testing.assert_array_equal(
            disc_upwind.evaluate(y=y_test), np.concatenate([[5], y_test])[:, np.newaxis]
        )
        np.testing.assert_array_equal(
            disc_downwind.evaluate(y=y_test),
            np.concatenate([y_test, [3]])[:, np.newaxis],
        )

        # Remove boundary conditions and check error is raised
        disc.bcs = {}
        disc._discretised_symbols = {}
        with pytest.raises(pybamm.ModelError, match=r"Boundary conditions"):
            disc.process_symbol(upwind)

        # Set wrong boundary conditions and check error is raised
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(5), "Neumann"),
                "right": (pybamm.Scalar(3), "Neumann"),
            }
        }
        with pytest.raises(pybamm.ModelError, match=r"Dirichlet boundary conditions"):
            disc.process_symbol(upwind)
        with pytest.raises(pybamm.ModelError, match=r"Dirichlet boundary conditions"):
            disc.process_symbol(downwind)

    def test_upwind_downwind_auxiliary_domains(self):
        mesh = get_p2d_mesh_for_testing()
        sp_meth = pybamm.FiniteVolume()
        sp_meth.build(mesh)
        disc = pybamm.Discretisation(
            mesh,
            {"macroscale": pybamm.FiniteVolume(), "negative particle": sp_meth},
        )

        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        bc_var = pybamm.Variable("bc_var", domain=["negative electrode"])
        disc.set_variable_slices([var, bc_var])
        disc_var = pybamm.StateVector(*disc.y_slices[var])
        disc_bc_var = pybamm.StateVector(*disc.y_slices[bc_var])

        n_prim = mesh["negative particle"].npts
        n_sec = mesh["negative electrode"].npts
        y_var = np.arange(1, n_prim * n_sec + 1, dtype=float)
        y_bc = -np.arange(1, n_sec + 1, dtype=float)
        y_test = np.concatenate([y_var, y_bc])
        y_var = y_var.reshape(n_sec, n_prim)

        # Symbolic boundary value, one per secondary point, and a constant one
        for value, expected in [(disc_bc_var, y_bc), (pybamm.Scalar(7), 7)]:
            bcs = {var: {"left": (value, "Dirichlet"), "right": (value, "Dirichlet")}}
            for direction in ["upwind", "downwind"]:
                faces = sp_meth.upwind_or_downwind(var, disc_var, bcs, direction)
                faces = faces.evaluate(None, y_test).reshape(n_sec, n_prim + 1)
                if direction == "upwind":
                    inflow, interior = faces[:, 0], faces[:, 1:]
                else:
                    inflow, interior = faces[:, -1], faces[:, :-1]
                np.testing.assert_array_equal(inflow, expected)
                np.testing.assert_array_equal(interior, y_var)

    def test_grad_div_with_bcs_on_tab(self):
        # Create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # var
        y_test = np.ones(mesh["current collector"].npts)
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        # grad
        grad_eqn = pybamm.grad(var)
        # div
        N = pybamm.grad(var)
        div_eqn = pybamm.div(N)

        # bcs (on each tab)
        boundary_conditions = {
            var: {
                "negative tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

        # bcs (one pos, one not tab)
        boundary_conditions = {
            var: {
                "no tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

        # bcs (one neg, one not tab)
        boundary_conditions = {
            var: {
                "negative tab": (pybamm.Scalar(1), "Neumann"),
                "no tab": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        grad_eqn_disc = disc.process_symbol(grad_eqn)
        grad_eqn_disc.evaluate(None, y_test)
        div_eqn_disc = disc.process_symbol(div_eqn)
        div_eqn_disc.evaluate(None, y_test)

    def test_neg_pos_bcs(self):
        # Create discretisation
        mesh = get_1p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # var
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        # grad
        grad_eqn = pybamm.grad(var)

        # bcs (on each tab)
        boundary_conditions = {
            var: {
                "negative tab": (pybamm.Scalar(1), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
                "no tab": (pybamm.Scalar(8), "Dirichlet"),
            }
        }
        disc.bcs = boundary_conditions

        # check after disc that negative tab goes to left and positive tab goes
        # to right
        disc.process_symbol(grad_eqn)
        assert disc.bcs[var]["left"][0] == pybamm.Scalar(1)
        assert disc.bcs[var]["left"][1] == "Dirichlet"
        assert disc.bcs[var]["right"][0] == pybamm.Scalar(0)
        assert disc.bcs[var]["right"][1] == "Neumann"

    def test_full_broadcast_domains(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator"], scale=100
        )
        model.rhs = {var: 0}
        a = pybamm.InputParameter("a")
        ic = pybamm.concatenation(
            pybamm.FullBroadcast(a * 100, "negative electrode"),
            pybamm.FullBroadcast(100, "separator"),
        )
        model.initial_conditions = {var: ic}

        mesh = get_mesh_for_testing()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

    def test_evaluate_at(self):
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        n = mesh["negative electrode"].npts
        var = pybamm.StateVector(slice(0, n), domain="negative electrode")

        idx = 3
        position = pybamm.Scalar(mesh["negative electrode"].nodes[idx])
        evaluate_at = pybamm.EvaluateAt(var, position)
        evaluate_at_disc = disc.process_symbol(evaluate_at)

        assert isinstance(evaluate_at_disc, pybamm.MatrixMultiplication)
        assert isinstance(evaluate_at_disc.left, pybamm.Matrix)
        assert isinstance(evaluate_at_disc.right, pybamm.StateVector)

        y = np.arange(n)[:, np.newaxis]
        assert evaluate_at_disc.evaluate(y=y) == y[idx]

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(n - 1), "Dirichlet"),
            }
        }
        position = pybamm.Scalar(mesh["negative electrode"].edges[-1])
        downwind_var = pybamm.downwind(var)
        evaluate_at = pybamm.EvaluateAt(downwind_var, position)

        assert evaluate_at.evaluates_on_edges("primary") is False

        evaluate_at_disc = disc.process_symbol(evaluate_at)

        assert evaluate_at_disc.evaluate(y=y) == n - 1

        mesh = get_mesh_for_testing_symbolic()
        spatial_methods = {"domain": pybamm.FiniteVolume()}
        var = pybamm.Variable("var", domain="domain")
        disc = pybamm.Discretisation(mesh, spatial_methods)
        evaluate_at = pybamm.EvaluateAt(var, position)
        with pytest.raises(pybamm.ModelError):
            disc.process_symbol(evaluate_at)

    def test_evaluate_at_secondary_domain(self):
        mesh = get_p2d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable(
            "c",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        disc.set_variable_slices([var])

        idx = 2
        position = pybamm.Scalar(mesh["negative particle"].nodes[idx])
        evaluate_at_disc = disc.process_symbol(pybamm.EvaluateAt(var, position))

        n_r = mesh["negative particle"].npts
        n_x = mesh["negative electrode"].npts
        y = np.arange(n_r * n_x, dtype=np.float64)[:, np.newaxis]
        # the particle value at ``position`` for every electrode point
        np.testing.assert_array_equal(
            evaluate_at_disc.evaluate(y=y).ravel(), y.reshape(n_x, n_r)[:, idx]
        )

    def test_inner(self):
        # standard
        mesh = get_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
        }

        var = pybamm.Variable("var", domain="negative particle")
        grad_var = pybamm.grad(var)
        inner = pybamm.inner(grad_var, grad_var)

        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.set_variable_slices([var])
        boundary_conditions = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
        disc.bcs = boundary_conditions
        inner_disc = disc.process_symbol(inner)

        assert isinstance(inner_disc, pybamm.Inner)
        assert isinstance(inner_disc.left, pybamm.MatrixMultiplication)
        assert isinstance(inner_disc.right, pybamm.MatrixMultiplication)

        n = mesh["negative particle"].npts
        y = np.ones(n)[:, np.newaxis]
        np.testing.assert_array_equal(inner_disc.evaluate(y=y), np.zeros((n, 1)))
        mesh = get_mesh_for_testing()

        # with secondary broadcast
        grad_var = pybamm.grad(pybamm.SecondaryBroadcast(var, "negative electrode"))
        inner = pybamm.inner(grad_var, grad_var)

        inner_disc = disc.process_symbol(inner)

        assert isinstance(inner_disc, pybamm.Inner)
        assert isinstance(inner_disc.left, pybamm.MatrixMultiplication)
        assert isinstance(inner_disc.right, pybamm.MatrixMultiplication)

        m = mesh["negative electrode"].npts
        np.testing.assert_array_equal(inner_disc.evaluate(y=y), np.zeros((n * m, 1)))
