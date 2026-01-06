"""Tests for TensorField, TensorProduct, and tensor divergence in 2D finite volume."""

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.tensor_field import TensorField
from tests import get_mesh_for_testing_2d


class TestTensorField:
    """Tests for TensorField creation and properties."""

    def test_rank1_tensor_creation(self):
        """Test creating a rank-1 tensor (vector)."""
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        t = TensorField([a, b])

        assert t.rank == 1
        assert t.shape == (2,)
        assert t[0] == a
        assert t[1] == b
        assert len(t.children) == 2

    def test_rank2_tensor_creation(self):
        """Test creating a rank-2 tensor (matrix)."""
        a, b, c, d = [pybamm.Scalar(i) for i in range(4)]
        t = TensorField([[a, b], [c, d]])

        assert t.rank == 2
        assert t.shape == (2, 2)
        assert t[0, 0] == a
        assert t[0, 1] == b
        assert t[1, 0] == c
        assert t[1, 1] == d
        assert len(t.children) == 4

    def test_rank2_row_access(self):
        """Test accessing a row of a rank-2 tensor."""
        a, b, c, d = [pybamm.Scalar(i) for i in range(4)]
        t = TensorField([[a, b], [c, d]])

        row0 = t[0]
        assert row0 == [a, b]
        row1 = t[1]
        assert row1 == [c, d]

    def test_tensor_domain_validation(self):
        """Test that components must have matching domains."""
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="positive electrode")

        with pytest.raises(ValueError, match="same domain"):
            TensorField([a, b])

    def test_tensor_empty_components_error(self):
        """Test that empty components raise error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TensorField([])

    def test_tensor_row_length_validation(self):
        """Test that rank-2 tensor rows must have same length."""
        a, b, c = [pybamm.Scalar(i) for i in range(3)]

        with pytest.raises(ValueError, match="Row 1 has"):
            TensorField([[a, b], [c]])

    def test_tensor_create_copy(self):
        """Test creating a copy of a tensor."""
        a, b = pybamm.Scalar(1), pybamm.Scalar(2)
        t = TensorField([a, b])
        t_copy = t.create_copy()

        assert t_copy.rank == t.rank
        assert t_copy.shape == t.shape

    def test_rank2_tensor_create_copy(self):
        """Test creating a copy of a rank-2 tensor."""
        components = [[pybamm.Scalar(i + j * 2) for i in range(2)] for j in range(2)]
        t = TensorField(components)
        t_copy = t.create_copy()

        assert t_copy.rank == 2
        assert t_copy.shape == (2, 2)

    def test_evaluates_on_edges_all_false(self):
        """Test evaluates_on_edges returns False when no components are on edges."""
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        t = TensorField([a, b])

        assert t.evaluates_on_edges("primary") is False

    def test_evaluates_on_edges_all_true(self):
        """Test evaluates_on_edges returns True when all components are on edges."""
        # Create variables and take gradients (which evaluate on edges)
        var = pybamm.Variable("var", domain="negative electrode")
        grad_var = pybamm.grad(var)

        # VectorField from gradient components - both evaluate on edges
        # Note: grad returns VectorField, so we test with that
        assert grad_var.evaluates_on_edges("primary") is True

    def test_evaluates_on_edges_mixed_raises(self):
        """Test evaluates_on_edges raises error for mixed edge/node components."""
        # Create symbols with different edge evaluation status
        var = pybamm.Variable("var", domain="negative electrode")
        # Gradient evaluates on edges
        edge_symbol = pybamm.grad(var)
        # Variable evaluates on nodes
        node_symbol = var

        # Create tensor with mixed edge status - should raise on evaluates_on_edges
        t = TensorField([edge_symbol, node_symbol], domain=["negative electrode"])

        with pytest.raises(ValueError, match="All tensor components must either"):
            t.evaluates_on_edges("primary")

    def test_rank2_evaluates_on_edges(self):
        """Test evaluates_on_edges for rank-2 tensor."""
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        c = pybamm.Scalar(3)
        d = pybamm.Scalar(4)

        t = TensorField([[a, b], [c, d]])
        assert t.evaluates_on_edges("primary") is False


class TestVectorFieldInheritance:
    """Tests for VectorField inheriting from TensorField."""

    def test_vectorfield_is_tensorfield(self):
        """Test that VectorField is a subclass of TensorField."""
        a, b = pybamm.Scalar(1), pybamm.Scalar(2)
        vf = pybamm.VectorField(a, b)

        assert isinstance(vf, TensorField)
        assert isinstance(vf, pybamm.VectorField)

    def test_vectorfield_backward_compatibility(self):
        """Test that VectorField lr_field/tb_field properties work."""
        a, b = pybamm.Scalar(1), pybamm.Scalar(2)
        vf = pybamm.VectorField(a, b)

        assert vf.lr_field == a
        assert vf.tb_field == b
        assert vf.rank == 1
        assert vf.shape == (2,)

    def test_vectorfield_domain_validation(self):
        """Test VectorField domain validation."""
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="positive electrode")

        with pytest.raises(ValueError, match="same domain"):
            pybamm.VectorField(a, b)


class TestTensorProduct:
    """Tests for TensorProduct operator."""

    def test_tensor_product_creation(self):
        """Test creating a tensor product."""
        a, b = pybamm.Scalar(1), pybamm.Scalar(2)
        vf1 = pybamm.VectorField(a, b)
        vf2 = pybamm.VectorField(a, b)

        tp = pybamm.TensorProduct(vf1, vf2)
        assert tp.result_rank == 2

    def test_tensor_product_convenience_function(self):
        """Test tensor_product convenience function."""
        a, b = pybamm.Scalar(1), pybamm.Scalar(2)
        vf1 = pybamm.VectorField(a, b)
        vf2 = pybamm.VectorField(a, b)

        tp = pybamm.tensor_product(vf1, vf2)
        assert isinstance(tp, pybamm.TensorProduct)
        assert tp.result_rank == 2

    def test_tensor_product_scalar_vector(self):
        """Test tensor product of scalar and vector (result rank 1)."""
        s = pybamm.Scalar(2)
        a, b = pybamm.Scalar(1), pybamm.Scalar(2)
        vf = pybamm.VectorField(a, b)

        tp = pybamm.tensor_product(s, vf)
        assert tp.result_rank == 1

    def test_tensor_product_rank_limit(self):
        """Test that tensor product rejects rank > 2 results."""
        components = [[pybamm.Scalar(i) for i in range(2)] for _ in range(2)]
        t1 = TensorField(components)
        a, b = pybamm.Scalar(1), pybamm.Scalar(2)
        vf = pybamm.VectorField(a, b)

        with pytest.raises(ValueError, match="exceeds maximum of 2"):
            pybamm.tensor_product(t1, vf)

    def test_tensor_product_evaluate(self):
        """Test tensor product numerical evaluation."""
        tp = pybamm.TensorProduct(pybamm.Scalar(2), pybamm.Scalar(3))
        result = tp.evaluate()
        np.testing.assert_array_equal(result, np.array([[6]]))


class TestTensorDivergence:
    """Tests for divergence of tensor fields."""

    @pytest.fixture
    def setup_divergence_test(self):
        """Set up mesh and discretisation for divergence tests."""
        mesh = get_mesh_for_testing_2d()
        fin_vol = pybamm.FiniteVolume2D()
        fin_vol.build(mesh)

        var = pybamm.Variable(
            "var",
            domain=["negative electrode", "separator", "positive electrode"],
        )
        disc = pybamm.Discretisation(mesh, {"macroscale": fin_vol})
        disc.set_variable_slices([var])

        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        n_lr = submesh.npts_lr
        n_tb = submesh.npts_tb

        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)

        return {
            "mesh": mesh,
            "fin_vol": fin_vol,
            "disc": disc,
            "var": var,
            "n_lr": n_lr,
            "n_tb": n_tb,
            "LR": LR,
            "TB": TB,
        }

    def test_divergence_accepts_tensorfield(self):
        """Test that Divergence accepts a TensorField."""
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="negative electrode")
        vf = pybamm.VectorField(pybamm.grad(a), pybamm.grad(b))

        # Divergence should accept TensorField (VectorField)
        div_vf = pybamm.div(vf)
        assert isinstance(div_vf, pybamm.Divergence)

    def test_vector_divergence_returns_scalar(self, setup_divergence_test):
        """Test that divergence of VectorField (rank-1) returns scalar."""
        disc = setup_divergence_test["disc"]
        var = setup_divergence_test["var"]
        LR = setup_divergence_test["LR"]

        # Set up boundary conditions
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        # div(grad(var)) should return scalar
        eqn = pybamm.div(pybamm.grad(var))
        eqn_disc = disc.process_symbol(eqn)

        # Result should be scalar (not VectorField)
        assert not isinstance(eqn_disc, pybamm.VectorField)

        # Should be evaluable
        result = eqn_disc.evaluate(None, LR.flatten())
        assert result is not None

    def test_rank2_tensor_divergence_returns_vector(self, setup_divergence_test):
        """Test that divergence of rank-2 TensorField returns VectorField."""
        disc = setup_divergence_test["disc"]
        fin_vol = setup_divergence_test["fin_vol"]
        var = setup_divergence_test["var"]
        n_lr = setup_divergence_test["n_lr"]
        n_tb = setup_divergence_test["n_tb"]

        # Set up boundary conditions
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        # Create gradient and tensor product
        grad_var = pybamm.grad(var)
        disc_grad = disc.process_symbol(grad_var)

        # Create rank-2 tensor from outer product of gradients
        disc_tensor = fin_vol._tensor_product(grad_var, grad_var, disc_grad, disc_grad)
        assert disc_tensor.rank == 2

        # Apply divergence to rank-2 tensor - should return VectorField
        # Create a mock symbol for the divergence call
        div_result = fin_vol.divergence(grad_var, disc_tensor, disc.bcs)

        assert isinstance(div_result, pybamm.VectorField)
        assert hasattr(div_result, "lr_field")
        assert hasattr(div_result, "tb_field")

        # Each component should be evaluable
        y_test = np.linspace(0, 1, n_lr * n_tb)
        lr_result = div_result.lr_field.evaluate(None, y_test)
        tb_result = div_result.tb_field.evaluate(None, y_test)

        assert lr_result.shape[0] == n_lr * n_tb
        assert tb_result.shape[0] == n_lr * n_tb

    def test_divergence_with_different_boundary_conditions(self, setup_divergence_test):
        """Test divergence evaluation with various boundary conditions."""
        disc = setup_divergence_test["disc"]
        var = setup_divergence_test["var"]
        LR = setup_divergence_test["LR"]
        TB = setup_divergence_test["TB"]

        eqn = pybamm.div(pybamm.grad(var))

        # Test Dirichlet BCs
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_disc.evaluate(None, LR.flatten())
        eqn_disc.evaluate(None, TB.flatten())

        # Test Neumann BCs
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(1), "Neumann"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(1), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_disc.evaluate(None, LR.flatten())
        eqn_disc.evaluate(None, TB.flatten())

        # Test mixed BCs
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Neumann"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Neumann"),
            }
        }
        eqn_disc = disc.process_symbol(eqn)
        eqn_disc.evaluate(None, LR.flatten())
        eqn_disc.evaluate(None, TB.flatten())

    def test_divergence_of_scaled_gradient(self, setup_divergence_test):
        """Test divergence of scaled gradient (common in diffusion equations)."""
        disc = setup_divergence_test["disc"]
        var = setup_divergence_test["var"]
        LR = setup_divergence_test["LR"]

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        # div(D * grad(var)) where D is a scalar diffusivity
        for eqn in [
            pybamm.div(2 * pybamm.grad(var)),
            pybamm.div(var * pybamm.grad(var)),
            pybamm.div(2 * pybamm.grad(var)) + 3 * var,
        ]:
            eqn_disc = disc.process_symbol(eqn)
            result = eqn_disc.evaluate(None, LR.flatten())
            assert result is not None


class TestTensorDiscretisation:
    """Tests for tensor discretisation in 2D finite volume."""

    @pytest.fixture
    def setup_2d_mesh(self):
        """Set up 2D mesh and discretisation."""
        mesh = get_mesh_for_testing_2d()
        fin_vol = pybamm.FiniteVolume2D()
        fin_vol.build(mesh)

        var = pybamm.Variable(
            "var",
            domain=["negative electrode", "separator", "positive electrode"],
        )
        disc = pybamm.Discretisation(mesh, {"macroscale": fin_vol})
        disc.set_variable_slices([var])

        n_lr = mesh[("negative electrode", "separator", "positive electrode")].npts_lr
        n_tb = mesh[("negative electrode", "separator", "positive electrode")].npts_tb

        return {
            "mesh": mesh,
            "fin_vol": fin_vol,
            "disc": disc,
            "var": var,
            "n_lr": n_lr,
            "n_tb": n_tb,
        }

    def test_tensorfield_discretisation(self, setup_2d_mesh):
        """Test that TensorField components get discretised."""
        disc = setup_2d_mesh["disc"]
        var = setup_2d_mesh["var"]

        # Create a rank-2 tensor with variable components
        t = TensorField(
            [[var, var * 2], [var * 3, var * 4]],
            domain=var.domain,
        )

        disc_t = disc.process_symbol(t)
        assert isinstance(disc_t, TensorField)
        assert disc_t.rank == 2
        assert disc_t.shape == (2, 2)

    def test_vectorfield_discretisation(self, setup_2d_mesh):
        """Test VectorField discretisation (backward compatibility)."""
        disc = setup_2d_mesh["disc"]
        var = setup_2d_mesh["var"]

        vf = pybamm.VectorField(var, var * 2)
        disc_vf = disc.process_symbol(vf)

        assert isinstance(disc_vf, pybamm.VectorField)
        assert hasattr(disc_vf, "lr_field")
        assert hasattr(disc_vf, "tb_field")

    def test_tensor_product_discretisation(self, setup_2d_mesh):
        """Test TensorProduct discretisation produces rank-2 TensorField."""
        disc = setup_2d_mesh["disc"]
        fin_vol = setup_2d_mesh["fin_vol"]
        var = setup_2d_mesh["var"]
        n_lr = setup_2d_mesh["n_lr"]
        n_tb = setup_2d_mesh["n_tb"]

        # Create gradient (VectorField) and tensor product
        grad_var = pybamm.grad(var)

        # Set up boundary conditions for gradient
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
            }
        }

        disc_grad = disc.process_symbol(grad_var)
        assert isinstance(disc_grad, pybamm.VectorField)

        # Tensor product of gradient with itself
        disc_tp = fin_vol._tensor_product(grad_var, grad_var, disc_grad, disc_grad)

        assert isinstance(disc_tp, TensorField)
        assert disc_tp.rank == 2
        assert disc_tp.shape == (2, 2)

        # Evaluate with test values
        y_test = np.linspace(0, 1, n_lr * n_tb)
        result_00 = disc_tp[0, 0].evaluate(None, y_test)
        assert result_00.shape[0] == n_lr * n_tb


class TestTensorAccuracy:
    """Accuracy tests for tensor operations following finite volume conventions."""

    @pytest.fixture
    def setup_accuracy_test(self):
        """Set up mesh and discretisation for accuracy tests."""
        mesh = get_mesh_for_testing_2d()
        fin_vol = pybamm.FiniteVolume2D()
        fin_vol.build(mesh)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        disc = pybamm.Discretisation(mesh, {"macroscale": fin_vol})
        disc.set_variable_slices([var])

        submesh = mesh[tuple(whole_cell)]
        n_lr = submesh.npts_lr
        n_tb = submesh.npts_tb

        LR, TB = np.meshgrid(submesh.nodes_lr, submesh.nodes_tb)

        return {
            "mesh": mesh,
            "fin_vol": fin_vol,
            "disc": disc,
            "var": var,
            "submesh": submesh,
            "n_lr": n_lr,
            "n_tb": n_tb,
            "LR": LR,
            "TB": TB,
        }

    def test_tensor_product_of_constant_gradient(self, setup_accuracy_test):
        """Test tensor product of grad(x) gives expected values.

        If u = x (linear in x), then grad(u) = [1, 0].
        Tensor product grad(u) ⊗ grad(u) should give:
        [[1, 0],
         [0, 0]]
        """
        disc = setup_accuracy_test["disc"]
        fin_vol = setup_accuracy_test["fin_vol"]
        var = setup_accuracy_test["var"]
        LR = setup_accuracy_test["LR"]
        n_lr = setup_accuracy_test["n_lr"]
        n_tb = setup_accuracy_test["n_tb"]

        # Set BCs for u = x (linear in x)
        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Neumann"),
                "bottom": (pybamm.Scalar(0), "Neumann"),
            }
        }

        # u = x values at nodes
        linear_x = LR.flatten()

        # Compute gradient
        grad_var = pybamm.grad(var)
        disc_grad = disc.process_symbol(grad_var)

        # Verify gradient is [1, 0]
        np.testing.assert_array_almost_equal(
            disc_grad.lr_field.evaluate(None, linear_x).flatten(),
            np.ones((n_lr + 1) * n_tb),
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            disc_grad.tb_field.evaluate(None, linear_x).flatten(),
            np.zeros(n_lr * (n_tb + 1)),
            decimal=5,
        )

        # Compute tensor product
        disc_tp = fin_vol._tensor_product(grad_var, grad_var, disc_grad, disc_grad)

        # Tensor product should be [[1, 0], [0, 0]] after edge-to-node conversion
        # T[0,0] = lr * lr = 1 * 1 = 1
        # T[0,1] = lr * tb = 1 * 0 = 0
        # T[1,0] = tb * lr = 0 * 1 = 0
        # T[1,1] = tb * tb = 0 * 0 = 0
        np.testing.assert_array_almost_equal(
            disc_tp[0, 0].evaluate(None, linear_x).flatten(),
            np.ones(n_lr * n_tb),
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            disc_tp[0, 1].evaluate(None, linear_x).flatten(),
            np.zeros(n_lr * n_tb),
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            disc_tp[1, 0].evaluate(None, linear_x).flatten(),
            np.zeros(n_lr * n_tb),
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            disc_tp[1, 1].evaluate(None, linear_x).flatten(),
            np.zeros(n_lr * n_tb),
            decimal=5,
        )

    def test_tensor_divergence_of_constant_tensor(self, setup_accuracy_test):
        """Test divergence of constant tensor is zero.

        For a constant tensor T = [[1, 0], [0, 1]]:
        div(T)_x = dT_xx/dx + dT_xy/dy = 0 + 0 = 0
        div(T)_y = dT_yx/dx + dT_yy/dy = 0 + 0 = 0
        """
        disc = setup_accuracy_test["disc"]
        fin_vol = setup_accuracy_test["fin_vol"]
        var = setup_accuracy_test["var"]
        n_lr = setup_accuracy_test["n_lr"]
        n_tb = setup_accuracy_test["n_tb"]
        LR = setup_accuracy_test["LR"]

        # Create a constant tensor [[1, 0], [0, 1]]
        one = pybamm.PrimaryBroadcast(pybamm.Scalar(1), var.domain)
        zero = pybamm.PrimaryBroadcast(pybamm.Scalar(0), var.domain)

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(0), "Dirichlet"),
                "bottom": (pybamm.Scalar(1), "Dirichlet"),
            }
        }

        # Discretise components
        disc_one = disc.process_symbol(one)
        disc_zero = disc.process_symbol(zero)

        # Create rank-2 tensor
        disc_tensor = TensorField(
            [[disc_one, disc_zero], [disc_zero, disc_one]],
            domain=var.domain,
        )

        # Apply tensor divergence using mock gradient symbol
        grad_var = pybamm.grad(var)
        div_result = fin_vol.divergence(grad_var, disc_tensor, disc.bcs)

        # Result should be VectorField with both components zero
        assert isinstance(div_result, pybamm.VectorField)

        y_test = LR.flatten()
        np.testing.assert_allclose(
            div_result.lr_field.evaluate(None, y_test).flatten(),
            np.zeros(n_lr * n_tb),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            div_result.tb_field.evaluate(None, y_test).flatten(),
            np.zeros(n_lr * n_tb),
            atol=1e-10,
        )

    def test_tensor_divergence_of_linear_tensor(self, setup_accuracy_test):
        """Test divergence of tensor with linear components.

        For tensor T = [[x, 0], [0, y]]:
        div(T)_x = dT_xx/dx + dT_xy/dy = 1 + 0 = 1
        div(T)_y = dT_yx/dx + dT_yy/dy = 0 + 1 = 1

        So div(T) = [1, 1] (approximately, with boundary effects)
        """
        disc = setup_accuracy_test["disc"]
        fin_vol = setup_accuracy_test["fin_vol"]
        var = setup_accuracy_test["var"]
        n_lr = setup_accuracy_test["n_lr"]
        n_tb = setup_accuracy_test["n_tb"]
        LR = setup_accuracy_test["LR"]

        # Create tensor [[x, 0], [0, y]] using spatial variables
        x_var = pybamm.SpatialVariable("x", var.domain, direction="lr")
        y_var = pybamm.SpatialVariable("y", var.domain, direction="tb")
        zero = pybamm.PrimaryBroadcast(pybamm.Scalar(0), var.domain)

        disc.bcs = {
            var: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "top": (pybamm.Scalar(1), "Dirichlet"),
                "bottom": (pybamm.Scalar(0), "Dirichlet"),
            }
        }

        # Discretise components
        disc_x = disc.process_symbol(x_var)
        disc_y = disc.process_symbol(y_var)
        disc_zero = disc.process_symbol(zero)

        # Create rank-2 tensor [[x, 0], [0, y]]
        disc_tensor = TensorField(
            [[disc_x, disc_zero], [disc_zero, disc_y]],
            domain=var.domain,
        )

        # Apply tensor divergence
        grad_var = pybamm.grad(var)
        div_result = fin_vol.divergence(grad_var, disc_tensor, disc.bcs)

        # Result should be VectorField with components approximately [1, 1]
        # (interior points; boundaries may differ)
        assert isinstance(div_result, pybamm.VectorField)

        y_test = LR.flatten()
        lr_result = div_result.lr_field.evaluate(None, y_test).flatten()
        tb_result = div_result.tb_field.evaluate(None, y_test).flatten()

        # Check that interior values are close to 1
        # The central difference for d(x)/dx should give 1
        np.testing.assert_allclose(lr_result, np.ones(n_lr * n_tb), rtol=0.1, atol=0.1)
        np.testing.assert_allclose(tb_result, np.ones(n_lr * n_tb), rtol=0.1, atol=0.1)
