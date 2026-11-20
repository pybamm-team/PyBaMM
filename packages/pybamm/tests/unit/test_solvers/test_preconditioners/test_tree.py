#
# Tests for preconditioner block solvers, model defaults and compiled trees
#
import dataclasses
import json

import numpy as np
import pytest
from pybammsolvers import idaklu
from scipy.sparse import csc_matrix

import pybamm

P = pybamm.preconditioners
VAR_PTS = {"x_n": 5, "x_s": 4, "x_p": 6, "r_n": 7, "r_p": 8}
C_N = "Negative particle concentration [mol.m-3]"
C_P = "Positive particle concentration [mol.m-3]"
PHI_E = "Electrolyte potential [V]"
C_E = "Porosity times concentration [mol.m-3]"


@dataclasses.dataclass(frozen=True)
class ExampleJacobi(P.BlockSolver):
    """The example plugin in pybammsolvers, as a block solver."""

    label: str = "example"

    def specification(self):
        return {
            "solver": "external",
            "capsule": idaklu.example_block_solver(),
            "options": json.dumps({"label": self.label}),
        }


def build_dfn(preconditioner=None):
    sim = pybamm.Simulation(
        pybamm.lithium_ion.DFN(), var_pts=VAR_PTS, preconditioner=preconditioner
    )
    sim.build()
    return sim.built_model, csc_matrix(sim.built_model.jacobian_sparsity())


class TestBlockSolvers:
    def test_equality_includes_settings(self):
        assert P.KLU() == P.KLU(stage=None)
        assert P.Tridiagonal(stage=0) != P.Tridiagonal(stage=1)
        assert P.ILU(fill=1) != P.ILU()
        assert P.KLU(stage=0).stage_index == 0
        assert P.KLU(stage=P.Stage(2)).stage_index == 2

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: P.KLU(stage=1.5), r"stage must be an integer"),
            (lambda: P.Jacobi(stage=True), r"stage must be an integer"),
            (lambda: P.ILU(fill=-1), r"must be non-negative"),
            (lambda: P.ILU(fill=0.5), r"fill level must be an integer"),
            (lambda: P.Stage(0, composition="sideways"), r"Unknown stage composition"),
            (lambda: P.Stage(0, schur_complement="rough"), r"Unknown Schur complement"),
        ],
    )
    def test_invalid(self, make, match):
        with pytest.raises(pybamm.OptionError, match=match):
            make()

    @pytest.mark.parametrize(
        "solver",
        [
            P.KLU(),
            P.ILU(fill=2, stage=1),
            P.Jacobi(stage=P.Stage(0, composition="multiplicative")),
            P.Tridiagonal(stage=P.Stage(0, schur_complement="diagonal")),
            ExampleJacobi(label="custom", stage=3),
        ],
    )
    def test_config_round_trip(self, solver):
        config = json.loads(json.dumps(solver.to_config()))
        assert P.BlockSolver.from_config(config) == solver

    def test_from_config_needs_a_block_solver(self):
        with pytest.raises(pybamm.OptionError, match=r"not a block solver"):
            P.BlockSolver.from_config({"a": 1})

    def test_specification(self):
        assert P.KLU().specification() == {"solver": "klu"}
        assert P.ILU(fill=2).specification() == {"solver": "ilu", "fill": 2}
        assert P.Jacobi().specification() == {"solver": "jacobi"}
        assert P.Tridiagonal().specification() == {"solver": "tridiagonal"}
        assert ExampleJacobi().specification()["solver"] == "external"
        with pytest.raises(NotImplementedError, match="does not define"):
            P.BlockSolver().specification()


class DFNWithPreconditioner(pybamm.lithium_ion.DFN):
    @property
    def default_preconditioner(self):
        return {C_N: P.Tridiagonal(stage=0), C_P: P.Tridiagonal(stage=0)}


class TestModelPreconditioner:
    def test_default_is_empty(self):
        assert pybamm.BaseModel().default_preconditioner == {}
        assert pybamm.lithium_ion.SPM().default_preconditioner == {}

    def test_simulation_uses_default_or_override(self):
        sim = pybamm.Simulation(DFNWithPreconditioner(), var_pts=VAR_PTS)
        assert sim.preconditioner == DFNWithPreconditioner().default_preconditioner
        sim.build()
        assert isinstance(sim.built_model.preconditioner, P.Schur)

        override = {PHI_E: P.ILU()}
        sim = pybamm.Simulation(
            DFNWithPreconditioner(), var_pts=VAR_PTS, preconditioner=override
        )
        assert sim.preconditioner == override
        sim.build()
        assert isinstance(sim.built_model.preconditioner, P.Additive)

    def test_discretisation_compiles_it(self):
        model = pybamm.BaseModel()
        a, b = pybamm.Variable("a"), pybamm.Variable("b")
        model.rhs = {a: -a, b: -b}
        model.initial_conditions = {a: 1, b: 1}
        pybamm.Discretisation(preconditioner={"b": P.Jacobi(stage=0)}).process_model(
            model
        )
        tree = model.preconditioner
        assert isinstance(tree, P.Schur)
        assert tree.eliminated.variables == ("b",)
        assert tree.complement.solver == P.KLU()

        stateless = pybamm.BaseModel()
        stateless.y_slices = {}
        assert P.compile_tree(stateless, {}) is None

    def test_serialised_with_the_model(self, tmp_path):
        from pybamm.expression_tree.operations.serialise import Serialise

        preconditioner = {
            "a": P.Tridiagonal(stage=P.Stage(0, schur_complement="diagonal")),
            "b": ExampleJacobi(label="custom"),
        }

        class ModelWithPreconditioner(pybamm.BaseModel):
            @property
            def default_preconditioner(self):
                return preconditioner

        model = ModelWithPreconditioner(name="model")
        a, b = pybamm.Variable("a"), pybamm.Variable("b")
        model.rhs = {a: -a, b: -b}
        model.initial_conditions = {a: pybamm.Scalar(1), b: pybamm.Scalar(1)}
        file_path = tmp_path / "model.json"
        Serialise.save_custom_model(model, filename=str(file_path))

        # Load as a bare BaseModel, so the default comes from the saved recipe
        data = json.loads(file_path.read_text())
        data["model"]["base_class"] = "nonexistent_module.Model"
        data["model"]["base_class_mro"] = ["nonexistent_module.Model"]
        file_path.write_text(json.dumps(data))
        with pytest.warns(UserWarning, match=r"Falling back to pybamm\.BaseModel"):
            loaded = Serialise.load_custom_model(str(file_path))
        assert loaded.default_preconditioner == preconditioner


class TestCompileTree:
    def test_config_round_trip(self):
        last = P.Stage(None, composition="multiplicative")
        model, _ = build_dfn(
            {
                C_N: P.Tridiagonal(stage=P.Stage(0, schur_complement="diagonal")),
                C_P: P.Tridiagonal(stage=0),
                PHI_E: P.ILU(fill=1, stage=last),
                C_E: ExampleJacobi(stage=last),
            }
        )
        tree = model.preconditioner
        restored = P.tree_from_config(json.loads(json.dumps(P.tree_to_config(tree))))
        assert P.describe(restored) == P.describe(tree)

        def leaves(node):
            if isinstance(node, P.Leaf):
                return [node]
            if isinstance(node, P.Schur):
                return leaves(node.eliminated) + leaves(node.complement)
            return [leaf for child in node.children for leaf in leaves(child)]

        for original, copy in zip(leaves(tree), leaves(restored), strict=True):
            np.testing.assert_array_equal(copy.indices, original.indices)
            assert copy.solver == original.solver
            assert copy.variables == original.variables

    def test_config_with_split_and_empty_blocks(self):
        leaf = P.Leaf(np.array([0, 1, 2, 7, 8, 11]), P.KLU(), ("a",))
        config = P.tree_to_config(leaf)
        assert config["runs"] == [[0, 3], [7, 9], [11, 12]]
        np.testing.assert_array_equal(P.tree_from_config(config).indices, leaf.indices)
        empty = P.Leaf(np.zeros(0, dtype=np.int64), P.KLU(), ())
        assert len(P.tree_from_config(P.tree_to_config(empty)).indices) == 0
        with pytest.raises(pybamm.OptionError, match="Unknown preconditioner"):
            P.tree_from_config({"type": "unknown"})

    def test_default_is_one_exact_klu(self):
        model, pattern = build_dfn()
        tree = model.preconditioner
        assert isinstance(tree, P.Leaf)
        assert tree.solver == P.KLU()
        assert len(tree.indices) == model.len_rhs_and_alg
        assert P.analyse(tree, pattern).exact

    def test_stages_nest_and_solvers_group(self):
        model, pattern = build_dfn(
            {C_N: P.Tridiagonal(stage=0), C_P: P.Tridiagonal(stage=1), PHI_E: P.ILU()}
        )
        tree = model.preconditioner
        assert isinstance(tree, P.Schur)
        assert tree.eliminated.variables == (C_N,)
        assert isinstance(tree.complement, P.Schur)
        assert tree.complement.eliminated.variables == (C_P,)
        last = tree.complement.complement
        assert isinstance(last, P.Additive)
        assert [child.solver for child in last.children] == [P.ILU(), P.KLU()]
        assert not P.analyse(tree, pattern).exact
        text = P.describe(tree)
        assert "schur (exact complement)" in text
        assert "additive" in text

    def test_stage_settings(self):
        last = P.Stage(None, composition="multiplicative")
        model, _ = build_dfn(
            {
                C_N: P.Tridiagonal(stage=P.Stage(0, schur_complement="diagonal")),
                C_P: P.Tridiagonal(stage=0),
                PHI_E: P.ILU(stage=last),
                C_E: P.KLU(stage=last),
            }
        )
        tree = model.preconditioner
        assert tree.schur_complement == "diagonal"
        assert tree.eliminated.variables == (C_N, C_P)
        assert isinstance(tree.complement, P.Multiplicative)
        assert "multiplicative" in P.describe(tree)

    def test_block_order_follows_entries(self):
        last = P.Stage(None, composition="multiplicative")
        model, _ = build_dfn({PHI_E: P.ILU(stage=last), C_E: P.Jacobi(stage=last)})
        tree = model.preconditioner
        assert [child.variables[0] for child in tree.children[:2]] == [PHI_E, C_E]
        # Variables without an entry come last, in state order
        assert tree.children[-1].solver == P.KLU()

    def test_exact_elimination(self):
        model, pattern = build_dfn(
            {C_N: P.Tridiagonal(stage=0), C_P: P.Tridiagonal(stage=0)}
        )
        assert P.analyse(model.preconditioner, pattern).exact

    def test_errors(self):
        with pytest.raises(pybamm.ModelError, match=r"once its states"):
            P.compile_tree(pybamm.lithium_ion.SPM(), {})
        with pytest.raises(pybamm.ModelError, match=r"Close matches: \['Electrolyte"):
            build_dfn({"Electrolyte potentail [V]": P.KLU()})
        with pytest.raises(pybamm.ModelError, match=r"conflicting settings"):
            build_dfn(
                {
                    C_N: P.Tridiagonal(stage=P.Stage(0, composition="multiplicative")),
                    C_P: P.Tridiagonal(stage=P.Stage(0)),
                }
            )
