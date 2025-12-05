import pybamm
from pybamm.expression_tree.operations.replace_symbols import SymbolReplacer, VariableReplacementMap

def test_symbol_replacements():
    a = pybamm.Parameter("a")
    b = pybamm.Parameter("b")
    c = pybamm.Parameter("c")
    d = pybamm.Parameter("d")
    replacer = SymbolReplacer({a: b, c: d})

    for symbol_in, symbol_out in [
        (a, b),  # just the symbol
        (a + a, b + b),  # binary operator
        (2 * pybamm.sin(a), 2 * pybamm.sin(b)),  # function
        (3 * b, 3 * b),  # no replacement
        (a + c, b + d),  # two replacements
    ]:
        replaced_symbol = replacer.process_symbol(symbol_in)
        assert replaced_symbol == symbol_out

    var1 = pybamm.Variable("var 1", domain="dom 1")
    var2 = pybamm.Variable("var 2", domain="dom 2")
    var3 = pybamm.Variable("var 3", domain="dom 1")
    conc = pybamm.concatenation(var1, var2)

    replacer = SymbolReplacer({var1: var3})
    replaced_symbol = replacer.process_symbol(conc)
    assert replaced_symbol == pybamm.concatenation(var3, var2)


def test_process_model():
    model = pybamm.BaseModel()
    a = pybamm.Parameter("a")
    b = pybamm.Parameter("b")
    c = pybamm.Parameter("c")
    d = pybamm.Parameter("d")
    var1 = pybamm.Variable("var1", domain="test")
    var2 = pybamm.Variable("var2", domain="test")
    model.rhs = {var1: a * pybamm.grad(var1)}
    model.algebraic = {var2: c * var2}
    model.initial_conditions = {var1: b, var2: d}
    model.boundary_conditions = {
        var1: {"left": (c, "Dirichlet"), "right": (d, "Neumann")}
    }
    model.variables = {
        "var1": var1,
        "var2": var2,
        "grad_var1": pybamm.grad(var1),
        "d_var1": d * var1,
    }

    replacer = SymbolReplacer(
        {
            pybamm.Parameter("a"): pybamm.Scalar(4),
            pybamm.Parameter("b"): pybamm.Scalar(2),
            pybamm.Parameter("c"): pybamm.Scalar(3),
            pybamm.Parameter("d"): pybamm.Scalar(42),
        }
    )
    replacer.process_model(model)
    # rhs
    var1 = model.variables["var1"]
    assert isinstance(model.rhs[var1], pybamm.Multiplication)
    assert isinstance(model.rhs[var1].children[0], pybamm.Scalar)
    assert isinstance(model.rhs[var1].children[1], pybamm.Gradient)
    assert model.rhs[var1].children[0].value == 4
    # algebraic
    var2 = model.variables["var2"]
    assert isinstance(model.algebraic[var2], pybamm.Multiplication)
    assert isinstance(model.algebraic[var2].children[0], pybamm.Scalar)
    assert isinstance(model.algebraic[var2].children[1], pybamm.Variable)
    assert model.algebraic[var2].children[0].value == 3
    # initial conditions
    assert isinstance(model.initial_conditions[var1], pybamm.Scalar)
    assert model.initial_conditions[var1].value == 2
    # boundary conditions
    bc_key = list(model.boundary_conditions.keys())[0]
    assert isinstance(bc_key, pybamm.Variable)
    bc_value = list(model.boundary_conditions.values())[0]
    assert isinstance(bc_value["left"][0], pybamm.Scalar)
    assert bc_value["left"][0].value == 3
    assert isinstance(bc_value["right"][0], pybamm.Scalar)
    assert bc_value["right"][0].value == 42
    # variables
    assert model.variables["var1"] == var1
    assert isinstance(model.variables["grad_var1"], pybamm.Gradient)
    assert isinstance(model.variables["grad_var1"].children[0], pybamm.Variable)
    assert model.variables["d_var1"] == (pybamm.Scalar(42) * var1)
    assert isinstance(model.variables["d_var1"].children[0], pybamm.Scalar)
    assert isinstance(model.variables["d_var1"].children[1], pybamm.Variable)


    assert replacement_map.get(pybamm.Scalar(1)) is None