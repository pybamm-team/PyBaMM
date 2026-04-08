from __future__ import annotations

import numbers

import casadi
import numpy as np
import numpy.typing as npt
import scipy.sparse
import sympy

import pybamm


class Conditional(pybamm.Symbol):
    """
    A node in the expression tree representing a branch selection on a scalar selector.

    The first child is the selector ``s``. Branch ``i`` is active when
    ``i - 0.5 < s < i + 0.5`` using 1-based indexing. If no branch matches, the
    expression evaluates to zero with the same shape as the branch children.
    """

    def __init__(self, selector: pybamm.Symbol, *branches: pybamm.Symbol):
        selector = pybamm.convert_to_symbol(selector)
        branches = [pybamm.convert_to_symbol(branch) for branch in branches]
        if not branches:
            raise ValueError("Conditional requires at least one branch child")
        if selector.shape_for_testing != ():
            raise ValueError("Conditional selector must evaluate to a scalar")

        domains = self.get_children_domains(branches)
        first_shape = branches[0].shape_for_testing
        first_size = branches[0].size_for_testing
        returns_scalar = first_size == 1
        for branch in branches[1:]:
            compatible_shape = (
                branch.size_for_testing == 1
                if returns_scalar
                else branch.shape_for_testing == first_shape
            )
            if not compatible_shape:
                raise ValueError(
                    "Conditional branches must all have the same shape, not "
                    f"{first_shape} and {branch.shape_for_testing}"
                )

        self._returns_scalar = returns_scalar
        super().__init__("conditional", children=[selector, *branches], domains=domains)

    @classmethod
    def _from_json(cls, snippet: dict):
        return cls(*snippet["children"])

    @property
    def selector(self):
        return self.children[0]

    @property
    def branches(self):
        return self.children[1:]

    def __str__(self):
        children = ", ".join(str(child) for child in self.children)
        return f"conditional({children})"

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        children = self._children_for_copying(new_children)
        if len(children) < 2:
            raise ValueError("Conditional must have a selector and at least one branch")
        return self.__class__(*children)

    @staticmethod
    def _coerce_selector_value(value):
        if isinstance(value, numbers.Number):
            return float(value)

        value = np.asarray(value)
        if value.size != 1:
            raise ValueError("Conditional selector must evaluate to a scalar")
        return float(value.reshape(-1)[0])

    def _active_branch_index(
        self,
        t: float | None = None,
        y: npt.NDArray[np.float64] | None = None,
        y_dot: npt.NDArray[np.float64] | None = None,
        inputs: dict | str | None = None,
    ):
        selector_value = self._coerce_selector_value(
            self.selector.evaluate(t, y, y_dot, inputs)
        )
        for branch_index, _branch in enumerate(self.branches, start=1):
            if branch_index - 0.5 < selector_value < branch_index + 0.5:
                return branch_index - 1
        return None

    @staticmethod
    def _zero_like(template):
        if isinstance(template, numbers.Number):
            return 0
        if scipy.sparse.issparse(template):
            return scipy.sparse.csr_matrix(template.shape)
        return np.zeros_like(template)

    def evaluate(
        self,
        t: float | None = None,
        y: npt.NDArray[np.float64] | None = None,
        y_dot: npt.NDArray[np.float64] | None = None,
        inputs: dict | str | None = None,
    ):
        branch_index = self._active_branch_index(t, y, y_dot, inputs)
        if branch_index is None:
            if self._returns_scalar:
                return 0
            return self._zero_like(self.branches[0].evaluate_for_shape())
        value = self.branches[branch_index].evaluate(t, y, y_dot, inputs)
        if self._returns_scalar and not isinstance(value, numbers.Number):
            return self._coerce_selector_value(value)
        return value

    def _evaluate_for_shape(self):
        if self._returns_scalar:
            return np.nan
        return self.branches[0].evaluate_for_shape()

    def _evaluates_on_edges(self, dimension: str) -> bool:
        return any(branch.evaluates_on_edges(dimension) for branch in self.branches)

    def is_constant(self):
        return all(child.is_constant() for child in self.children)

    def _diff(self, variable):
        return Conditional(
            self.selector,
            *[branch.diff(variable) for branch in self.branches],
        )

    def _jac(self, variable):
        return Conditional(
            self.selector,
            *[branch.jac(variable) for branch in self.branches],
        )

    def _to_casadi(self, t, y, y_dot, inputs, casadi_symbols):
        """See :meth:`pybamm.Symbol._to_casadi()`."""
        converted_selector = self.selector._to_casadi_inner(
            t, y, y_dot, inputs, casadi_symbols
        )
        first_branch = self.branches[0]._to_casadi_inner(
            t, y, y_dot, inputs, casadi_symbols
        )
        result = casadi.MX.zeros(*first_branch.shape)
        for branch_index in range(len(self.branches), 0, -1):
            if branch_index == 1:
                converted_branch = first_branch
            else:
                converted_branch = self.branches[branch_index - 1]._to_casadi_inner(
                    t, y, y_dot, inputs, casadi_symbols
                )
            condition = casadi.logic_and(
                converted_selector > (branch_index - 0.5),
                converted_selector < (branch_index + 0.5),
            )
            result = casadi.if_else(condition, converted_branch, result)
        return result

    def to_equation(self):
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        return sympy.Function("Conditional")(
            *[child.to_equation() for child in self.children]
        )
