#
# Parameter classes
#
from __future__ import annotations

import sys
from typing import Literal

import numpy as np
import sympy

import pybamm


class Parameter(pybamm.Symbol):
    """
    A node in the expression tree representing a parameter.

    This node will be replaced by a :class:`pybamm.Scalar` node

    Parameters
    ----------

    name : str
        name of the node
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def create_copy(
        self,
        new_children=None,
        perform_simplifications=True,
    ) -> pybamm.Parameter:
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return Parameter(self.name)

    def _evaluate_for_shape(self) -> float:
        """
        Returns the scalar 'NaN' to represent the shape of a parameter.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return np.nan

    def is_constant(self) -> Literal[False]:
        """See :meth:`pybamm.Symbol.is_constant()`."""
        # Parameter is not constant since it can become an InputParameter
        return False

    def to_equation(self) -> sympy.Symbol:
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            return sympy.Symbol(self.name)

    def to_json(self):
        raise NotImplementedError(
            "pybamm.Parameter: Serialisation is only implemented for discretised models"
        )

    @classmethod
    def _from_json(cls, snippet):
        raise NotImplementedError(
            "pybamm.Parameter: Please use a discretised model when reading in from JSON"
        )


class FunctionParameter(pybamm.Symbol):
    """
    A node in the expression tree representing a function parameter.

    This node will be replaced by a :class:`pybamm.Function` node if a callable function
    is passed to the parameter values, and otherwise (in some rarer cases, such as
    constant current) a :class:`pybamm.Scalar` node.

    Parameters
    ----------

    name : str
        name of the node
    inputs : dict
        A dictionary with string keys and :class:`pybamm.Symbol` values representing
        the function inputs. The string keys should provide a reasonable description
        of what the input to the function is
        (e.g. "Electrolyte concentration [mol.m-3]")
    diff_variable : :class:`pybamm.Symbol`, optional
        if diff_variable is specified, the FunctionParameter node will be replaced by a
        :class:`pybamm.Function` and then differentiated with respect to diff_variable.
        Default is None.
    print_name : str, optional
        The name to show when printing. Default is 'calculate', in which case the name
        is calculated using sys._getframe().
    post_processor : callable, optional
        A callable that takes a pybamm.Symbol and returns a pybamm.Symbol. If provided,
        it will be applied to the function output after the function is evaluated during
        parameter processing. This can be used to apply regularisation or other
        transformations. Default is None.
    """

    def __init__(
        self,
        name: str,
        inputs: dict[str, pybamm.Symbol],
        diff_variable: pybamm.Symbol | None = None,
        print_name="calculate",
        post_processor=None,
    ) -> None:
        # assign diff variable
        self.diff_variable = diff_variable
        self.post_processor = post_processor
        children_list = list(inputs.values())

        # Turn numbers into scalars
        for idx, child in enumerate(children_list):
            if isinstance(child, float | int | np.number):
                children_list[idx] = pybamm.Scalar(child)

        domains = self.get_children_domains(children_list)
        super().__init__(name, children=children_list, domains=domains)

        self.input_names = list(inputs.keys())

        # Use the inspect module to find the function's "short name" from the
        # Parameters module that called it
        if print_name != "calculate":
            self.print_name = print_name
        else:
            frame = sys._getframe().f_back
            if frame is not None:
                print_name = frame.f_code.co_name
                if print_name.startswith("_"):
                    self.print_name = None
                else:
                    try:
                        parent_param = frame.f_locals["self"]
                    except KeyError:
                        parent_param = None
                    if (
                        hasattr(parent_param, "domain")
                        and parent_param.domain is not None
                    ):
                        # add "_n" or "_s" or "_p" if this comes from a Parameter class with
                        # a domain
                        d = parent_param.domain[0]
                        print_name += f"_{d}"
                    self.print_name = print_name

    def print_input_names(self):
        if self._input_names:
            for inp in self._input_names:
                print(inp)

    @property
    def input_names(self):
        return self._input_names

    @input_names.setter
    def input_names(self, inp=None):
        if inp:
            if inp.__class__ is list:
                for i in inp:
                    if i.__class__ is not str:
                        raise TypeError(
                            "Inputs must be a provided as"
                            + "a dictionary of the form:"
                            + "{{str: :class:`pybamm.Symbol`}}"
                        )
            else:
                raise TypeError(
                    "Inputs must be a provided as"
                    + " a dictionary of the form:"
                    + "{{str: :class:`pybamm.Symbol`}}"
                )

        self._input_names = inp

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id`"""
        self._id = hash(
            (
                self.__class__,
                self.name,
                self.diff_variable,
                *tuple([child.id for child in self.children]),
                *tuple(self.domain),
            )
        )

    def diff(self, variable: pybamm.Symbol) -> pybamm.FunctionParameter:
        """See :meth:`pybamm.Symbol.diff()`."""
        # return a new FunctionParameter, that knows it will need to be differentiated
        # when the parameters are set
        children_list = self.orphans
        input_names = self._input_names

        input_dict = {input_names[i]: children_list[i] for i in range(len(input_names))}

        return FunctionParameter(
            self.name,
            input_dict,
            diff_variable=variable,
            print_name=self.print_name + "'",
            post_processor=self.post_processor,
        )

    def create_copy(self, new_children=None, perform_simplifications=True):
        """See :meth:`pybamm.Symbol.new_copy()`."""

        input_dict = {
            self._input_names[i]: self.children[i]
            for i in range(len(self._input_names))
        }

        return FunctionParameter(
            self.name,
            input_dict,
            diff_variable=self.diff_variable,
            print_name=self.print_name,
            post_processor=self.post_processor,
        )

    def _evaluate_for_shape(self):
        """
        Returns the sum of the evaluated children
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        # add 1e-300 to avoid division by zero
        return sum(child.evaluate_for_shape() for child in self.children) + 1e-300

    def to_equation(self) -> sympy.Symbol:
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            return sympy.Symbol(self.name)

    def to_json(self):
        raise NotImplementedError(
            "pybamm.FunctionParameter:"
            "Serialisation is only implemented for discretised models."
        )

    @classmethod
    def _from_json(cls, snippet):
        raise NotImplementedError(
            "pybamm.FunctionParameter:"
            "Please use a discretised model when reading in from JSON."
        )


class NamedFunctionParameter:
    """A parameter value that binds to a model input by name rather than position.

    A plain callable supplied for a :class:`FunctionParameter` is bound to the
    model's inputs *positionally* — the callable's arguments are matched to
    ``FunctionParameter.input_names`` in order, so the caller must know both how
    many inputs the consuming model declares and the order it declares them in.
    Wrapping the callable in a ``NamedFunctionParameter`` and naming the inputs
    it consumes removes that coupling: binding selects and orders children by
    matching ``inputs`` against ``FunctionParameter.input_names``, so the
    declaration order of the model's inputs is irrelevant and unused inputs are
    dropped.

    Parameters
    ----------
    function : callable
        The function to evaluate. It is called with one positional argument per
        entry in ``inputs``, in the order ``inputs`` is given — not the order the
        model declares them.
    inputs : sequence of str
        The model input names this function consumes (e.g.
        ``["Electrolyte concentration [mol.m-3]"]``). Each must match one of the
        consuming ``FunctionParameter``'s ``input_names``.

    Examples
    --------
    .. code-block:: python

        # conductivity depends only on electrolyte concentration; the model calls
        # the parameter with (c_e, T) but we never need to know that, or its order
        kappa = pybamm.NamedFunctionParameter(
            lambda c_e: pybamm.Interpolant(x, y, c_e, name="kappa"),
            inputs=["Electrolyte concentration [mol.m-3]"],
        )
        param = pybamm.ParameterValues({"Electrolyte conductivity [S.m-1]": kappa})
    """

    def __init__(self, function, inputs):
        if not callable(function):
            raise TypeError("NamedFunctionParameter `function` must be callable")
        self.function = function
        self.inputs = list(inputs)

    def bind(self, input_names, children):
        """Select and order ``children`` by matching ``self.inputs`` against the
        consuming ``FunctionParameter``'s ``input_names``, then evaluate.

        Parameters
        ----------
        input_names : list of str
            The consuming ``FunctionParameter``'s declared input names.
        children : list of pybamm.Symbol
            The processed children, in the same order as ``input_names``.

        Returns
        -------
        pybamm.Symbol
            ``self.function`` evaluated on the named subset of children.
        """
        name_to_child = dict(zip(input_names, children, strict=False))
        missing = [name for name in self.inputs if name not in name_to_child]
        if missing:
            raise KeyError(
                f"NamedFunctionParameter inputs {missing} not found among the "
                f"consuming FunctionParameter's input_names {list(input_names)}"
            )
        return self.function(*[name_to_child[name] for name in self.inputs])
