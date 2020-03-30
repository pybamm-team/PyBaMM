#
# Parameter classes
#
import numbers
import numpy as np
import pybamm


class Parameter(pybamm.Symbol):
    """A node in the expression tree representing a parameter

    This node will be replaced by a :class:`.Scalar` node by :class`.Parameter`

    Parameters
    ----------

    name : str
        name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list

    """

    def __init__(self, name, domain=[]):
        super().__init__(name, domain=domain)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return Parameter(self.name, self.domain)

    def _evaluate_for_shape(self):
        """
        Returns the scalar 'NaN' to represent the shape of a parameter.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return np.nan


class FunctionParameter(pybamm.Symbol):
    """A node in the expression tree representing a function parameter

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
    """

    def __init__(
        self, name, inputs, diff_variable=None,
    ):
        # assign diff variable
        self.diff_variable = diff_variable
        children_list = list(inputs.values())

        # Turn numbers into scalars
        for idx, child in enumerate(children_list):
            if isinstance(child, numbers.Number):
                children_list[idx] = pybamm.Scalar(child)

        domain = self.get_children_domains(children_list)
        auxiliary_domains = self.get_children_auxiliary_domains(children_list)
        super().__init__(
            name,
            children=children_list,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
        )

        self.input_names = list(inputs.keys())

    @property
    def input_names(self):
        return self._input_names

    def print_input_names(self):
        if self._input_names:
            for inp in self._input_names:
                print(inp)

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
        """See :meth:`pybamm.Symbol.set_id` """
        self._id = hash(
            (self.__class__, self.name, self.diff_variable)
            + tuple([child.id for child in self.children])
            + tuple(self.domain)
        )

    def get_children_domains(self, children_list):
        """Obtains the unique domain of the children. If the
        children have different domains then raise an error"""
        domains = [child.domain for child in children_list if child.domain != []]

        # check that there is one common domain amongst children
        distinct_domains = set(tuple(dom) for dom in domains)

        if len(distinct_domains) > 1:
            raise pybamm.DomainError(
                "Functions can only be applied to variables on the same domain"
            )
        elif len(distinct_domains) == 0:
            domain = []
        else:
            domain = domains[0]

        return domain

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # return a new FunctionParameter, that knows it will need to be differentiated
        # when the parameters are set
        children_list = self.orphans
        input_names = self._input_names

        input_dict = {input_names[i]: children_list[i] for i in range(len(input_names))}

        return FunctionParameter(self.name, input_dict, diff_variable=variable)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return self._function_parameter_new_copy(self._input_names, self.orphans)

    def _function_parameter_new_copy(self, input_names, children):
        """Returns a new copy of the function parameter.

        Inputs
        ------
        input_names : : list
            A list of str of the names of the children/function inputs
        children : : list
            A list of the children of the function

        Returns
        -------
            : :pybamm.FunctionParameter
            A new copy of the function parameter
        """

        input_dict = {input_names[i]: children[i] for i in range(len(input_names))}

        return FunctionParameter(
            self.name, input_dict, diff_variable=self.diff_variable
        )

    def _evaluate_for_shape(self):
        """
        Returns the sum of the evaluated children
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return sum(child.evaluate_for_shape() for child in self.children)
