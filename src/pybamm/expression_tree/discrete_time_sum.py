import pybamm
import numpy.typing as npt


class DiscreteTimeData(pybamm.Interpolant):
    """
    A class for representing data that is only defined at discrete points in time.
    This is implemented as a 1D interpolant with the time points as the nodes.

    Parameters
    ----------

    time_points : :class:`numpy.ndarray`
        The time points at which the data is defined
    data : :class:`numpy.ndarray`
        The data to be interpolated
    name : str
        The name of the data

    """

    def __init__(self, time_points: npt.NDArray, data: npt.NDArray, name: str):
        super().__init__(time_points, data, pybamm.t, name)

    def create_copy(self, new_children=None, perform_simplifications=True):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return pybamm.DiscreteTimeData(self.x[0], self.y, self.name)


class DiscreteTimeSum(pybamm.UnaryOperator):
    """
    A node in the expression tree representing a discrete time sum operator.

    .. math::
        \\sum_{i=0}^{N} f(y(t_i), t_i)

    where f is the expression given by the child, and the sum is over the discrete
    time points t_i. The set of time points is given by the :class:`pybamm.DiscreteTimeData` node,
    which must be somewhere in the expression tree given by the child. If the child
    does not contain a :class:`pybamm.DiscreteTimeData` node, then an error will be raised when
    the node is created. If the child contains multiple :class:`pybamm.DiscreteTimeData` nodes,
    an error will be raised when the node is created.


    Parameters
    ----------
    child: :class:`pybamm.Symbol`
        The symbol to be summed

    Attributes
    ----------
    data: :class:`pybamm.DiscreteTimeData`
        The discrete time data node in the child

    Raises
    ------
    :class:`pybamm.ModelError`
        If the child does not contain a :class:`pybamm.DiscreteTimeData` node, or if the child
        contains multiple :class:`pybamm.DiscreteTimeData` nodes.
    """

    def __init__(self, child: pybamm.Symbol):
        self.data = None
        for node in child.pre_order():
            if isinstance(node, DiscreteTimeData):
                # Check that there is exactly one DiscreteTimeData node in the child
                if self.data is not None:
                    raise pybamm.ModelError(
                        "DiscreteTimeSum can only have one DiscreteTimeData node in the child"
                    )
                self.data = node
        if self.data is None:
            raise pybamm.ModelError(
                "DiscreteTimeSum must contain a DiscreteTimeData node"
            )
        super().__init__("discrete time sum", child)

    @property
    def sum_values(self):
        return self.data.y

    @property
    def sum_times(self):
        return self.data.x[0]

    def _unary_evaluate(self, child):
        # return result of evaluating the child, we'll only implement the sum once the model is solved (in pybamm.ProcessedVariable)
        return child
