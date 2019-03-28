import pybamm
import numpy as np
from tests import get_discretisation_for_testing

y = pybamm.StateVector(slice(0, 3))
a = pybamm.Variable('a', domain=[])
t = pybamm.t
equation = 2 * a + t
equation.render()

disc = get_discretisation_for_testing()
disc._y_slices = {a.id: slice(0, 1)}
equation = disc.process_symbol(equation)
equation.render()

diff_wrt_a = equation.diff(a).render()

#diff_wrt_equation = equation.diff(t).simplify()
#diff_wrt_equation.render()
#diff_wrt_equation.evaluate(1, np.array([1,2]))

#diff_y = equation.diff(y).simplify().render()


#D = pybamm.Parameter('D')
#c = pybamm.Variable('c', domain=['negative electrode'])
#dcdt = D * pybamm.div(pybamm.grad(c))
#dcdt.render()
#parameter_values = pybamm.ParameterValues({'D': 2})
#dcdt = parameter_values.process_symbol(dcdt)
#dcdt.render()
#disc = get_discretisation_for_testing()
#disc._y_slices = {c.id: slice(0, 40)}
#dcdt = disc.process_symbol(dcdt)
#dcdt.render()
