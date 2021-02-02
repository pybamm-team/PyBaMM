import pybamm
import copy

a = pybamm.Symbol("a")
b = pybamm.Symbol("b")
c = pybamm.Symbol("c")

print(a + b)
print(a + c)

print(a + copy.copy((a + c).right))
