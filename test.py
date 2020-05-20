import pybamm

param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
D_fun = param["Negative electrode diffusivity [m2.s-1]"]
x = pybamm.linspace(0, 1, 100)
y = pybamm.linspace(-10, 20, 100) + 300
[X, Y] = pybamm.meshgrid(x, y)

D = D_fun(x, 300)
pybamm.plot(x, D, xlabel="x", ylabel="y")

D = D_fun(X, Y)
pybamm.plot2D(X, Y, D, xlabel="x", ylabel="y")
