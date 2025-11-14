import pybamm


def func_correct(x):
    return 2 * x


def func_incorrect(x, y=1):
    return 2 * x


x = 2
func_param = pybamm.FunctionParameter("func", {"x": x})

parameter_values = pybamm.ParameterValues({"func": func_correct})
print("Original with correct function: ", parameter_values.evaluate(func_param))
# 4

serialized = parameter_values.to_json()
parameter_values = pybamm.ParameterValues.from_json(serialized)
print("Deserialized with correct function: ", parameter_values.evaluate(func_param))
# 4

parameter_values = pybamm.ParameterValues({"func": func_incorrect})
print("Original with incorrect function: ", parameter_values.evaluate(func_param))
# 4

serialized = parameter_values.to_json()
parameter_values = pybamm.ParameterValues.from_json(serialized)
print("Deserialized with incorrect function: ", parameter_values.evaluate(func_param))
# fails
