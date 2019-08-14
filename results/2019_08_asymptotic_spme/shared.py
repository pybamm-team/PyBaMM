import pybamm
import numpy as np


def find_time(desired_discharge_capacity, models):
    """
    This function finds the time at which the discharge capacity is at the desired value

    """
    t, y = models.solutions[1].t, models.solutions[1].y

    discharge_capacity_function = pybamm.ProcessedVariable(
        models.models[1].variables["Discharge capacity [A.h.m-2]"],
        t,
        y,
        models.discretisations[1].mesh,
    )

    def bisection(t_left, t_right):
        t_mid = (t_left + t_right) / 2

        if (
            np.abs(discharge_capacity_function(t_mid) - desired_discharge_capacity)
            < 1e-6
        ):
            return t_mid

        if (discharge_capacity_function(t_mid) - desired_discharge_capacity) * (
            discharge_capacity_function(t_left) - desired_discharge_capacity
        ) < 0:
            return bisection(t_left, t_mid)
        else:
            return bisection(t_mid, t_right)

    time = bisection(0, t[-1])
    return time


class ModelGroup(object):
    def __init__(self, *models):
        self.models = models

        # load defaults
        self._parameters = models[0].default_parameter_values
        self._geometry = [model.default_geometry for model in self.models]
        self._spatial_methods = [model.default_spatial_methods for model in self.models]
        self._submesh_types = [model.default_submesh_types for model in self.models]

        self.discretisations = []

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters=None):
        if parameters:
            self._parameters = parameters

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry=None):
        if geometry:
            self._geometry = geometry

    @property
    def submesh_types(self):
        return self._submesh_types

    @submesh_types.setter
    def submesh_types(self, submesh_types=None):
        if submesh_types:
            self._submesh_types = submesh_types

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods=None):
        if spatial_methods:
            self._spatial_methods = spatial_methods

    def process_parameters(self, param=None):

        self.parameters = param

        for geo in self.geometry:
            self.parameters.process_geometry(geo)

        for model in self.models:
            self.parameters.process_model(model)

    def discretise(self, var_pts):

        self.discretisations = []
        self.meshes = []
        for i, model in enumerate(self.models):
            mesh = pybamm.Mesh(self.geometry[i], self.submesh_types[-1], var_pts)
            self.meshes.append(mesh)

            disc = pybamm.Discretisation(mesh, self.spatial_methods[i])
            disc.process_model(model)
            self.discretisations.append(disc)

    def solve(self, t_eval, parameters=None):

        if parameters:
            self.parameters.update(parameters)
            for i, model in enumerate(self.models):
                self.parameters.update_model(model, self.discretisations[i])

        self.solutions = [
            model.default_solver.solve(model, t_eval) for model in self.models
        ]

    def process_variables(self, variables):

        processed_variables = {}
        for i, model in enumerate(self.models):
            variables_dict = {var: model.variables[var] for var in variables}
            processed_variables[model] = pybamm.post_process_variables(
                variables_dict, self.solutions[i].t, self.solutions[i].y, self.meshes[i]
            )

        return processed_variables

