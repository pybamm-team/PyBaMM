#
# Class to export csv file of data
#

import os
import numpy as np
import pybamm


class ExportCSV(object):
    """
    A class to allow for the easy exportation of solution
    data in the form of a csv file.
    """

    def __init__(self, directory_path):

        self.models = []
        self.meshes = []
        self.solutions = []

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        self.directory_path = directory_path

        self.output_times = None
        self.output_locations = None

        self.stage = np.asarray([])

    def set_model_solutions(self, models, mesh, solutions):

        if isinstance(models, pybamm.BaseModel):
            models = [models]
        elif not isinstance(models, list):
            raise TypeError("'models' must be 'pybamm.BaseModel' or list")

        if isinstance(solutions, pybamm.Solution):
            solutions = [solutions]
        elif not isinstance(solutions, list):
            raise TypeError("'solutions' must be 'pybamm.Solution' or list")

        if len(models) == len(solutions):
            self.num_models = len(models)
        else:
            raise ValueError("must provide the same number of models and solutions")

        self.models = models
        self.mesh = mesh
        self.solutions = solutions

    def set_output_times(self, t):
        self.output_times = t

    def add_to_stage(self, variables):
        """
            Adds a list of variables to the stage for exporting

            Parameters
            ----------
            variables : : list
                A list of variables names (str)
        """

        # Process output variables into a form that can be exported
        processed_variables = {}
        for i, model in enumerate(self.models):
            variables_to_process = {}

            variables_to_process.update(
                {var: model.variables[var] for var in variables}
            )

            processed_variables[model] = pybamm.post_process_variables(
                variables_to_process,
                self.solutions[i].t,
                self.solutions[i].y,
                self.mesh,
            )

    def clear_stage(self):
        self.stage = np.asarray([])

    def export(self, filename):

        if not filename[-4:] in [".csv", ".dat"]:
            filename += ".csv"

        np.savetxt(self.directory_path + filename, self.stage, delimiter=",")

