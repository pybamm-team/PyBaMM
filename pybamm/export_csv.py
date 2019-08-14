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

        self.output_times = np.asarray([])
        self.output_x_locs = np.asarray([])

        self.stage = None
        self.column_names = []

        self.exported_files = []

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

    def set_output_points(self, times, x_locs=None, r_locs=None):

        self.times = times
        self.x_locs = x_locs
        self.r_locs = r_locs

        if x_locs:
            x_size = self.x_locs.size
        else:
            x_size = 0

        if r_locs:
            r_size = self.r_locs.size
        else:
            r_size = 0

        # check output is one-dimensional
        if sum([self.times.size > 1, x_size > 1, r_size > 1]) > 1:
            raise NotImplementedError("Can only output 1D variables at the moment")

    def add(self, variables):
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

            processed_variables = pybamm.post_process_variables(
                variables_to_process,
                self.solutions[i].t,
                self.solutions[i].y,
                self.mesh,
            )

            for var_name, var in processed_variables.items():

                evaluated_variable = var(self.times, x=self.x_locs, r=self.r_locs)

                self.add_array(evaluated_variable)

                self.column_names = np.append(self.column_names, var_name)

    def add_array(self, array):

        if self.stage is None:
            self.stage = array
        else:
            self.stage = np.column_stack((self.stage, array))

    def add_error(self, variables, truth_index=0):

        # process truth
        truth_variables_to_process = {}
        truth_variables_to_process.update(
            {var: self.models[truth_index].variables[var] for var in variables}
        )
        truth_processed_variables = pybamm.post_process_variables(
            truth_variables_to_process,
            self.solutions[truth_index].t,
            self.solutions[truth_index].y,
            self.mesh,
        )

        for i, model in enumerate(self.models):

            if i != truth_index:
                variables_to_process = {}

                variables_to_process.update(
                    {var: model.variables[var] for var in variables}
                )

                processed_variables = pybamm.post_process_variables(
                    variables_to_process,
                    self.solutions[i].t,
                    self.solutions[i].y,
                    self.mesh,
                )
                for var_name, var in processed_variables.items():

                    error = np.abs(
                        var(self.times, x=self.x_locs, r=self.r_locs)
                        - truth_processed_variables[var_name](
                            self.times, x=self.x_locs, r=self.r_locs
                        )
                    )

                    if self.stage is None:
                        self.stage = error
                    else:
                        self.stage = np.column_stack((self.stage, error))

                    self.column_names = np.append(self.column_names, var_name)

    def reset_stage(self):
        self.stage = None
        self.column_names = []

    def export(self, filename):

        if not filename[-4:] in [".csv", ".dat"]:
            filename += ".csv"

        if not self.directory_path[-1] == "/":
            self.directory_path += "/"

        export_path = self.directory_path + filename
        np.savetxt(export_path, self.stage, delimiter=",")

        self.exported_files = np.append(self.exported_files, export_path)

    def delete_exported(self, to_delete):

        if to_delete == "all":
            files_to_delete = self.exported_files
        elif to_delete == "last":
            files_to_delete = self.exported_files[-1]
        else:
            raise KeyError(
                "Please state whether you wish to delete either 'all' or 'last' exported files"
            )

        for file in files_to_delete:
            os.remove(file)

