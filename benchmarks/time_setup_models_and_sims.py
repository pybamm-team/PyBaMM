import pybamm


class TimeBuildSPM:
    def __init__(self):
        self.param = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Marquis2019
        )

    def compute_discretisation(self):
        var_pts = {
            pybamm.standard_spatial_vars.x_n: 20,
            pybamm.standard_spatial_vars.x_s: 20,
            pybamm.standard_spatial_vars.x_p: 20,
            pybamm.standard_spatial_vars.r_n: 30,
            pybamm.standard_spatial_vars.r_p: 30,
            pybamm.standard_spatial_vars.y: 10,
            pybamm.standard_spatial_vars.z: 10,
        }
        geometry = self.model.default_geometry
        self.param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry,
                           self.model.default_submesh_types,
                           var_pts)
        return pybamm.Discretisation(mesh,
                                     self.model.default_spatial_methods)

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        self.compute_discretisation().process_model(self.model)


class TimeBuildSPMe:
    def __init__(self):
        self.param = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Marquis2019
        )

    def compute_discretisation(self):
        var_pts = {
            pybamm.standard_spatial_vars.x_n: 20,
            pybamm.standard_spatial_vars.x_s: 20,
            pybamm.standard_spatial_vars.x_p: 20,
            pybamm.standard_spatial_vars.r_n: 30,
            pybamm.standard_spatial_vars.r_p: 30,
            pybamm.standard_spatial_vars.y: 10,
            pybamm.standard_spatial_vars.z: 10,
        }
        geometry = self.model.default_geometry
        self.param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry,
                           self.model.default_submesh_types,
                           var_pts)
        return pybamm.Discretisation(mesh,
                                     self.model.default_spatial_methods)

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        self.compute_discretisation().process_model(self.model)


class TimeBuildDFN:
    def __init__(self):
        self.param = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Marquis2019
        )

    def compute_discretisation(self):
        var_pts = {
            pybamm.standard_spatial_vars.x_n: 20,
            pybamm.standard_spatial_vars.x_s: 20,
            pybamm.standard_spatial_vars.x_p: 20,
            pybamm.standard_spatial_vars.r_n: 30,
            pybamm.standard_spatial_vars.r_p: 30,
            pybamm.standard_spatial_vars.y: 10,
            pybamm.standard_spatial_vars.z: 10,
        }
        geometry = self.model.default_geometry
        self.param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry,
                           self.model.default_submesh_types,
                           var_pts)
        return pybamm.Discretisation(mesh,
                                     self.model.default_spatial_methods)

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        self.compute_discretisation().process_model(self.model)
