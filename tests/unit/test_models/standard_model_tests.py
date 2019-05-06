#
# Standard basic tests for any model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import numpy as np


class StandardModelTest(object):
    """ Basic processing test for the models. """

    def __init__(
        self,
        model,
        parameter_values=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
    ):
        self.model = model
        # Set parameters, geometry, spatial methods etc
        # The code below is equivalent to:
        #    if parameter_values is None:
        #       self.parameter_values = model.default_parameter_values
        #    else:
        #       self.parameter_values = parameter_values
        self.parameter_values = parameter_values or model.default_parameter_values
        geometry = geometry or model.default_geometry
        submesh_types = submesh_types or model.default_submesh_types
        var_pts = var_pts or model.default_var_pts
        spatial_methods = spatial_methods or model.default_spatial_methods
        self.solver = solver or model.default_solver
        # Process geometry
        self.parameter_values.process_geometry(geometry)
        # Set discretisation
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        self.disc = pybamm.Discretisation(mesh, spatial_methods)

    def test_processing_parameters(self, parameter_values=None):
        # Overwrite parameters if given
        if parameter_values is not None:
            self.parameter_values = parameter_values
        self.parameter_values.process_model(self.model)
        # Model should still be well-posed after processing
        self.model.check_well_posedness()
        # No Parameter or FunctionParameter nodes in the model
        for eqn in {**self.model.rhs, **self.model.algebraic}.values():
            if any(
                [
                    isinstance(x, (pybamm.Parameter, pybamm.FunctionParameter))
                    for x in eqn.pre_order()
                ]
            ):
                raise TypeError(
                    "Not all Parameter and FunctionParameter objects processed"
                )

    def test_processing_disc(self, disc=None):
        # Overwrite discretisation if given
        if disc is not None:
            self.disc = disc
        self.disc.process_model(self.model)

        # Model should still be well-posed after processing
        self.model.check_well_posedness(post_discretisation=True)

    def test_solving(self, solver=None, t_eval=None):
        # Overwrite solver if given
        if solver is not None:
            self.solver = solver
        if t_eval is None:
            t_eval = np.linspace(0, 1, 100)

        self.solver.solve(self.model, t_eval)

    def test_outputs(self):
        # run the standard output tests
        std_out_test = tests.StandardOutputTests(
            self.model, self.disc, self.solver, self.parameter_values
        )
        std_out_test.test_all()

    def test_all(self, param=None, disc=None, solver=None, t_eval=None):
        self.model.check_well_posedness()
        self.test_processing_parameters(param)
        self.test_processing_disc(disc)
        self.test_solving(solver, t_eval)

        # cannot test dfn yet, and lead acid composite voltage
        # only test the full models
        if isinstance(
            self.model, (pybamm.LithiumIonBaseModel, pybamm.LeadAcidBaseModel)
        ):
            # cannot test dfn at moment and Composite fails the voltage test
            # annoyingly reaction-diffusion is an instance of LeadAcidBaseModel
            # so need to exclude it here
            if not (
                isinstance(
                    self.model,
                    (
                        pybamm.lithium_ion.DFN,
                        pybamm.lead_acid.Composite,
                        pybamm.lead_acid.CompositeCapacitance,
                        pybamm.lead_acid.NewmanTiedemannCapacitance,
                        pybamm.ReactionDiffusionModel,
                    ),
                )
            ):
                self.test_outputs()

    def test_update_parameters(self, param):
        # check if geometry has changed, throw error if so (need to re-discretise)
        if any(
            [
                length in param.keys()
                and param[length] != self.parameter_values[length]
                for length in [
                    "Negative electrode width",
                    "Separator width",
                    "Positive electrode width",
                ]
            ]
        ):
            raise ValueError(
                "geometry has changed, the orginal model needs to be re-discretised"
            )
        # otherwise update self.param and change the parameters in the discretised model
        self.param = param
        param.update_model(self.model, self.disc)
        # Model should still be well-posed after processing
        self.model.check_well_posedness()


class OptimisationsTest(object):
    """ Test that the optimised models give the same result as the original model. """

    def __init__(self, model, parameter_values=None, disc=None):
        # Set parameter values
        if parameter_values is None:
            parameter_values = model.default_parameter_values
        # Process model and geometry
        parameter_values.process_model(model)
        parameter_values.process_geometry(model.default_geometry)
        geometry = model.default_geometry
        # Set discretisation
        if disc is None:
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        # Discretise model
        disc.process_model(model)

        self.model = model

    def evaluate_model(self, simplify=False, use_known_evals=False):
        result = np.array([])
        for eqn in [self.model.concatenated_rhs, self.model.concatenated_algebraic]:
            if eqn is not None:
                if simplify:
                    eqn = eqn.simplify()

                y = self.model.concatenated_initial_conditions
                if use_known_evals:
                    eqn_eval, known_evals = eqn.evaluate(0, y, known_evals={})
                else:
                    eqn_eval = eqn.evaluate(0, y)
            else:
                eqn_eval = np.array([])

            result = np.concatenate([result, eqn_eval])

        return result
