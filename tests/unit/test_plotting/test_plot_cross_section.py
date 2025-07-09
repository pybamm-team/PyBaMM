import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import use

import pybamm

use("Agg")


class TestPlotCrossSection:
    @pytest.fixture
    def box_3d_solution(self):
        model = pybamm.lithium_ion.BasicSPM_with_3DThermal(
            options={"cell geometry": "box", "dimensionality": 3}
        )
        params = pybamm.ParameterValues("Chen2020")
        params.update({
            "Current function [A]": 1.0,
            "Cell width [m]": 0.01,
            "Cell height [m]": 0.01,
            "Left face heat transfer coefficient [W.m-2.K-1]": 10,
            "Right face heat transfer coefficient [W.m-2.K-1]": 10,
            "Front face heat transfer coefficient [W.m-2.K-1]": 10,
            "Back face heat transfer coefficient [W.m-2.K-1]": 10,
            "Bottom face heat transfer coefficient [W.m-2.K-1]": 10,
            "Top face heat transfer coefficient [W.m-2.K-1]": 10,
        }, check_already_exists=False)
        
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "x": 3, "y": 3, "z": 3}
        sim = pybamm.Simulation(model, parameter_values=params, var_pts=var_pts)
        return sim.solve([0, 100])

    @pytest.fixture
    def cylindrical_3d_solution(self):
        model = pybamm.lithium_ion.BasicSPM_with_3DThermal(
            options={"cell geometry": "cylindrical", "dimensionality": 3}
        )
        params = pybamm.ParameterValues("NCA_Kim2011")
        params.update({
            "Current function [A]": 1.0,
            "Inner cell radius [m]": 0.005,
            "Outer cell radius [m]": 0.018,
            "Inner radius heat transfer coefficient [W.m-2.K-1]": 10,
            "Outer radius heat transfer coefficient [W.m-2.K-1]": 10,
            "Bottom face heat transfer coefficient [W.m-2.K-1]": 10,
            "Top face heat transfer coefficient [W.m-2.K-1]": 10,
        }, check_already_exists=False)
        
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "r_macro": 5, "y": 3, "z": 3}
        sim = pybamm.Simulation(model, parameter_values=params, var_pts=var_pts)
        return sim.solve([0, 100])

    def test_plot_cross_section_3d_box_geometry(self, box_3d_solution):
        ax = pybamm.plot_cross_section(box_3d_solution, show_plot=False)
        assert ax is not None
        
        ax = pybamm.plot_cross_section(box_3d_solution, t=50, show_plot=False)
        assert ax is not None

    def test_plot_cross_section_3d_cylindrical_geometry(self, cylindrical_3d_solution):
        ax = pybamm.plot_cross_section(cylindrical_3d_solution, plane="rz", show_plot=False)
        assert ax is not None

    def test_plot_cross_section_all_planes(self, box_3d_solution):
        planes = ["xy", "yz", "xz"]
        for plane in planes:
            ax = pybamm.plot_cross_section(box_3d_solution, plane=plane, show_plot=False)
            assert ax is not None

    def test_plot_cross_section_different_positions(self, box_3d_solution):
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        for position in positions:
            ax = pybamm.plot_cross_section(box_3d_solution, position=position, show_plot=False)
            assert ax is not None

    def test_plot_cross_section_custom_variable(self, box_3d_solution):
        ax = pybamm.plot_cross_section(
            box_3d_solution, variable="Cell temperature [K]", show_plot=False
        )
        assert ax is not None

    def test_plot_cross_section_custom_kwargs(self, box_3d_solution):
        ax = pybamm.plot_cross_section(
            box_3d_solution, cmap="viridis", interpolation="nearest", show_plot=False
        )
        assert ax is not None

    def test_plot_cross_section_custom_n_pts(self, box_3d_solution):
        for n_pts in [20, 50, 150]:
            ax = pybamm.plot_cross_section(box_3d_solution, n_pts=n_pts, show_plot=False)
            assert ax is not None

    def test_plot_cross_section_custom_axes(self, box_3d_solution):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax_out = pybamm.plot_cross_section(box_3d_solution, ax=ax, show_plot=False)
        assert ax_out == ax

    def test_plot_cross_section_cylindrical_xy_masking(self, cylindrical_3d_solution):
        ax = pybamm.plot_cross_section(cylindrical_3d_solution, plane="xy", show_plot=False)
        assert ax is not None

    def test_plot_cross_section_errors(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 100])
        
        with pytest.raises(TypeError, match="plot_cross_section requires a solution from a model with 3D"):
            pybamm.plot_cross_section(sol)

    def test_plot_cross_section_invalid_plane(self, box_3d_solution):
        with pytest.raises(ValueError, match="Plane 'invalid' must be one of"):
            pybamm.plot_cross_section(box_3d_solution, plane="invalid", show_plot=False)

    def test_plot_cross_section_levels_kwargs_removed(self, box_3d_solution):
        ax = pybamm.plot_cross_section(box_3d_solution, levels=20, show_plot=False)
        assert ax is not None

    def test_plot_cross_section_default_kwargs(self, box_3d_solution):
        ax = pybamm.plot_cross_section(box_3d_solution, show_plot=False)
        assert ax is not None
        
        images = ax.get_images()
        assert len(images) > 0

    def test_plot_cross_section_time_none_uses_last_timestep(self, box_3d_solution):
        ax1 = pybamm.plot_cross_section(box_3d_solution, t=None, show_plot=False)
        ax2 = pybamm.plot_cross_section(box_3d_solution, t=box_3d_solution.t[-1], show_plot=False)
        
        assert ax1 is not None
        assert ax2 is not None

    def test_plot_cross_section_aspect_ratio_and_colorbar(self, box_3d_solution):
        ax = pybamm.plot_cross_section(box_3d_solution, show_plot=False)
        
        assert ax.get_aspect() == 1.0
        
        fig = ax.get_figure()
        assert len(fig.axes) == 2