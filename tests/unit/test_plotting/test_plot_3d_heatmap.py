import matplotlib.pyplot as plt
import pytest
from matplotlib import use

import pybamm

use("Agg")


class TestPlot3DHeatmap:
    @pytest.fixture
    def box_3d_solution(self):
        model = pybamm.lithium_ion.Basic3DThermalSPM(
            options={"cell geometry": "pouch", "dimensionality": 3}
        )
        params = pybamm.ParameterValues("Marquis2019")
        params.update(
            {
                "Current function [A]": 1.0,
                "Cell width [m]": 0.01,
                "Cell height [m]": 0.01,
                "Left face heat transfer coefficient [W.m-2.K-1]": 10,
                "Right face heat transfer coefficient [W.m-2.K-1]": 10,
                "Front face heat transfer coefficient [W.m-2.K-1]": 10,
                "Back face heat transfer coefficient [W.m-2.K-1]": 10,
                "Bottom face heat transfer coefficient [W.m-2.K-1]": 10,
                "Top face heat transfer coefficient [W.m-2.K-1]": 10,
            },
            check_already_exists=False,
        )

        var_pts = {
            "x_n": 5,
            "x_s": 5,
            "x_p": 5,
            "r_n": 5,
            "r_p": 5,
            "x": 3,
            "y": 3,
            "z": 3,
        }
        sim = pybamm.Simulation(model, parameter_values=params, var_pts=var_pts)
        return sim.solve([0, 100])

    @pytest.fixture
    def cylindrical_3d_solution(self):
        model = pybamm.lithium_ion.Basic3DThermalSPM(
            options={"cell geometry": "cylindrical", "dimensionality": 3}
        )
        params = pybamm.ParameterValues("NCA_Kim2011")
        params.update(
            {
                "Current function [A]": 1.0,
                "Inner cell radius [m]": 0.005,
                "Outer cell radius [m]": 0.018,
                "Inner radius heat transfer coefficient [W.m-2.K-1]": 10,
                "Outer radius heat transfer coefficient [W.m-2.K-1]": 10,
                "Bottom face heat transfer coefficient [W.m-2.K-1]": 10,
                "Top face heat transfer coefficient [W.m-2.K-1]": 10,
            },
            check_already_exists=False,
        )

        var_pts = {
            "x_n": 5,
            "x_s": 5,
            "x_p": 5,
            "r_n": 5,
            "r_p": 5,
            "r_macro": 5,
            "y": 3,
            "z": 3,
        }
        sim = pybamm.Simulation(model, parameter_values=params, var_pts=var_pts)
        return sim.solve([0, 100])

    @pytest.fixture
    def non_3d_solution(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        return sim.solve([0, 100])

    def test_plot_3d_heatmap_default(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, show_plot=False, variable="Cell temperature [K]", t=None
        )
        assert ax is not None
        assert ax.name == "3d"

    def test_plot_3d_heatmap_with_time(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, t=50, show_plot=False, variable="Cell temperature [K]"
        )
        assert ax is not None

    def test_plot_3d_heatmap_time_none(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, t=None, show_plot=False, variable="Cell temperature [K]"
        )
        assert ax is not None

    def test_plot_3d_heatmap_cylindrical(self, cylindrical_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            cylindrical_3d_solution,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_heatmap_custom_variable(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, variable="Cell temperature [K]", show_plot=False, t=None
        )
        assert ax is not None

    def test_plot_3d_heatmap_custom_cmap(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution,
            cmap="viridis",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_heatmap_custom_marker_size(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution,
            marker_size=20,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_heatmap_custom_alpha(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution,
            alpha=0.5,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_heatmap_use_offset_true(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution,
            use_offset=True,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_heatmap_use_offset_false(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution,
            use_offset=False,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_heatmap_custom_ax(self, box_3d_solution):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax_out = pybamm.plot_3d_heatmap(
            box_3d_solution,
            ax=ax,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax_out == ax

    def test_plot_3d_heatmap_kwargs(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution,
            edgecolors="black",
            linewidths=0.5,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_heatmap_show_plot_true(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, show_plot=True, variable="Cell temperature [K]", t=None
        )
        assert ax is not None

    def test_plot_3d_heatmap_non_3d_model_error(self, non_3d_solution):
        with pytest.raises(
            TypeError, match=r"This function requires a 3D model solution"
        ):
            pybamm.plot_3d_heatmap(
                non_3d_solution, variable="Cell temperature [K]", t=None
            )

    def test_plot_3d_heatmap_non_3d_axes_error(self, box_3d_solution):
        _, ax = plt.subplots()
        with pytest.raises(
            TypeError, match=r"The provided axes `ax` must be a 3D projection"
        ):
            pybamm.plot_3d_heatmap(
                box_3d_solution,
                ax=ax,
                show_plot=False,
                variable="Cell temperature [K]",
                t=None,
            )

    def test_plot_3d_heatmap_colorbar_present(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, show_plot=False, variable="Cell temperature [K]", t=None
        )
        fig = ax.get_figure()
        assert len(fig.axes) == 2

    def test_plot_3d_heatmap_axis_labels(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, show_plot=False, variable="Cell temperature [K]", t=None
        )
        assert "x [m]" in ax.get_xlabel()
        assert "y [m]" in ax.get_ylabel()
        assert "z [m]" in ax.get_zlabel()

    def test_plot_3d_heatmap_title_content(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, t=50, show_plot=False, variable="Cell temperature [K]"
        )
        title = ax.get_title()
        assert "3D Heatmap" in title
        assert "t=50.0s" in title

    def test_plot_3d_heatmap_view_angle_set(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution, show_plot=False, variable="Cell temperature [K]", t=None
        )
        elev, azim = ax.elev, ax.azim
        assert elev == 20
        assert azim == -65

    def test_plot_3d_heatmap_scatter_properties(self, box_3d_solution):
        ax = pybamm.plot_3d_heatmap(
            box_3d_solution,
            marker_size=15,
            alpha=0.8,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        collections = ax.collections
        assert len(collections) > 0
        scatter = collections[0]
        assert scatter.get_alpha() == 0.8
