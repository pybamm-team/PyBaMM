import matplotlib.pyplot as plt
import pytest
from matplotlib import use

import pybamm

use("Agg")


class TestPlot3DCrossSection:
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

    def test_plot_3d_cross_section_default_params(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution, show_plot=False, variable="Cell temperature [K]", t=None
        )
        assert ax is not None

    def test_plot_3d_cross_section_with_time(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution, t=50, show_plot=False, variable="Cell temperature [K]"
        )
        assert ax is not None

    def test_plot_3d_cross_section_time_none(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution, t=None, show_plot=False, variable="Cell temperature [K]"
        )
        assert ax is not None

    def test_plot_3d_cross_section_all_planes_box(self, box_3d_solution):
        planes = ["xy", "yz", "xz"]
        for plane in planes:
            ax = pybamm.plot_3d_cross_section(
                box_3d_solution,
                plane=plane,
                show_plot=False,
                variable="Cell temperature [K]",
                t=None,
            )
            assert ax is not None

    def test_plot_3d_cross_section_cylindrical_xy(self, cylindrical_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            cylindrical_3d_solution,
            plane="xy",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_cylindrical_rz(self, cylindrical_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            cylindrical_3d_solution,
            plane="rz",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_positions(self, box_3d_solution):
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        for position in positions:
            ax = pybamm.plot_3d_cross_section(
                box_3d_solution,
                position=position,
                show_plot=False,
                variable="Cell temperature [K]",
                t=None,
            )
            assert ax is not None

    def test_plot_3d_cross_section_custom_variable(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution, variable="Cell temperature [K]", show_plot=False, t=None
        )
        assert ax is not None

    def test_plot_3d_cross_section_different_n_pts(self, box_3d_solution):
        for n_pts in [20, 50, 150]:
            ax = pybamm.plot_3d_cross_section(
                box_3d_solution,
                n_pts=n_pts,
                show_plot=False,
                variable="Cell temperature [K]",
                t=None,
            )
            assert ax is not None

    def test_plot_3d_cross_section_custom_cmap(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            cmap="viridis",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_custom_levels(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            levels=10,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_use_offset_true(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            use_offset=True,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_use_offset_false(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            use_offset=False,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_custom_ax(self, box_3d_solution):
        _, ax = plt.subplots(figsize=(6, 4))
        ax_out = pybamm.plot_3d_cross_section(
            box_3d_solution,
            ax=ax,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax_out == ax

    def test_plot_3d_cross_section_custom_ax_polar(self, cylindrical_3d_solution):
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax_out = pybamm.plot_3d_cross_section(
            cylindrical_3d_solution,
            plane="xy",
            ax=ax,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax_out == ax

    def test_plot_3d_cross_section_kwargs(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            alpha=0.8,
            linewidths=0.5,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_show_plot_true(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution, show_plot=True, variable="Cell temperature [K]", t=None
        )
        assert ax is not None

    def test_plot_3d_cross_section_non_3d_model_error(self, non_3d_solution):
        with pytest.raises(
            TypeError, match=r"This function requires a 3D model solution"
        ):
            pybamm.plot_3d_cross_section(
                non_3d_solution,
                t=None,
                show_plot=False,
                variable="Cell temperature [K]",
            )

    def test_plot_3d_cross_section_invalid_plane_error(self, box_3d_solution):
        with pytest.raises(ValueError, match=r"Plane 'invalid' invalid"):
            pybamm.plot_3d_cross_section(
                box_3d_solution,
                plane="invalid",
                show_plot=False,
                variable="Cell temperature [K]",
                t=None,
            )

    def test_plot_3d_cross_section_colorbar_present(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution, show_plot=False, variable="Cell temperature [K]", t=None
        )
        fig = ax.get_figure()
        assert len(fig.axes) == 2

    def test_plot_3d_cross_section_aspect_ratio(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution, show_plot=False, variable="Cell temperature [K]", t=None
        )
        assert ax.get_aspect() == "auto"

    def test_plot_3d_cross_section_cylindrical_polar_limits(
        self, cylindrical_3d_solution
    ):
        ax = pybamm.plot_3d_cross_section(
            cylindrical_3d_solution,
            plane="xy",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax.get_ylim()[0] >= 0

    def test_plot_3d_cross_section_title_content(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution, t=50, show_plot=False, variable="Cell temperature [K]"
        )
        title = ax.get_title()
        assert "t=50.0s" in title

    def test_plot_3d_cross_section_labels_box_xy(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            plane="xy",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert "x [m]" in ax.get_xlabel()
        assert "y [m]" in ax.get_ylabel()

    def test_plot_3d_cross_section_labels_box_yz(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            plane="yz",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert "y [m]" in ax.get_xlabel()
        assert "z [m]" in ax.get_ylabel()

    def test_plot_3d_cross_section_labels_box_xz(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            plane="xz",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert "x [m]" in ax.get_xlabel()
        assert "z [m]" in ax.get_ylabel()

    def test_plot_3d_cross_section_labels_cylindrical_rz(self, cylindrical_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            cylindrical_3d_solution,
            plane="rz",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert "Radius r [m]" in ax.get_xlabel()
        assert "Height z [m]" in ax.get_ylabel()

    def test_plot_3d_cross_section_show_mesh_box_xy(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            plane="xy",
            show_mesh=True,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_show_mesh_box_yz(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            plane="yz",
            show_mesh=True,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_show_mesh_box_xz(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            plane="xz",
            show_mesh=True,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_show_mesh_cylindrical_xy(
        self, cylindrical_3d_solution
    ):
        ax = pybamm.plot_3d_cross_section(
            cylindrical_3d_solution,
            plane="xy",
            show_mesh=True,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_show_mesh_cylindrical_rz(
        self, cylindrical_3d_solution
    ):
        ax = pybamm.plot_3d_cross_section(
            cylindrical_3d_solution,
            plane="rz",
            show_mesh=True,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_mesh_color_custom(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            show_mesh=True,
            mesh_color="red",
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_mesh_alpha_custom(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            show_mesh=True,
            mesh_alpha=0.8,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_mesh_linewidth_custom(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            show_mesh=True,
            mesh_linewidth=1.5,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_mesh_tolerance_custom(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            show_mesh=True,
            mesh_tolerance=0.001,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_mesh_tolerance_none(self, box_3d_solution):
        ax = pybamm.plot_3d_cross_section(
            box_3d_solution,
            show_mesh=True,
            mesh_tolerance=None,
            show_plot=False,
            variable="Cell temperature [K]",
            t=None,
        )
        assert ax is not None

    def test_plot_3d_cross_section_cylindrical_invalid_plane(
        self, cylindrical_3d_solution
    ):
        with pytest.raises(
            ValueError, match=r"Plane 'yz' invalid for cylindrical geometry"
        ):
            pybamm.plot_3d_cross_section(
                cylindrical_3d_solution,
                plane="yz",
                show_plot=False,
                variable="Cell temperature [K]",
                t=None,
            )

    def test_plot_3d_cross_section_cylindrical_invalid_plane_xz(
        self, cylindrical_3d_solution
    ):
        with pytest.raises(
            ValueError, match=r"Plane 'xz' invalid for cylindrical geometry"
        ):
            pybamm.plot_3d_cross_section(
                cylindrical_3d_solution,
                plane="xz",
                show_plot=False,
                variable="Cell temperature [K]",
                t=None,
            )
