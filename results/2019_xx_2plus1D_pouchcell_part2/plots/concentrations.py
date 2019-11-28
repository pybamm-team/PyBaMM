import numpy as np
import matplotlib.pyplot as plt


def plot_x_av_surf_concentration(t, reduced, full):

    var_names = [
        "X-averaged negative particle surface concentration",
        "X-averaged positive particle surface concentration",
    ]

    plot_names = [
        r"$c_{\mathregular{s,n,surf}}$",
        r"$c_{\mathregular{s,p,surf}}$",
    ]

    # color_lim = [(-0.00025, 0), (0, 0.0005), None]
    color_lim = [None, None, None]

    num_of_models = 0

    if reduced:
        num_of_models += 1
    if full:
        num_of_models += 1

    fig, ax = plt.subplots(len(var_names), num_of_models)
    fig.set_figheight(9)

    if num_of_models == 1:
        fig.set_figwidth(6)
    elif num_of_models == 2:
        fig.set_figwidth(11)
    elif num_of_models == 3:
        fig.set_figwidth(15)

    fig.tight_layout()

    model_idx = 0

    if reduced:
        model_name = "2+1 SPMe"
        plot_model(
            fig, ax, model_idx, t, reduced, model_name, var_names, plot_names, color_lim
        )
        model_idx += 1

    if full:
        model_name = "2+1 DFN"
        plot_model(
            fig, ax, model_idx, t, reduced, model_name, var_names, plot_names, color_lim
        )
        model_idx += 1


def plot_model(
    fig, ax, model_idx, t, model, model_name, var_names, plot_names, color_lim
):
    y = np.linspace(0, 1.5, 100)
    z = np.linspace(0, 1, 100)
    y_dim = y * model["L_z"]
    z_dim = z * model["L_z"]

    for i, var_name in enumerate(var_names):
        # just doing this because effective resistance model is
        # inconsistent with other models
        name = var_name
        make_2D_plot(
            fig,
            ax[i, model_idx],
            y_dim,
            z_dim,
            np.transpose(model[name](t=t, y=y, z=z)),
            model_name + " " + plot_names[i],
            color_lim[i],
        )


def make_2D_plot(fig, ax, y, z, var, name, color_lim):

    if color_lim is not None:
        im = ax.pcolormesh(
            y, z, var, vmin=color_lim[0], vmax=color_lim[1], shading="gouraud"
        )
    else:
        im = ax.pcolormesh(y, z, var, shading="gouraud")
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z$")
    ax.set_title(name)

    fig.colorbar(im, ax=ax)

