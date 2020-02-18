import pybamm
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})


# Figure 4 - Particle size distribution

fig4, axes4 = plt.subplots(1, 3, num=4, figsize=(6, 2))
particle_distribution_graphite = pd.read_csv(
    "~/LGM50/data/particle_distribution_graphite.csv"
)

particle_distribution_silicon = pd.read_csv(
    "~/LGM50/data/particle_distribution_silicon.csv"
)

particle_distribution_NMC = pd.read_csv(
    "~/LGM50/data/particle_distribution_NMC.csv"
)

data_graphite = []
for v in particle_distribution_graphite.to_numpy():
    data_graphite = np.append(data_graphite, np.full(int(v[1]), v[0]))

data_silicon = []
for v in particle_distribution_silicon.to_numpy():
    data_silicon = np.append(data_silicon, np.full(int(v[1]), v[0]))

data_NMC = []
for v in particle_distribution_NMC.to_numpy():
    data_NMC = np.append(data_NMC, np.full(int(v[1]), v[0]))

axes4[0].hist(data_NMC, bins=np.arange(0, 15))
axes4[0].set_xlim(0, 14)
axes4[0].set_xlabel("Particle radius ($\mu$m)")
axes4[0].set_ylabel("Count")
axes4[0].set_title("Positive elect.: NMC")

axes4[1].hist(data_graphite, bins=np.arange(0, 13))
axes4[1].set_xlim(0, 12)
axes4[1].set_xlabel("Particle radius ($\mu$m)")
axes4[1].set_ylabel("Count")
axes4[1].set_title("Negative elect.: graphite")

axes4[2].hist(data_silicon, bins=np.arange(0, 4.5, 0.5))
axes4[2].set_xlim(0, 4)
axes4[2].set_xlabel("Particle radius ($\mu$m)")
axes4[2].set_ylabel("Count")
axes4[2].set_title("Negative elect.: silicon")

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig4.png",
    dpi=300
)

# Figure 8 - Sketch pseudo vs true OCV
capacity = np.linspace(0, 1, 100)
capacity_red = np.linspace(0.5, 3.5, 4)
a = -0.25


plt.figure(num=8, figsize=(6, 4))

for i in range(-1,4):
    plt.plot(
        capacity + 0.5 + i, a * (capacity + 0.5 + i) + 0.5 - np.sqrt(capacity) - 0.25,
        color="black", linestyle="dashed"
    )
    plt.plot(
        capacity + 0.5 + i, a * (capacity + 0.5 + i) + 1.5 + np.sqrt(1 - capacity) + 0.25,
        color="black", linestyle="dashed"
    )

for i in range(0,4):
    plt.plot(
        (0.5 + i) * np.array([1, 1]), a * (0.5 + i) + np.array([0.5, -0.75]),
        color="black"
    )
    plt.plot(
        (0.5 + i) * np.array([1, 1]), a * (0.5 + i) + np.array([1.5, 2.75]),
        color="black"
    )

plt.plot(
    4 * capacity, a * 4 * capacity + 2, color="blue", label="delithiation pseudo"
)
plt.plot(
    4 * capacity, a * 4 * capacity + 0, color="red", label="lithiation pseudo"
)
plt.plot(
    capacity_red, a * capacity_red + 1.5, 
    color="blue", marker="x", linestyle="None",
    label="delithiation true"
)
plt.plot(
    capacity_red, a * capacity_red + 0.5, 
    color="red", marker="x", linestyle="None",
    label="lithiation true"
)

plt.arrow(
    1.75, a * 1.75 + 0.25, 0.5, a * 0.5,
    color="red", width=0.01, head_width=0.1, length_includes_head=True
)

plt.arrow(
    2.25, a * 2.25 + 1.75, -0.5, a * (-0.5),
    color="blue", width=0.01, head_width=0.1, length_includes_head=True
)

plt.xlabel("Capacity")
plt.ylabel("Electrode potential")
plt.xlim([0, 4])
plt.xticks([])
plt.yticks([])
plt.legend()

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig8.png",
    dpi=300
)

# Figure 9 - Pseudo-OCV vs true OCV
ElCell_OCP = pd.read_csv(
    "~/LGM50/data/ElCell_OCP.csv"
)
ElCell_pseudo = pd.read_csv(
    "~/LGM50/data/ElCell_pseudo.csv"
)

fig9, axes9 = plt.subplots(1, 2, num=9, figsize=(6, 2.5))
axes9[0].plot(
    ElCell_OCP.to_numpy()[:, 0],
    ElCell_OCP.to_numpy()[:, 1],
    color="blue",
    linewidth=1,
    label="true OCV"
)
axes9[0].plot(
    ElCell_pseudo.to_numpy()[:, 0] - 0.0556,
    ElCell_pseudo.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="pseudo OCV"
)
axes9[0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes9[0].set_ylabel("Potential (V)")
axes9[0].set_title("Positive electrode")

axes9[1].plot(
    ElCell_OCP.to_numpy()[:, 0],
    ElCell_OCP.to_numpy()[:, 2],
    color="blue",
    linewidth=1,
    label="true OCV"
)
axes9[1].plot(
    ElCell_pseudo.to_numpy()[:, 0] + 0.1942,
    ElCell_pseudo.to_numpy()[:, 2],
    color="red",
    linewidth=1,
    label="pseudo OCV"
)
axes9[1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes9[1].set_ylabel("Potential (V)")
axes9[1].set_title("Negative electrode")
axes9[1].legend()

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig9.png",
    dpi=300
)

# # Figure 10 - dQ/dV plots
# cathode_dQdE_lithiation = pd.read_csv(
#     "~/LGM50/data/cathode_dQdE_lithiation.csv"
# )
# cathode_dQdE_delithiation = pd.read_csv(
#     "~/LGM50/data/cathode_dQdE_delithiation.csv"
# )
# anode_dQdE_lithiation = pd.read_csv(
#     "~/LGM50/data/anode_dQdE_lithiation.csv"
# )
# anode_dQdE_delithiation = pd.read_csv(
#     "~/LGM50/data/anode_dQdE_delithiation.csv"
# )
# cathode_dQdE_pseudo_lithiation = pd.read_csv(
#     "~/LGM50/data/cathode_dQdE_pseudo_lithiation.csv"
# )
# cathode_dQdE_pseudo_delithiation = pd.read_csv(
#     "~/LGM50/data/cathode_dQdE_pseudo_delithiation.csv"
# )
# anode_dQdE_pseudo_lithiation = pd.read_csv(
#     "~/LGM50/data/anode_dQdE_pseudo_lithiation.csv"
# )
# anode_dQdE_pseudo_delithiation = pd.read_csv(
#     "~/LGM50/data/anode_dQdE_pseudo_delithiation.csv"
# )



# fig10, axes10 = plt.subplots(1, 2, num=10, figsize=(6, 2.5))
# axes10[0].plot(
#     cathode_dQdE_delithiation.to_numpy()[:, 0],
#     cathode_dQdE_delithiation.to_numpy()[:, 1],
#     color="blue",
#     linewidth=1,
#     label="delithiation true"
# )
# axes10[0].plot(
#     cathode_dQdE_lithiation.to_numpy()[:, 0],
#     cathode_dQdE_lithiation.to_numpy()[:, 1],
#     color="red",
#     linewidth=1,
#     label="lithiation true"
# )
# axes10[0].plot(
#     cathode_dQdE_pseudo_delithiation.to_numpy()[:, 0],
#     cathode_dQdE_pseudo_delithiation.to_numpy()[:, 1],
#     color="blue",
#     linestyle="--",
#     linewidth=1,
#     label="delithiation pseudo"
# )
# axes10[0].plot(
#     cathode_dQdE_pseudo_lithiation.to_numpy()[:, 0],
#     cathode_dQdE_pseudo_lithiation.to_numpy()[:, 1],
#     color="red",
#     linestyle="--",
#     linewidth=1,
#     label="lithiation pseudo"
# )
# axes10[0].set_xlim(3.5, 4.3)
# axes10[0].set_xlabel("Potential (V)")
# axes10[0].set_ylabel("dQ/dE (mAh/V)")
# axes10[0].set_title("Positive electrode")
# # axes10[0].legend()

# axes10[1].plot(
#     anode_dQdE_delithiation.to_numpy()[:, 0],
#     anode_dQdE_delithiation.to_numpy()[:, 1],
#     color="blue",
#     linewidth=1,
#     label="delithiation true"
# )
# axes10[1].plot(
#     anode_dQdE_lithiation.to_numpy()[:, 0],
#     anode_dQdE_lithiation.to_numpy()[:, 1],
#     color="red",
#     linewidth=1,
#     label="lithiation true"
# )
# axes10[1].plot(
#     anode_dQdE_pseudo_delithiation.to_numpy()[:, 0],
#     anode_dQdE_pseudo_delithiation.to_numpy()[:, 1],
#     color="blue",
#     linestyle="--",
#     linewidth=1,
#     label="delithiation pseudo"
# )
# axes10[1].plot(
#     anode_dQdE_pseudo_lithiation.to_numpy()[:, 0],
#     anode_dQdE_pseudo_lithiation.to_numpy()[:, 1],
#     color="red",
#     linestyle="--",
#     linewidth=1,
#     label="lithiation pseudo"
# )
# axes10[1].set_xlim(0.05, 0.25)
# axes10[1].set_xlabel("Potential (V)")
# axes10[1].set_ylabel("dQ/dE (mAh/V)")
# axes10[1].set_title("Negative electrode")
# axes10[1].legend()

# plt.tight_layout()

# plt.savefig(
#     "~/LGM50/figures/fig10.png",
#     dpi=300
# )

# # Figure 11 - 2-electrode vs 3-electrode OCV
# anode_OCP_half = pd.read_csv(
#     "~/LGM50/data/anode_OCP_half.csv"
# )
# cathode_OCP_half = pd.read_csv(
#     "~/LGM50/data/cathode_OCP_half.csv"
# )

# fig11, axes11 = plt.subplots(1, 2, num=11, figsize=(6, 3))
# axes11[0].plot(
#     ElCell_OCP.to_numpy()[:, 0],
#     ElCell_OCP.to_numpy()[:, 1],
#     color="blue",
#     linewidth=1,
#     label="3-electrode"
# )
# axes11[0].plot(
#     cathode_OCP_half.to_numpy()[:, 0] - 0.3397,
#     cathode_OCP_half.to_numpy()[:, 1],
#     color="red",
#     linewidth=1,
#     label="2-electrode"
# )
# axes11[0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
# axes11[0].set_ylabel("Potential (V)")
# axes11[0].set_xlim((-1, 5))
# axes11[0].set_title("Positive electrode")

# a_cat = -0.1470
# b_cat = 0.9072

# def C2S_cathode(x):
#     return a_cat * x + b_cat


# def S2C_cathode(x):
#     return (x - b_cat) / a_cat

# secaxcat = axes11[0].secondary_xaxis('top', functions=(C2S_cathode, S2C_cathode))
# secaxcat.set_xlabel("Stoichiometry")


# axes11[1].plot(
#     ElCell_OCP.to_numpy()[:, 0],
#     ElCell_OCP.to_numpy()[:, 2],
#     color="blue",
#     linewidth=1,
#     label="3-electrode"
# )
# axes11[1].plot(
#     anode_OCP_half.to_numpy()[:, 0] - 0.1391,
#     anode_OCP_half.to_numpy()[:, 1],
#     color="red",
#     linewidth=1,
#     label="2-electrode"
# )
# axes11[1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
# axes11[1].set_ylabel("Potential (V)")
# axes11[1].set_title("Negative electrode")
# axes11[1].legend()

# a_an = 0.1974
# b_an = 0.0279

# def C2S_anode(x):
#     return a_an* x + b_an


# def S2C_anode(x):
#     return (x - b_an) / a_an

# secaxcat = axes11[1].secondary_xaxis('top', functions=(C2S_anode, S2C_anode))
# secaxcat.set_xlabel("Stoichiometry")

# plt.tight_layout()

# plt.savefig(
#     "~/LGM50/figures/fig11.png",
#     dpi=300
# )


# # Figure 12 - Diffusion coefficients & GITT data
# cathode_GITT_lithiation = pd.read_csv(
#     "~/LGM50/data/cathode_GITT_lithiation.csv"
# )
# cathode_GITT_delithiation = pd.read_csv(
#     "~/LGM50/data/cathode_GITT_delithiation.csv"
# )
# anode_GITT_lithiation = pd.read_csv(
#     "~/LGM50/data/anode_GITT_lithiation.csv"
# )
# anode_GITT_delithiation = pd.read_csv(
#     "~/LGM50/data/anode_GITT_delithiation.csv"
# )
# cathode_diffusivity_lithiation = pd.read_csv(
#     "~/LGM50/data/cathode_diffusivity_lithiation.csv"
# )
# cathode_diffusivity_delithiation = pd.read_csv(
#     "~/LGM50/data/cathode_diffusivity_delithiation.csv"
# )
# anode_diffusivity_lithiation = pd.read_csv(
#     "~/LGM50/data/anode_diffusivity_lithiation.csv"
# )
# anode_diffusivity_delithiation = pd.read_csv(
#     "~/LGM50/data/anode_diffusivity_delithiation.csv"
# )

# D_cathode = np.concatenate(
#     (cathode_diffusivity_lithiation.to_numpy()[:, 1],
#     cathode_diffusivity_delithiation.to_numpy()[:, 1]),
#     axis=0
# )
# D_anode = np.concatenate(
#     (anode_diffusivity_lithiation.to_numpy()[:, 1],
#     anode_diffusivity_delithiation.to_numpy()[:, 1]),
#     axis=0
# )


# print("Average diffusion cathode: ", np.average(D_cathode), " +- ", np.std(D_cathode) )
# print("Average diffusion anode: ", np.average(D_anode), " +- ", np.std(D_anode) )

# fig12, axes12 = plt.subplots(2, 2, num=12, figsize=(6, 4.5))
# axes12[0, 0].semilogy(
#     cathode_diffusivity_delithiation.to_numpy()[:, 0],
#     cathode_diffusivity_delithiation.to_numpy()[:, 1],
#     color="blue", linestyle="None", marker="o", markersize=3, label="delithiation"
# )
# axes12[0, 0].semilogy(
#     cathode_diffusivity_lithiation.to_numpy()[:, 0],
#     cathode_diffusivity_lithiation.to_numpy()[:, 1],
#     color="red", linestyle="None", marker="o", markersize=3, label="lithiation"
# )
# axes12[0, 0].set_xlim(left = -1)
# axes12[0, 0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
# axes12[0, 0].set_ylabel("Diffusivity ($\mathrm{cm}^2 \mathrm{s}^{-1}$)")
# axes12[0, 0].set_title("(a) Positive electrode")
# # axes12[0, 0].legend(loc="upper left")

# axes12[1, 0].plot(
#     cathode_GITT_delithiation.to_numpy()[:, 0],
#     cathode_GITT_delithiation.to_numpy()[:, 1],
#     color="blue",
#     # linewidth=0.5,
#     label="delithiation"
# )
# axes12[1, 0].plot(
#     cathode_GITT_lithiation.to_numpy()[:, 0],
#     cathode_GITT_lithiation.to_numpy()[:, 1],
#     color="red",
#     # linewidth=0.5,
#     label="lithiation"
# )
# axes12[1, 0].set_xlim(left = -1)
# axes12[1, 0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
# axes12[1, 0].set_ylabel("Potential (V)")
# axes12[1, 0].set_title("(c) Positive electrode")
# # axes12[0, 1].legend(loc="upper left")

# axes12[0, 1].semilogy(
#     anode_diffusivity_delithiation.to_numpy()[:, 0],
#     anode_diffusivity_delithiation.to_numpy()[:, 1],
#     color="blue", linestyle="None", marker="o", markersize=3, label="delithiation"
# )
# axes12[0, 1].semilogy(
#     anode_diffusivity_lithiation.to_numpy()[:, 0],
#     anode_diffusivity_lithiation.to_numpy()[:, 1],
#     color="red", linestyle="None", marker="o", markersize=3, label="lithiation"
# )
# axes12[0, 1].set_ylim(bottom=1E-18)
# axes12[0, 1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
# axes12[0, 1].set_ylabel("Diffusivity ($\mathrm{cm}^2 \mathrm{s}^{-1}$)")
# axes12[0, 1].set_title("(b) Negative electrode")
# # axes12[1, 0].legend(loc="upper left")

# axes12[1, 1].plot(
#     np.abs(anode_GITT_delithiation.to_numpy()[:, 0]),
#     anode_GITT_delithiation.to_numpy()[:, 1],
#     color="blue",
#     # linewidth=0.5,
#     label="delithiation"
# )
# axes12[1, 1].plot(
#     np.abs(anode_GITT_lithiation.to_numpy()[:, 0]),
#     anode_GITT_lithiation.to_numpy()[:, 1],
#     color="red",
#     # linewidth=0.5,
#     label="lithiation"
# )
# axes12[1, 1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
# axes12[1, 1].set_ylabel("Potential (V)")
# axes12[1, 1].set_title("(d) Negative electrode")
# axes12[1, 1].legend(loc="upper right")

# plt.tight_layout()

# plt.savefig(
#     "~/LGM50/figures/fig12.png",
#     dpi=300
# )


# # Figure 14 - EIS at different temperatures
# cathode_EIS_25degC = pd.read_csv(
#     "~/LGM50/data/cathode_EIS_25degC.csv"
# )
# cathode_EIS_30degC = pd.read_csv(
#     "~/LGM50/data/cathode_EIS_30degC.csv"
# )
# cathode_EIS_40degC = pd.read_csv(
#     "~/LGM50/data/cathode_EIS_40degC.csv"
# )
# cathode_EIS_50degC = pd.read_csv(
#     "~/LGM50/data/cathode_EIS_50degC.csv"
# )
# cathode_EIS_60degC = pd.read_csv(
#     "~/LGM50/data/cathode_EIS_60degC.csv"
# )
# anode_EIS_25degC = pd.read_csv(
#     "~/LGM50/data/anode_EIS_25degC.csv"
# )
# anode_EIS_30degC = pd.read_csv(
#     "~/LGM50/data/anode_EIS_30degC.csv"
# )
# anode_EIS_40degC = pd.read_csv(
#     "~/LGM50/data/anode_EIS_40degC.csv"
# )
# anode_EIS_50degC = pd.read_csv(
#     "~/LGM50/data/anode_EIS_50degC.csv"
# )
# anode_EIS_60degC = pd.read_csv(
#     "~/LGM50/data/anode_EIS_60degC.csv"
# )

# fig14, axes14 = plt.subplots(1, 2, num=14, figsize=(6, 2.5))
# # axes14[0].scatter(
# #     cathode_EIS_25degC.to_numpy()[:, 0],
# #     cathode_EIS_25degC.to_numpy()[:, 1],
# #     c="orange", s=5, label="25°C"
# # )
# axes14[0].scatter(
#     cathode_EIS_30degC.to_numpy()[:, 0],
#     cathode_EIS_30degC.to_numpy()[:, 1],
#     c="black", s=5, label="30°C"
# )
# axes14[0].scatter(
#     cathode_EIS_40degC.to_numpy()[:, 0],
#     cathode_EIS_40degC.to_numpy()[:, 1],
#     c="red", s=5, label="40°C"
# )
# axes14[0].scatter(
#     cathode_EIS_50degC.to_numpy()[:, 0],
#     cathode_EIS_50degC.to_numpy()[:, 1],
#     c="green", s=5, label="50°C"
# )
# axes14[0].scatter(
#     cathode_EIS_60degC.to_numpy()[:, 0],
#     cathode_EIS_60degC.to_numpy()[:, 1],
#     c="blue", s=5, label="60°C"
# )
# axes14[0].set_xlabel("Re(Z) ($\Omega$)")
# axes14[0].set_ylabel("-Im(Z) ($\Omega$)")
# axes14[0].set_title("Positive electrode")
# # axes14[0].legend()

# # axes14[1].scatter(
# #     anode_EIS_25degC.to_numpy()[:, 0],
# #     anode_EIS_25degC.to_numpy()[:, 1],
# #     c="orange", s=5, label="25°C"
# # )
# axes14[1].scatter(
#     anode_EIS_30degC.to_numpy()[:, 0],
#     anode_EIS_30degC.to_numpy()[:, 1],
#     c="black", s=5, label="30°C"
# )
# axes14[1].scatter(
#     anode_EIS_40degC.to_numpy()[:, 0],
#     anode_EIS_40degC.to_numpy()[:, 1],
#     c="red", s=5, label="40°C"
# )
# axes14[1].scatter(
#     anode_EIS_50degC.to_numpy()[:, 0],
#     anode_EIS_50degC.to_numpy()[:, 1],
#     c="green", s=5, label="50°C"
# )
# axes14[1].scatter(
#     anode_EIS_60degC.to_numpy()[:, 0],
#     anode_EIS_60degC.to_numpy()[:, 1],
#     c="blue", s=5, label="60°C"
# )
# axes14[1].set_xlabel("Re(Z) ($\Omega$)")
# axes14[1].set_ylabel("-Im(Z) ($\Omega$)")
# axes14[1].set_title("Negative electrode")
# axes14[1].legend()

# plt.tight_layout()

# plt.savefig(
#     "~/LGM50/figures/fig14.png",
#     dpi=300
# )

# # Figure 15 - Arrhenius plot
# arrhenius_Rct = pd.read_csv(
#     "~/LGM50/data/arrhenius_Rct.csv"
# )

# # arrhenius_T = arrhenius_Rct.to_numpy()[1:, 0]
# # arrhenius_Rct_cathode = arrhenius_Rct.to_numpy()[1:, 1]
# # arrhenius_Rct_anode = arrhenius_Rct.to_numpy()[1:, 2]

# arrhenius_T = arrhenius_Rct.to_numpy()[:, 0]
# arrhenius_Rct_cathode = arrhenius_Rct.to_numpy()[:, 1]
# arrhenius_Rct_anode = arrhenius_Rct.to_numpy()[:, 2]

# R = 8.314
# F = 96485
# Sp = 4.9706E-3
# Sn = 5.7809E-3

# arrhenius_j0_cathode = R/(Sp * F) * np.divide(arrhenius_T, arrhenius_Rct_cathode)/10
# arrhenius_j0_anode = R/(Sn * F) * np.divide(arrhenius_T, arrhenius_Rct_anode)/10

# print(arrhenius_T)
# print(arrhenius_Rct_cathode)
# print(arrhenius_Rct_anode)

# fit_cathode = np.polyfit(
#     1. / arrhenius_T,
#     np.log(arrhenius_j0_cathode),
#     deg=1
# )
# fit_anode = np.polyfit(
#     1. / arrhenius_T,
#     np.log(arrhenius_j0_anode),
#     deg=1
# )

# plt.figure(num=15, figsize=(6, 4))
# plt.semilogy(
#     1. / arrhenius_T,
#     arrhenius_j0_cathode,
#     color="black", marker='o', markersize=5, linestyle="None",
#     label="positive electrode"
# )
# plt.semilogy(
#     1. / arrhenius_T,
#     arrhenius_j0_anode,
#     color="black", marker='x', markersize=5, linestyle="None",
#     label="negative electrode"
# )
# plt.semilogy(
#     1. / arrhenius_T,
#     np.exp(fit_cathode[0] / arrhenius_T + fit_cathode[1]),
#     color="blue",
# )
# plt.semilogy(
#     1. / arrhenius_T,
#     np.exp(fit_anode[0] / arrhenius_T + fit_anode[1]),
#     color="blue",
# )
# plt.xlabel("1/T ($\mathrm{K}^{-1})$")
# plt.ylabel("$j_0$ (mA $\mathrm{cm}^{-2}$)")
# plt.ylim((1E-2, 1))
# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     "~/LGM50/figures/fig15.png",
#     dpi=300
# )

# print("Cathode: ", fit_cathode[0] * R)
# print("Anode: ", fit_anode[0] * R)

# # Figure S7 - EIS at room temperature
# cathode_Rct_RT = pd.read_csv(
#     "~/LGM50/data/cathode_Rct_RT.csv"
# )
# anode_Rct_RT = pd.read_csv(
#     "~/LGM50/data/anode_Rct_RT.csv"
# )

# j0_cathode = R/(Sp * F) * 298.15 * np.divide(1, cathode_Rct_RT.to_numpy()[:,1])/10
# j0_anode = R/(Sn * F) * 298.15 * np.divide(1, anode_Rct_RT.to_numpy()[:,1])/10

# fig27, axes27 = plt.subplots(1, 2, num=27, figsize=(6, 2.5))
# axes27[0].scatter(
#     cathode_Rct_RT.to_numpy()[:,0],
#     j0_cathode,
#     c="black", s=5
# )
# axes27[0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
# axes27[0].set_ylabel("$j_0$ (mA $\mathrm{cm}^{-2}$)")
# axes27[0].set_title("Positive electrode")

# axes27[1].scatter(
#     anode_Rct_RT.to_numpy()[:,0],
#     j0_anode,
#     c="black", s=5
# )
# axes27[1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
# axes27[1].set_ylabel("$j_0$ (mA $\mathrm{cm}^{-2}$)")
# axes27[1].set_title("Negative electrode")

# plt.tight_layout()

# plt.savefig(
#     "~/LGM50/figures/figS7.png",
#     dpi=300
# )


plt.show()