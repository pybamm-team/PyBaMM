import pybamm
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})


# Figure 5 - Particle size distribution

fig5, axes5 = plt.subplots(1, 3, num=5, figsize=(6, 2))
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

axes5[0].hist(data_NMC, bins=np.arange(0, 15))
axes5[0].set_xlim(0, 14)
axes5[0].set_xlabel("Particle radius ($\mu$m)")
axes5[0].set_ylabel("Count")
axes5[0].set_title("(a)")
# axes5[0].set_title("Positive elect.: NMC")

axes5[1].hist(data_graphite, bins=np.arange(0, 13))
axes5[1].set_xlim(0, 12)
axes5[1].set_xlabel("Particle radius ($\mu$m)")
axes5[1].set_ylabel("Count")
# axes5[1].set_title("Negative elect.: graphite")
axes5[1].set_title("(b)")

axes5[2].hist(data_silicon, bins=np.arange(0, 4.5, 0.5))
axes5[2].set_xlim(0, 4)
axes5[2].set_xlabel("Particle radius ($\mu$m)")
axes5[2].set_ylabel("Count")
# axes5[2].set_title("Negative elect.: silicon")
axes5[2].set_title("(c)")

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig5.png",
    dpi=300
)

# Figure 9 - Sketch pseudo vs true OCV
capacity = np.linspace(0, 1, 100)
capacity_red = np.linspace(0.5, 3.5, 4)
a = -0.25


plt.figure(num=9, figsize=(6, 4))

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
    4 * capacity, a * 4 * capacity + 2, color="blue", label="delithiation pseudo-OCV"
)
plt.plot(
    4 * capacity, a * 4 * capacity + 0, color="red", label="lithiation pseudo-OCV"
)
plt.plot(
    capacity_red, a * capacity_red + 1.5, 
    color="blue", marker="x", linestyle="None",
    label="delithiation GITT-OCV or EMF"
)
plt.plot(
    capacity_red, a * capacity_red + 0.5, 
    color="red", marker="x", linestyle="None",
    label="lithiation GITT-OCV or EMF"
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
    "/home/ferranbrosa/LGM50/figures/fig9.png",
    dpi=300
)

# Figure 10 - Pseudo-OCV vs true OCV
ElCell_OCP = pd.read_csv(
    "~/LGM50/data/ElCell_OCP.csv"
)
ElCell_pseudo = pd.read_csv(
    "~/LGM50/data/ElCell_pseudo.csv"
)

fig10, axes10 = plt.subplots(2, 2, num=10, figsize=(6, 4.5))
axes10[0, 0].plot(
    ElCell_OCP.to_numpy()[:, 0],
    ElCell_OCP.to_numpy()[:, 1],
    color="blue",
    linewidth=1,
    label="GITT-OCV"
)
axes10[0, 0].plot(
    ElCell_pseudo.to_numpy()[:, 0] - 0.0556,
    ElCell_pseudo.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="pseudo-OCV"
)
axes10[0, 0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes10[0, 0].set_ylabel("Potential (V)")
axes10[0, 0].set_title("(a)")

axes10[0, 1].plot(
    ElCell_OCP.to_numpy()[:, 0],
    ElCell_OCP.to_numpy()[:, 2],
    color="blue",
    linewidth=1,
    label="GITT-OCV"
)
axes10[0, 1].plot(
    ElCell_pseudo.to_numpy()[:, 0] + 0.1942,
    ElCell_pseudo.to_numpy()[:, 2],
    color="red",
    linewidth=1,
    label="pseudo-OCV"
)
axes10[0, 1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes10[0, 1].set_ylabel("Potential (V)")
axes10[0, 1].set_title("(b)")
axes10[0, 1].legend()

axes10[1, 0].plot(
    ElCell_OCP.to_numpy()[:, 0],
    ElCell_OCP.to_numpy()[:, 1],
    color="blue",
    linewidth=1,
    label="true OCV"
)
axes10[1, 0].plot(
    ElCell_pseudo.to_numpy()[:, 0] - 0.0556,
    ElCell_pseudo.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="pseudo OCV"
)
axes10[1, 0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes10[1, 0].set_ylabel("Potential (V)")
axes10[1, 0].set_title("(c)")
axes10[1, 0].set_xlim(2.5, 3.5)
axes10[1, 0].set_ylim(3.8, 4.2)

axes10[1, 1].plot(
    ElCell_OCP.to_numpy()[:, 0],
    ElCell_OCP.to_numpy()[:, 2],
    color="blue",
    linewidth=1,
    label="true OCV"
)
axes10[1, 1].plot(
    ElCell_pseudo.to_numpy()[:, 0] + 0.1942,
    ElCell_pseudo.to_numpy()[:, 2],
    color="red",
    linewidth=1,
    label="pseudo OCV"
)
axes10[1, 1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes10[1, 1].set_ylabel("Potential (V)")
axes10[1, 1].set_title("(d)")
axes10[1, 1].set_xlim(2.6, 3.1)
axes10[1, 1].set_ylim(0, 0.2)

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig10.png",
    dpi=300
)


# Figure 11 - dQ/dV plots
cathode_dQdE_lithiation = pd.read_csv(
    "~/LGM50/data/cathode_dQdE_lithiation.csv"
)
cathode_dQdE_delithiation = pd.read_csv(
    "~/LGM50/data/cathode_dQdE_delithiation.csv"
)
anode_dQdE_lithiation = pd.read_csv(
    "~/LGM50/data/anode_dQdE_lithiation.csv"
)
anode_dQdE_delithiation = pd.read_csv(
    "~/LGM50/data/anode_dQdE_delithiation.csv"
)
cathode_dQdE_pseudo_lithiation = pd.read_csv(
    "~/LGM50/data/cathode_dQdE_pseudo_lithiation.csv"
)
cathode_dQdE_pseudo_delithiation = pd.read_csv(
    "~/LGM50/data/cathode_dQdE_pseudo_delithiation.csv"
)
anode_dQdE_pseudo_lithiation = pd.read_csv(
    "~/LGM50/data/anode_dQdE_pseudo_lithiation.csv"
)
anode_dQdE_pseudo_delithiation = pd.read_csv(
    "~/LGM50/data/anode_dQdE_pseudo_delithiation.csv"
)



fig11, axes11 = plt.subplots(2, 2, num=11, figsize=(6, 4.5))
axes11[0, 0].plot(
    cathode_dQdE_delithiation.to_numpy()[:, 0],
    cathode_dQdE_delithiation.to_numpy()[:, 1],
    color="blue",
    linewidth=1,
    label="delithiation GITT-OCV"
)
axes11[0, 0].plot(
    cathode_dQdE_lithiation.to_numpy()[:, 0],
    cathode_dQdE_lithiation.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="lithiation GITT-OCV"
)
axes11[0, 0].plot(
    cathode_dQdE_pseudo_delithiation.to_numpy()[:, 0],
    cathode_dQdE_pseudo_delithiation.to_numpy()[:, 1],
    color="blue",
    linestyle="--",
    linewidth=1,
    label="delithiation pseudo-OCV"
)
axes11[0, 0].plot(
    cathode_dQdE_pseudo_lithiation.to_numpy()[:, 0],
    cathode_dQdE_pseudo_lithiation.to_numpy()[:, 1],
    color="red",
    linestyle="--",
    linewidth=1,
    label="lithiation pseudo-OCV"
)
axes11[0, 0].set_xlim(3.5, 4.3)
axes11[0, 0].set_xlabel("Potential (V)")
axes11[0, 0].set_ylabel("dQ/dE (mAh/V)")
# axes11[0, 0].set_title("Positive electrode")
axes11[0, 0].set_title("(a)")
# axes11[0, 0].legend()

axes11[0, 1].plot(
    anode_dQdE_delithiation.to_numpy()[:, 0],
    anode_dQdE_delithiation.to_numpy()[:, 1],
    color="blue",
    linewidth=1,
    label="delithiation GITT-OCV"
)
axes11[0, 1].plot(
    anode_dQdE_lithiation.to_numpy()[:, 0],
    anode_dQdE_lithiation.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="lithiation GITT-OCV"
)
axes11[0, 1].plot(
    anode_dQdE_pseudo_delithiation.to_numpy()[:, 0],
    anode_dQdE_pseudo_delithiation.to_numpy()[:, 1],
    color="blue",
    linestyle="--",
    linewidth=1,
    label="delithiation pseudo-OCV"
)
axes11[0, 1].plot(
    anode_dQdE_pseudo_lithiation.to_numpy()[:, 0],
    anode_dQdE_pseudo_lithiation.to_numpy()[:, 1],
    color="red",
    linestyle="--",
    linewidth=1,
    label="lithiation pseudo-OCV"
)
axes11[0, 1].set_xlim(0.05, 0.25)
axes11[0, 1].set_xlabel("Potential (V)")
axes11[0, 1].set_ylabel("dQ/dE (mAh/V)")
# axes11[0, 1].set_title("Negative electrode")
axes11[0, 1].set_title("(b)")
axes11[0, 1].legend()

axes11[1, 0].plot(
    cathode_dQdE_delithiation.to_numpy()[:, 0],
    cathode_dQdE_delithiation.to_numpy()[:, 1],
    color="blue",
    linewidth=1,
    label="delithiation GITT-OCV"
)
axes11[1, 0].plot(
    cathode_dQdE_lithiation.to_numpy()[:, 0],
    cathode_dQdE_lithiation.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="lithiation GITT-OCV"
)
axes11[1, 0].plot(
    cathode_dQdE_pseudo_delithiation.to_numpy()[:, 0],
    cathode_dQdE_pseudo_delithiation.to_numpy()[:, 1],
    color="blue",
    linestyle="--",
    linewidth=1,
    label="delithiation pseudo-OCV"
)
axes11[1, 0].plot(
    cathode_dQdE_pseudo_lithiation.to_numpy()[:, 0],
    cathode_dQdE_pseudo_lithiation.to_numpy()[:, 1],
    color="red",
    linestyle="--",
    linewidth=1,
    label="lithiation pseudo-OCV"
)
axes11[1, 0].set_xlim(3.5, 4.3)
axes11[1, 0].set_xlabel("Potential (V)")
axes11[1, 0].set_ylabel("dQ/dE (mAh/V)")
axes11[1, 0].set_xlim(3.8, 4.1)
axes11[1, 0].set_ylim(-20, 20)
# axes11[1, 0].set_title("Positive electrode")
axes11[1, 0].set_title("(c)")
# axes11[1, 0].legend()

axes11[1, 1].plot(
    anode_dQdE_delithiation.to_numpy()[:, 0],
    anode_dQdE_delithiation.to_numpy()[:, 1],
    color="blue",
    linewidth=1,
    label="delithiation GITT-OCV"
)
axes11[1, 1].plot(
    anode_dQdE_lithiation.to_numpy()[:, 0],
    anode_dQdE_lithiation.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="lithiation GITT-OCV"
)
axes11[1, 1].plot(
    anode_dQdE_pseudo_delithiation.to_numpy()[:, 0],
    anode_dQdE_pseudo_delithiation.to_numpy()[:, 1],
    color="blue",
    linestyle="--",
    linewidth=1,
    label="delithiation pseudo-OCV"
)
axes11[1, 1].plot(
    anode_dQdE_pseudo_lithiation.to_numpy()[:, 0],
    anode_dQdE_pseudo_lithiation.to_numpy()[:, 1],
    color="red",
    linestyle="--",
    linewidth=1,
    label="lithiation pseudo-OCV"
)
axes11[1, 1].set_xlim(0.05, 0.25)
axes11[1, 1].set_xlabel("Potential (V)")
axes11[1, 1].set_ylabel("dQ/dE (mAh/V)")
axes11[1, 1].set_xlim(0.10, 0.15)
axes11[1, 1].set_ylim(-500, 500)
# axes11[1, 1].set_title("Negative electrode")
axes11[1, 1].set_title("(d)")
# axes11[1, 1].legend()

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig11.png",
    dpi=300
)

# Figure 12 - 2-electrode vs 3-electrode OCV
anode_OCP_half = pd.read_csv(
    "~/LGM50/data/anode_OCP_half.csv"
)
cathode_OCP_half = pd.read_csv(
    "~/LGM50/data/cathode_OCP_half.csv"
)

fig12, axes12 = plt.subplots(1, 2, num=12, figsize=(6, 3))
axes12[0].plot(
    ElCell_OCP.to_numpy()[:, 0],
    ElCell_OCP.to_numpy()[:, 1],
    color="blue",
    linewidth=1,
    label="3-electrode"
)
axes12[0].plot(
    cathode_OCP_half.to_numpy()[:, 0] - 0.3397,
    cathode_OCP_half.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="2-electrode"
)
axes12[0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes12[0].set_ylabel("Potential (V)")
axes12[0].set_xlim((-1, 5))
# axes12[0].set_title("Positive electrode")
axes12[0].set_title("(a)")

a_cat = -0.1470
b_cat = 0.9072

def C2S_cathode(x):
    return a_cat * x + b_cat


def S2C_cathode(x):
    return (x - b_cat) / a_cat

secaxcat = axes12[0].secondary_xaxis('top', functions=(C2S_cathode, S2C_cathode))
secaxcat.set_xlabel("Stoichiometry")


axes12[1].plot(
    ElCell_OCP.to_numpy()[:, 0],
    ElCell_OCP.to_numpy()[:, 2],
    color="blue",
    linewidth=1,
    label="3-electrode"
)
axes12[1].plot(
    anode_OCP_half.to_numpy()[:, 0] - 0.1391,
    anode_OCP_half.to_numpy()[:, 1],
    color="red",
    linewidth=1,
    label="2-electrode"
)
axes12[1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes12[1].set_ylabel("Potential (V)")
# axes12[1].set_title("Negative electrode")
axes12[1].set_title("(b)")
axes12[1].legend()

a_an = 0.1974
b_an = 0.0279

def C2S_anode(x):
    return a_an* x + b_an


def S2C_anode(x):
    return (x - b_an) / a_an

secaxcat = axes12[1].secondary_xaxis('top', functions=(C2S_anode, S2C_anode))
secaxcat.set_xlabel("Stoichiometry")

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig12.png",
    dpi=300
)


# Figure 13 - Diffusion coefficients & GITT data
cathode_GITT_lithiation = pd.read_csv(
    "~/LGM50/data/cathode_GITT_lithiation.csv"
)
cathode_GITT_delithiation = pd.read_csv(
    "~/LGM50/data/cathode_GITT_delithiation.csv"
)
anode_GITT_lithiation = pd.read_csv(
    "~/LGM50/data/anode_GITT_lithiation.csv"
)
anode_GITT_delithiation = pd.read_csv(
    "~/LGM50/data/anode_GITT_delithiation.csv"
)
cathode_diffusivity_lithiation = pd.read_csv(
    "~/LGM50/data/cathode_diffusivity_lithiation.csv"
)
cathode_diffusivity_delithiation = pd.read_csv(
    "~/LGM50/data/cathode_diffusivity_delithiation.csv"
)
anode_diffusivity_lithiation = pd.read_csv(
    "~/LGM50/data/anode_diffusivity_lithiation.csv"
)
anode_diffusivity_delithiation = pd.read_csv(
    "~/LGM50/data/anode_diffusivity_delithiation.csv"
)

D_cathode = np.concatenate(
    (cathode_diffusivity_lithiation.to_numpy()[:, 1],
    cathode_diffusivity_delithiation.to_numpy()[:, 1]),
    axis=0
)
D_anode = np.concatenate(
    (anode_diffusivity_lithiation.to_numpy()[:, 1],
    anode_diffusivity_delithiation.to_numpy()[:, 1]),
    axis=0
)


print("Average diffusion cathode: ", np.average(D_cathode), " +- ", np.std(D_cathode) )
print("Average diffusion anode: ", np.average(D_anode), " +- ", np.std(D_anode) )

fig13, axes13 = plt.subplots(3, 2, num=13, figsize=(6, 6.5))
axes13[0, 0].semilogy(
    cathode_diffusivity_delithiation.to_numpy()[:, 0],
    cathode_diffusivity_delithiation.to_numpy()[:, 1],
    color="blue", linestyle="None", marker="o", markersize=3, label="delithiation"
)
axes13[0, 0].semilogy(
    cathode_diffusivity_lithiation.to_numpy()[:, 0],
    cathode_diffusivity_lithiation.to_numpy()[:, 1],
    color="red", linestyle="None", marker="o", markersize=3, label="lithiation"
)
axes13[0, 0].set_xlim(left = -1)
axes13[0, 0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes13[0, 0].set_ylabel("Diffusivity ($\mathrm{cm}^2 \mathrm{s}^{-1}$)")
axes13[0, 0].set_title("(a)")
# axes13[0, 0].legend(loc="upper left")

axes13[0, 1].semilogy(
    np.abs(anode_diffusivity_delithiation.to_numpy()[:, 0]),
    anode_diffusivity_delithiation.to_numpy()[:, 1],
    color="blue", linestyle="None", marker="o", markersize=3, label="delithiation"
)
axes13[0, 1].semilogy(
    np.abs(anode_diffusivity_lithiation.to_numpy()[:, 0]),
    anode_diffusivity_lithiation.to_numpy()[:, 1],
    color="red", linestyle="None", marker="o", markersize=3, label="lithiation"
)
axes13[0, 1].set_ylim(bottom=1E-18)
axes13[0, 1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes13[0, 1].set_ylabel("Diffusivity ($\mathrm{cm}^2 \mathrm{s}^{-1}$)")
axes13[0, 1].set_title("(b)")
# axes13[1, 0].legend(loc="upper left")

axes13[1, 0].plot(
    cathode_GITT_delithiation.to_numpy()[:, 0],
    cathode_GITT_delithiation.to_numpy()[:, 1],
    color="blue",
    # linewidth=0.5,
    label="delithiation"
)
axes13[1, 0].plot(
    cathode_GITT_lithiation.to_numpy()[:, 0],
    cathode_GITT_lithiation.to_numpy()[:, 1],
    color="red",
    # linewidth=0.5,
    label="lithiation"
)
axes13[1, 0].set_xlim(left = -1)
axes13[1, 0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes13[1, 0].set_ylabel("Potential (V)")
axes13[1, 0].set_title("(c)")
# axes13[0, 1].legend(loc="upper left")

axes13[1, 1].plot(
    np.abs(anode_GITT_delithiation.to_numpy()[:, 0]),
    anode_GITT_delithiation.to_numpy()[:, 1],
    color="blue",
    # linewidth=0.5,
    label="delithiation"
)
axes13[1, 1].plot(
    np.abs(anode_GITT_lithiation.to_numpy()[:, 0]),
    anode_GITT_lithiation.to_numpy()[:, 1],
    color="red",
    # linewidth=0.5,
    label="lithiation"
)
axes13[1, 1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes13[1, 1].set_ylabel("Potential (V)")
axes13[1, 1].set_title("(d)")
# axes13[1, 1].legend(loc="upper right")

axes13[2, 0].plot(
    cathode_GITT_delithiation.to_numpy()[:, 0],
    cathode_GITT_delithiation.to_numpy()[:, 1],
    color="blue",
    # linewidth=0.5,
    label="delithiation"
)
axes13[2, 0].plot(
    cathode_GITT_lithiation.to_numpy()[:, 0],
    cathode_GITT_lithiation.to_numpy()[:, 1],
    color="red",
    # linewidth=0.5,
    label="lithiation"
)
axes13[2, 0].set_xlim(-0.4, 0.4)
axes13[2, 0].set_ylim(top = 3.7)
axes13[2, 0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes13[2, 0].set_ylabel("Potential (V)")
axes13[2, 0].set_title("(e)")
# axes13[0, 1].legend(loc="upper left")

axes13[2, 1].plot(
    np.abs(anode_GITT_delithiation.to_numpy()[:, 0]),
    anode_GITT_delithiation.to_numpy()[:, 1],
    color="blue",
    # linewidth=0.5,
    label="delithiation"
)
axes13[2, 1].plot(
    np.abs(anode_GITT_lithiation.to_numpy()[:, 0]),
    anode_GITT_lithiation.to_numpy()[:, 1],
    color="red",
    # linewidth=0.5,
    label="lithiation"
)
axes13[2, 1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes13[2, 1].set_ylabel("Potential (V)")
axes13[2, 1].set_xlim(-0.1, 0.4)
axes13[2, 1].set_title("(f)")
axes13[2, 1].legend(loc="upper right")

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig13.png",
    dpi=300
)


# Figure 15 - EIS at different temperatures
cathode_EIS_30degC = pd.read_csv(
    "~/LGM50/data/cathode_EIS_30degC.csv"
)
cathode_EIS_40degC = pd.read_csv(
    "~/LGM50/data/cathode_EIS_40degC.csv"
)
cathode_EIS_50degC = pd.read_csv(
    "~/LGM50/data/cathode_EIS_50degC.csv"
)
cathode_EIS_60degC = pd.read_csv(
    "~/LGM50/data/cathode_EIS_60degC.csv"
)
anode_EIS_30degC = pd.read_csv(
    "~/LGM50/data/anode_EIS_30degC.csv"
)
anode_EIS_40degC = pd.read_csv(
    "~/LGM50/data/anode_EIS_40degC.csv"
)
anode_EIS_50degC = pd.read_csv(
    "~/LGM50/data/anode_EIS_50degC.csv"
)
anode_EIS_60degC = pd.read_csv(
    "~/LGM50/data/anode_EIS_60degC.csv"
)

fig15, axes15 = plt.subplots(1, 2, num=15, figsize=(6, 2.5))
# axes15[0].scatter(
#     cathode_EIS_25degC.to_numpy()[:, 0],
#     cathode_EIS_25degC.to_numpy()[:, 1],
#     c="orange", s=5, label="25°C"
# )
axes15[0].scatter(
    cathode_EIS_30degC.to_numpy()[:, 1],
    - cathode_EIS_30degC.to_numpy()[:, 2],
    c="black", s=5, label="30°C"
)
axes15[0].scatter(
    cathode_EIS_40degC.to_numpy()[:, 1],
    - cathode_EIS_40degC.to_numpy()[:, 2],
    c="red", s=5, label="40°C"
)
axes15[0].scatter(
    cathode_EIS_50degC.to_numpy()[:, 1],
    - cathode_EIS_50degC.to_numpy()[:, 2],
    c="green", s=5, label="50°C"
)
axes15[0].scatter(
    cathode_EIS_60degC.to_numpy()[:, 1],
    - cathode_EIS_60degC.to_numpy()[:, 2],
    c="blue", s=5, label="60°C"
)
axes15[0].set_xlabel("Re(Z) ($\Omega$)")
axes15[0].set_ylabel("-Im(Z) ($\Omega$)")
# axes15[0].set_title("Positive electrode")
axes15[0].set_title("(a)")
# axes15[0].legend()

# axes15[1].scatter(
#     anode_EIS_25degC.to_numpy()[:, 0],
#     anode_EIS_25degC.to_numpy()[:, 1],
#     c="orange", s=5, label="25°C"
# )
axes15[1].scatter(
    anode_EIS_30degC.to_numpy()[:, 1],
    - anode_EIS_30degC.to_numpy()[:, 2],
    c="black", s=5, label="30°C"
)
axes15[1].scatter(
    anode_EIS_40degC.to_numpy()[:, 1],
    - anode_EIS_40degC.to_numpy()[:, 2],
    c="red", s=5, label="40°C"
)
axes15[1].scatter(
    anode_EIS_50degC.to_numpy()[:, 1],
    - anode_EIS_50degC.to_numpy()[:, 2],
    c="green", s=5, label="50°C"
)
axes15[1].scatter(
    anode_EIS_60degC.to_numpy()[:, 1],
    - anode_EIS_60degC.to_numpy()[:, 2],
    c="blue", s=5, label="60°C"
)
axes15[1].set_xlabel("Re(Z) ($\Omega$)")
axes15[1].set_ylabel("-Im(Z) ($\Omega$)")
# axes15[1].set_title("Negative electrode")
axes15[1].set_title("(b)")
axes15[1].legend()

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig15.png",
    dpi=300
)

# Figure 16 - Arrhenius plot
arrhenius_Rct = pd.read_csv(
    "~/LGM50/data/arrhenius_Rct.csv"
)

# arrhenius_T = arrhenius_Rct.to_numpy()[1:, 0]
# arrhenius_Rct_cathode = arrhenius_Rct.to_numpy()[1:, 1]
# arrhenius_Rct_anode = arrhenius_Rct.to_numpy()[1:, 2]

arrhenius_T = arrhenius_Rct.to_numpy()[:, 0]
arrhenius_Rct_cathode = arrhenius_Rct.to_numpy()[:, 1]
arrhenius_Rct_anode = arrhenius_Rct.to_numpy()[:, 2]

R = 8.314
F = 96485
Sp = 4.9706E-3
Sn = 5.7809E-3

arrhenius_j0_cathode = R/(Sp * F) * np.divide(arrhenius_T, arrhenius_Rct_cathode)/10
arrhenius_j0_anode = R/(Sn * F) * np.divide(arrhenius_T, arrhenius_Rct_anode)/10

print(arrhenius_T)
print(arrhenius_Rct_cathode)
print(arrhenius_Rct_anode)

fit_cathode = np.polyfit(
    1. / arrhenius_T,
    np.log(arrhenius_j0_cathode),
    deg=1
)
fit_anode = np.polyfit(
    1. / arrhenius_T,
    np.log(arrhenius_j0_anode),
    deg=1
)

plt.figure(num=16, figsize=(6, 4))
plt.semilogy(
    1. / arrhenius_T,
    arrhenius_j0_cathode,
    color="black", marker='o', markersize=5, linestyle="None",
    label="positive electrode"
)
plt.semilogy(
    1. / arrhenius_T,
    arrhenius_j0_anode,
    color="black", marker='x', markersize=5, linestyle="None",
    label="negative electrode"
)
plt.semilogy(
    1. / arrhenius_T,
    np.exp(fit_cathode[0] / arrhenius_T + fit_cathode[1]),
    color="blue",
)
plt.semilogy(
    1. / arrhenius_T,
    np.exp(fit_anode[0] / arrhenius_T + fit_anode[1]),
    color="blue",
)
plt.xlabel("1/T ($\mathrm{K}^{-1})$")
plt.ylabel("$j_0$ (mA $\mathrm{cm}^{-2}$)")
plt.ylim((1E-2, 1))
plt.legend()

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig16.png",
    dpi=300
)

print("Cathode: ", fit_cathode[0] * R)
print("Anode: ", fit_anode[0] * R)

# Figure S7 - EIS at room temperature
cathode_Rct_RT = pd.read_csv(
    "~/LGM50/data/cathode_Rct_RT.csv"
)
anode_Rct_RT = pd.read_csv(
    "~/LGM50/data/anode_Rct_RT.csv"
)

j0_cathode = R/(Sp * F) * 298.15 * np.divide(1, cathode_Rct_RT.to_numpy()[:,1])/10
j0_anode = R/(Sn * F) * 298.15 * np.divide(1, anode_Rct_RT.to_numpy()[:,1])/10

fig27, axes27 = plt.subplots(1, 2, num=27, figsize=(6, 2.5))
axes27[0].scatter(
    cathode_Rct_RT.to_numpy()[:,0],
    j0_cathode,
    c="black", s=5
)
axes27[0].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes27[0].set_ylabel("$j_0$ (mA $\mathrm{cm}^{-2}$)")
# axes27[0].set_title("Positive electrode")
axes27[0].set_title("(a)")

axes27[1].scatter(
    anode_Rct_RT.to_numpy()[:,0],
    j0_anode,
    c="black", s=5
)
axes27[1].set_xlabel("Capacity (mAh $\mathrm{cm}^{-2}$)")
axes27[1].set_ylabel("$j_0$ (mA $\mathrm{cm}^{-2}$)")
# axes27[1].set_title("Negative electrode")
axes27[1].set_title("(b)")

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/figS7.png",
    dpi=300
)


plt.show()