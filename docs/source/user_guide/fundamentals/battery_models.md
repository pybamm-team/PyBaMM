# Battery Models

References for the battery models used in PyBaMM simulations can be found calling
```python
pybamm.print_citations()
```
However, each model can be defined in slightly different ways depending on the paper. For this reason, in this page we state some of the standard battery models with the exact definition used in PyBaMM. Familiarising with the theory of battery model is fundamental before doing the PyBaMM tutorials.

## Lithium-ion battery models

The standard models for lithium-ion batteries can be classified in a hierarchical structure, from simple to complex. Here, as shown in the figure, we focus on the SPM, SPMe and DFN models. This hierarchy is discussed in a lot more detail in the review article by [Brosa Planella et al. (2022)](https://doi.org/10.1088/2516-1083/ac7d31), and also explains in detail how these models are derived.

<img title="Lithium-ion battery model hierarchy" alt="A diagram showing the hierarchy of lithium-ion battery models, from simple (SPM) to complex (DFN). The diagram also includes what each model captures and misses compared to the others." src="../../../_static/model_hierarchy.png">

These models impose conservation of lithium and charge, so the variables we are interested in are the lithium concentration in the solid and electrolyte phases, $c_k$ and $c_\mathrm{e}$ respectively, and the electric potential $\phi_k$ and $\phi_\mathrm{e}$ in the solid and electrolyte phases, respectively. The subscript $k \in \{\mathrm{n}, \mathrm{p}\}$ denotes the negative and positive electrodes, respectively.

**Important remark:** these models account for the electrochemistry of the battery, and thus not include the thermal and degradation effects. These effects can be added to the electrochemical models as additional equations (and through the PyBaMM model options).

The parameters involved in these models are defined in the following table (where parameters with $k \in \{\mathrm{n}, \mathrm{s}, \mathrm{p}\}$ may have different values for each electrode and separator):

| Symbol | Description | Units |
|:------:|:-----------:|:-----:|
| $R_k$ | Particle radius | m |
| $L_k$ | Electrode/separator thickness | m |
| $D_k (c_k)$ | Electrode particle diffusivity | m $^2$ s $^{-1}$ |
| $\sigma_k$ | Electrode conductivity | S m $^{-1}$ |
| $U_k (c_k)$ | Open-circuit potential | V |
| $j_{0 k} (c_k, c_\mathrm{e})$ | Exchange current density | A m $^{-2}$ |
| $\varepsilon_{\mathrm{s}, k}$ | Solid phase volume fraction | - |
| $\varepsilon$ | Porosity | - |
| $\mathcal{B}$ | Transport efficiency | - |
| $D_\mathrm{e}(c_\mathrm{e})$ | Electrolyte diffusivity | m $^2$ s $^{-1}$ |
| $\sigma_\mathrm{e}(c_\mathrm{e})$ | Electrolyte conductivity | S m $^{-1}$ |
| $t^+(c_\mathrm{e})$ | Transference number | - |
| $c_\mathrm{e0}$ | Initial electrolyte concentration | mol m $^{-3}$ |
| $T$ | Temperature | K |
| $i_\mathrm{app}$ | Applied current density | A m $^{-2}$ |

The surface area per unite volume $a_k$ is defined as $a_k = 3 \varepsilon_{\mathrm{s}, k}/ R_k$. The two physical constants appearing in the models are the Faraday constant is $F$, the gas constant $R$.

Note that not all models use all of these parameters. For example, the SPM only uses a subset.

### Single particle model (SPM)
The SPM is the simplest of the three models considered here. It assumes that the active material particles behave similarly so they can be described by an averaged particle. The model considers two particles: one for the positive electrode and one for the negative electrode, where we model the lithium concentration as a function of space and time, denoted as $c_\mathrm{p}$ and $c_\mathrm{n}$ respectively. Two (independent) diffusion equations need to be solved for the two particles, and any additional quantities (such as the cell voltage) are computed from explicit expressions.


In PyBaMM, the Single Particle Model is defined as follows:
$$\begin{align}
\frac{\partial c_\mathrm{n}}{\partial t} &= \frac{1}{r^2} \frac{\partial}{\partial r} \left (r^2 D_\mathrm{n} (c_\mathrm{n}) \frac{\partial c_\mathrm{n}}{\partial r} \right), & \text{ in } \quad 0 < r < R_\mathrm{n},\\
\frac{\partial c_\mathrm{p}}{\partial t} &= \frac{1}{r^2} \frac{\partial}{\partial r} \left (r^2 D_\mathrm{p} (c_\mathrm{p}) \frac{\partial c_\mathrm{p}}{\partial r} \right), & \text{ in } \quad 0 < r < R_\mathrm{p},
\end{align}$$

with boundary conditions
$$\begin{align}
\frac{\partial c_\mathrm{n}}{\partial r} &= 0, & \text{ at } \quad r = 0,\\
-D_\mathrm{p} (c_\mathrm{n}) \frac{\partial c_\mathrm{n}}{\partial r} &= \frac{i_\mathrm{app}(t)}{a_\mathrm{n} L_\mathrm{n} F}, & \text{ at } \quad r = R_\mathrm{n},\\
\frac{\partial c_\mathrm{p}}{\partial r} &= 0, & \text{ at } \quad r = 0,\\
-D_\mathrm{p} (c_\mathrm{p}) \frac{\partial c_\mathrm{p}}{\partial r} &= -\frac{i_\mathrm{app}(t)}{a_\mathrm{p} L_\mathrm{p} F}, & \text{ at } \quad r = R_\mathrm{p},
\end{align}$$

and initial conditions
$$\begin{align}
c_\mathrm{n}(r, 0) &= c_\mathrm{n0}(r), & \text{ at } \quad t = 0,\\
c_\mathrm{p}(r, 0) &= c_\mathrm{p0}(r), & \text{ at } \quad t = 0.
\end{align}
$$

The voltage can then be computed from $c_\mathrm{n}$ and $c_\mathrm{p}$ as
$$\begin{align}
V(t) &= U_\mathrm{p}(c_\mathrm{p}(R_\mathrm{p}, t)) - U_\mathrm{n}(c_\mathrm{n}(R_\mathrm{n}, t)) \\
& \quad - \frac{2 R T}{F} \mathrm{arcsinh} \left(\frac{i_\mathrm{app}(t)}{a_\mathrm{n} L_\mathrm{n} j_{0\mathrm{n}}} \right) - \frac{2 R T}{F} \mathrm{arcsinh} \left(\frac{i_\mathrm{app}(t)}{a_\mathrm{p} L_\mathrm{p} j_{0\mathrm{p}}} \right),\nonumber
\end{align}$$

where $j_{0\mathrm{n}}(c_\mathrm{n})$ and $j_{0\mathrm{p}}(c_\mathrm{p})$ are the interfacial current densities at the negative and positive electrodes, respectively, and can be defined in the parameter set by the user.

### Single particle model with electrolyte (SPMe)

### Doyle-Fuller-Newman model (DFN)

## Review Articles

[Review of physics-based lithium-ion battery models](https://doi.org/10.1088/2516-1083/ac7d31)

[Review of parameterisation and a novel database for Li-ion battery models](https://doi.org/10.1088/2516-1083/ac692c)

## Model References

### Lithium-Ion Batteries

[Doyle-Fuller-Newman model](https://doi.org/10.1149/1.2221597)

[Single particle model](https://doi.org/10.1149/2.0341915jes)


### Lead-Acid Batteries

[Isothermal porous-electrode model](https://doi.org/10.1149/2.0301910jes)

[Leading-Order Quasi-Static model](https://doi.org/10.1149/2.0441908jes)
