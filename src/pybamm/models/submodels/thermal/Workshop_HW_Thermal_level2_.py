# %%
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pybamm.experiment.plot_cylinder as plt_
from pybamm.models.submodels.interface.open_circuit_potential.RK_Polynomial_OCP import RK_Open_Circuit_Potential

# %%
model = pybamm.lithium_ion.SPMe(build=False)

# %%
def my_rk_ocp_positive(x, T):
    kb = 1.380649e-23
    m0 = 3.8724339986937832
    coeffs = [-0.38545242, 0.07207664, 0.08256196, -0.27144603, -0.41604248, 1.38708789, 1.11072656, -3.67485474, -1.02940457, 4.24441946, 0.24045833, -1.74123012]
    b = 0.88124821
    delx = 0.10572341
    x = 1-x
    x = b*x+delx
    func = 0
    for i, a in enumerate(coeffs):
        # For i==0, the second term becomes zero (avoid potential issues with negative exponents)
        if i == 0:
            term = a * ((1 - 2*x)**(i + 1))
        else:
            term = a * (((1 - 2*x)**(i + 1)) - (2*i*x*(1-x)*((1-2*x)**(i-1))))
        func += term
    return (kb * T * np.log(x/(1-x)) + func + m0)

def my_rk_ocp_negative(x, T):
    kb = 1.380649e-23
    m0 = 0.11787222863387316
    coeffs = [0.10686728, 0.04120991, -0.0697988, 0.27019182, 1.35141614, -1.64799623,-5.90470322, 5.4219514, 13.50384686, -7.70496124, -14.79406063, 4.03252355, 6.34233454]
    b = 0.67338648
    delx = 0.00612092
    x = b*x+delx
    func = 0
    for i, a in enumerate(coeffs):
        # For i==0, the second term becomes zero (avoid potential issues with negative exponents)
        if i == 0:
            term = a * ((1 - 2*x)**(i + 1))
        else:
            term = a * (((1 - 2*x)**(i + 1)) - (2*i*x*(1-x)*((1-2*x)**(i-1))))
        func += term
    return (kb * T * np.log(x/(1-x)) + func + m0)

# %%
class MyThermalModel (pybamm.models.submodels.thermal.BaseThermal):
    def __init__(self, param, options=None, x_average=False):
        super().__init__(param, options=options, x_average=x_average)

    def get_fundamental_variables(self):
        T_cylinder = pybamm.Variable("Cylinder temperature", domain = "full cell")
        T = pybamm.boundary_value(T_cylinder,'left')
        T_x_av = pybamm.PrimaryBroadcast(T, ["current collector"])
        T_dict = {
            "negative current collector": T_x_av,
            "positive current collector": T_x_av,
            "x-averaged cell": T_x_av,
            "volume-averaged cell": T,
        }
        for domain in ["negative electrode", "separator", "positive electrode"]:
            T_dict[domain] = pybamm.PrimaryBroadcast(T_x_av, domain)

        variables = self._get_standard_fundamental_variables(T_dict)

        # define parameters
        R = pybamm.Parameter("Cell radius [m]")
        k = pybamm.Parameter("Heat conductivity [W/m2-K]")
        h = pybamm.Parameter("Convection heat transfer coefficient [W/m2-K]")
        T0 = pybamm.Parameter("Initial temperature [K]")
        Height = pybamm.Parameter("Cell height [m]")
        Tamb = pybamm.Parameter("Ambient temperature [K]")
        m = pybamm.Parameter("mass [kg]")
        Cp = pybamm.Parameter("Heat Capacity of the cell")
        
        
        # update variables
        variables[R.name] = R
        variables [k.name] = k
        variables[h.name] = h
        variables[Cp.name] = Cp
        variables[T0.name] = T0
        variables[Height.name] = Height
        variables[Tamb.name] = Tamb
        variables[m.name] = m
        variables[T_cylinder.name] = T_cylinder
            
        return variables

            
    def get_coupled_variables(self,variables):
        variables.update(self._get_standard_coupled_variables(variables))

        #std_vars = self._get_standard_coupled_variables(variables)
        # Remove conflicting keys if they exist
        #for key in ["Throughput capacity [A.h]"]:
            #if key in std_vars:
                #del std_vars[key]
        #variables.update(std_vars)

        T_cylinder = variables["Cylinder temperature"]
        #T=variables["Volume-averaged cell temperature [K]"]
        R = variables["Cell radius [m]"]
        k = variables["Heat conductivity [W/m2-K]"]
        h = variables["Convection heat transfer coefficient [W/m2-K]"]
        Height = variables["Cell height [m]"]
        Tamb = variables["Ambient temperature [K]"]
        m = variables["mass [kg]"]

        # derived parameters
        T_s = pybamm.surf(T_cylinder)
        A = 2 * 3.14 * R * Height
        V = 3.14 * (R**2) * Height
        ro = m/V 
        Q_cool=h*(T_s-Tamb)

        variables.update({
            "Cooling [W]": Q_cool,
            "Area [m^2]": A,
            "Volume [m^3]": V,
            "density [kg/m^3]": ro,
            "Surface temperature [K]": T_s
        })

        return variables

    def set_rhs(self,variables):

        T_cylinder = variables["Cylinder temperature"]
        Cp = variables["Heat Capacity of the cell"]
        ro = variables["density [kg/m^3]"]
        k = variables["Heat conductivity [W/m2-K]"]
        V = variables["Volume [m^3]"]
        Q_heat=variables["Total heating [W]"]

        dTdt = (1/(ro * Cp))*pybamm.div(k*pybamm.grad(T_cylinder)) + Q_heat / (V * ro * Cp)
        self.rhs = {T_cylinder: dTdt}

    def set_initial_conditions(self,variables):
        T0 = variables["Initial temperature [K]"]
        T_cylinder=variables["Cylinder temperature"]
        self.initial_conditions={T_cylinder: T0}  

    def  set_boundary_conditions(self,variables):
        T_cylinder = variables["Cylinder temperature"]
        Q_cool = variables["Cooling [W]"]
        k = variables["Heat conductivity [W/m2-K]"]
        
        # boundary conditions
        lbc = pybamm.Scalar(0)
        rbc = - Q_cool / k
        self.boundary_conditions = {T_cylinder: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}

    def output_variables(self,variables):   
        T_cylinder = variables["Cylinder temperature"]
        T_s = pybamm.surf(T_cylinder)
        Q_heat=variables["Total heating [W]"]
        R = variables["Cell radius [m]"]
        
        # define output variables
        self.variables = {
            "Time [s]": pybamm.t,
            "Total heating [W]": Q_heat,
            "Surface temperature [K]": T_s,
            "Cylinder temperature":T_cylinder,
            "Cell radius [m]": R,
        }  

# %%
# here 1 is for cell model while 2 is for thermal model
# Parameters
param1=model.default_parameter_values
parameter_values = pybamm.ParameterValues("Chen2020")
param1.update(parameter_values, check_already_exists=False)
param1.update({"Positive electrode RK_OCP [V]": my_rk_ocp_positive,
"Negative electrode RK_OCP [V]": my_rk_ocp_negative, "Current function [A]": 7.2}, check_already_exists=False)

param2 = pybamm.ParameterValues({"Heat conductivity [W/m2-K]": 0.2,
"Convection heat transfer coefficient [W/m2-K]": 20, 
"Initial temperature [K]": 300, "Cell height [m]": 0.07, 
"Cell radius [m]": 0.0105, "Ambient temperature [K]": 300, 
"Heat Capacity of the cell": 1200, "mass [kg]":0.068})

# Generate models
#submodel=MyThermalModel(model.param)
model.submodels["thermal"] = MyThermalModel(model.param)
model.build_model()

# Create geometry and mesh
geometry1 = model.default_geometry
submesh_types1 = model.default_submesh_types
var_pts1 = model.default_var_pts
spatial_methods1 = model.default_spatial_methods

R = pybamm.Parameter("Cell radius [m]")
# define geometry
r_cell = pybamm.SpatialVariable(
    "r_cell", domain=["full cell"], coord_sys="cylindrical"
)
geometry2 = {
    "full cell": {r_cell: {"min": pybamm.Scalar(0), "max": R}}
}
submesh_types2 = {"full cell": pybamm.Uniform1DSubMesh}
var_pts2 = {r_cell: 10}

# merge geometry and parameters fro cell and thermal models
geometry = {**geometry1, **geometry2}
submesh_types = {**submesh_types1, **submesh_types2}
var_pts = {**var_pts1,**var_pts2}
param1.update(param2,check_already_exists = False)
param = param1

# process param and geometry
param.process_geometry(geometry)
mesh=pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {**spatial_methods1, "full cell": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)

param.process_model(model)
disc.process_model(model)

# %%
# solve
solver=pybamm.ScipySolver()
t_eval=np.linspace(0, 7200, 100)
solution= solver.solve(model, t_eval)

# %%
# plot
solution.plot(["Surface temperature [K]", "Total heating [W]", "Cylinder temperature","Voltage [V]" ])


# %%
# Extract variables
T_cyl = solution ["Cylinder temperature"]
T_cyl = T_cyl.entries

# %%
R_ = solution["Cell radius [m]"].entries[0]
time = solution["Time [s]"].entries[-1]
def plot2D_cell(T_cyl, n):
    """
    Function to plot 'n' static temperature contour plots of a full cylinder
    at 'n' equidistant time steps.

    Parameters:
    - T_cyl: np.ndarray of shape (N_radial, N_time)
    - n: int, number of plots to generate (must be <= N_time)
    """
    # Get the number of radial and time points
    N_radial, N_time = T_cyl.shape  

    # Ensure n does not exceed the number of available time steps
    if n > N_time:
        raise ValueError("n cannot be greater than the number of time steps in T_cyl.")

    # Define radial and angular grids
    r = np.linspace(0, 1, N_radial)  # Radial positions
    theta = np.linspace(0, 2 * np.pi, 100)  # Full circle (100 angular points)
    R, Theta = np.meshgrid(r, theta, indexing='ij')

    # Convert to Cartesian coordinates for plotting
    X = R_ * 1000 * R * np.cos(Theta)
    Y = R_ * 1000 * R * np.sin(Theta)

    # Select n time steps at equal intervals
    time_steps = np.linspace(0, N_time - 1, n, dtype=int)

    # Create figure with 'n' subplots
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))  # Adjust figure size dynamically

    for i, t in enumerate(time_steps):
        # Generate temperature data for the current time step
        temperature_data = np.tile(T_cyl[:, t].reshape(N_radial, 1), (1, 100))

        # Get min and max for this specific time step
        T_min, T_max = np.min(T_cyl[:, t]), np.max(T_cyl[:, t])

        # Avoid ValueError by ensuring levels are strictly increasing
        if T_min == T_max:
            T_max += 1e-6  # Small numerical adjustment

        # Create contour plot with dynamically scaled color mapping
        contour = axes[i].contourf(X, Y, temperature_data, cmap='coolwarm', levels=np.linspace(T_min, T_max, 10))

        # Formatting
        axes[i].set_xlim(-R_*1000, R_*1000)
        axes[i].set_ylim(-R_*1000, R_*1000)
        axes[i].set_aspect('equal')
        axes[i].set_title(f"Time Step {int(t*time/N_time)}")

        # Add colorbar for each subplot
        cbar = plt.colorbar(contour, ax=axes[i], format="%.1f")
        cbar.set_label("Temperature")

    plt.tight_layout()
    plt.show()
    

# %%
# plot n number of plots over time
plt_.plot2D_cell(T_cyl,3,R_,time)