using ModelingToolkit, DiffEqOperators

# # 'line' -> x
# @parameters t
# # 'x' -> u1
# @variables u1(..)
# # @variables u1(t)
# @derivatives Dt'~t

# eqs = [
#    Dt(u1(t)) ~ -1,
#    # Dt(u1) ~ -1,
# ]

# # ics = [u1(0) ~ 1]
# ics = [u1(t) => 1]

# sys = ODESystem(eqs, t)

# # Convert the PDE problem into an ODE problem
# prob = ODEProblem(sys, ics, (0.0, 10.0))

# # Solve ODE problem
# using OrdinaryDiffEq
# sol = solve(prob,Tsit5(),saveat=0.1)
# 'line' -> x
@parameters t x
# 'x' -> u1
@variables u1(..)
@derivatives Dt'~t
@derivatives Dx'~x

# 'x' equation
cache_4964620357949541853 = Dx(u1(t,x))
cache_m2869545278201409051 = Dx(cache_4964620357949541853)

eqs = [
   Dt(u1(t,x)) ~ cache_m2869545278201409051,
]

ics_bcs = [u1(0,x) ~ -x*(x-1)*sin(x),
u1(t,0) ~ 0.0,
u1(t,1) ~ 0.0]

t_domain = IntervalDomain(0.0, 10.0)
x_domain = IntervalDomain(0.0, 1.0)

domains = [
   t in t_domain,
   x in x_domain,
]
ind_vars = [t, x]
dep_vars = [u1]

pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)

# Method of lines discretization
dx = 0.1
order = 2
discretization = MOLFiniteDifference(dx,order)

# Convert the PDE problem into an ODE problem
prob = DiffEqOperators.discretize(pde_system,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,saveat=0.1)