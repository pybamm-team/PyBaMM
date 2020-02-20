import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt

OCP_an = pd.read_csv(
    "~/PyBaMM/input/parameters/lithium-ion/anodes/graphite_Chen2020/graphite_LGM50_ocp_Chen2020.csv",
    comment="#", 
).to_numpy()

OCP_cat = pd.read_csv(
    "~/PyBaMM/input/parameters/lithium-ion/cathodes/nmc_Chen2020/nmc_LGM50_ocp_Chen2020.csv",
    comment="#", 
).to_numpy()


def func_an(x, a, b, c, d, e, f, g, h, i, j, k, l):
    y = a * np.exp(-b * x) + c - d * np.tanh(e * (x - f)) - g * np.tanh(h * (x - i)) - j * np.tanh(k * (x - l))
    return y

# def func_cat(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n):
#     y = - a * x + b - 0*c * np.tanh(d * (x - e)) - f * np.tanh(g * (x - h)) - i * np.tanh(j * (x - k)) - l * np.tanh(m * (x - n))
#     return y

def func_cat(x, a, b, c, d, e, f, g, h, i, j, k):
    y = - a * x + b - c * np.tanh(d * (x - e)) - f * np.tanh(g * (x - h)) - i * np.tanh(j * (x - k))
    return y

p0_an = [1.3062, 18.1711, 0.192382, 0.044, 33.92, 0.1612, 0.0363, 17.59, 0.3367, 0.0204, 52.34, 0.6422]
# p0_cat = [0.62, 4.12, 0.22, 16.49, -0.01, 0.1, 9.77, 0.39, 4.09, 5.94, 0.14, -4.43, 5.45, 0.13]
# p0_cat = [0.95, 5, 0.23, 25.37, 0.25, 0.1, 15.02, 0.51, 4.09, 9.13, 0.35, -4.43, 8.37, 0.34] # Ivan's after transformation

# p0_cat = [0.7514, 4.646, 0.5871, 12.62, 0.2277, 0.1156, 11.17, 0.5196, 4.634, 11.21, 0.3595, -4.939, 10.67, 0.3555]

p0_cat = [0.7514, 4.646, 0.1156, 11.17, 0.5196, 4.634, 11.21, 0.3595, -4.939, 10.67, 0.3555]
params_an, params_an_covariance = optimize.curve_fit(
    func_an, OCP_an[10:, 0], OCP_an[10:, 1], 
    p0=p0_an, maxfev=10000
)

params_cat, params_cat_covariance = optimize.curve_fit(
    func_cat, OCP_cat[:-5, 0], OCP_cat[:-5, 1], 
    p0=p0_cat, maxfev=100000
)

res_an = OCP_an[:,1] - func_an(OCP_an[:, 0], params_an[0], params_an[1], params_an[2], params_an[3], params_an[4], params_an[5], params_an[6], params_an[7], params_an[8], params_an[9], params_an[10], params_an[11])
ss_res_an = np.sum(res_an**2)
ss_tot_an = np.sum((OCP_an[:, 1] - np.mean(OCP_an[:, 1]))**2)
R_an = 1 - (ss_res_an/ss_tot_an)

print(params_an)
print(R_an)

plt.figure(figsize=(6, 4))
plt.scatter(OCP_an[:, 0], OCP_an[:, 1], label='Data', marker="x")
plt.plot(
    OCP_an[:, 0], 
    func_an(OCP_an[:, 0], params_an[0], params_an[1], params_an[2], params_an[3], params_an[4], params_an[5], params_an[6], params_an[7], params_an[8], params_an[9], params_an[10], params_an[11]),
    label='Fitted function',
    color='black'
)

res_cat = OCP_cat[:,1] - func_cat(OCP_cat[:, 0], params_cat[0], params_cat[1], params_cat[2], params_cat[3], params_cat[4], params_cat[5], params_cat[6], params_cat[7], params_cat[8], params_cat[9], params_cat[10])#, params_cat[11], params_cat[12], params_cat[13])
ss_res_cat = np.sum(res_cat**2)
ss_tot_cat = np.sum((OCP_cat[:, 1] - np.mean(OCP_cat[:, 1]))**2)
R_cat = 1 - (ss_res_cat/ss_tot_cat)

print(params_cat)
print(R_cat)

plt.figure(figsize=(6, 4))
plt.scatter(OCP_cat[:-5, 0], OCP_cat[:-5, 1], label='Data', marker="x")
plt.plot(
    OCP_cat[:-5, 0], 
    func_cat(OCP_cat[:-5, 0], params_cat[0], params_cat[1], params_cat[2], params_cat[3], params_cat[4], params_cat[5], params_cat[6], params_cat[7], params_cat[8], params_cat[9], params_cat[10]),#, params_cat[11], params_cat[12], params_cat[13]),
    label='Fitted function',
    color='black'
)

plt.legend(loc='best')

plt.show()