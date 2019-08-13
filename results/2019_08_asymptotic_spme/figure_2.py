#
# Figure 2: DFN, SPMe, SPM voltage comparison
#
import pybamm
import numpy as np
import os

generate_plots = False
export_data = True


a = np.asarray(([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

if generate_plots:
    print("hi")

if export_data:
    directory_path = "results/2019_08_asymptotic_spme/data/figure_2"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    np.savetxt(directory_path + "/figure_2.csv", a, delimiter=",")

