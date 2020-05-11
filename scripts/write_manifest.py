import os
import pathlib
import glob

pybamm_data = []
for file_ext in ["*.csv", "*.py", "*.md"]:
    # Get all the files ending in file_ext in pybamm/input dir.
    # list_of_files = [
    #    'pybamm/input/drive_cycles/car_current.csv',
    #    'pybamm/input/drive_cycles/US06.csv',
    # ...
    list_of_files = glob.glob("pybamm/input/**/" + file_ext, recursive=True)

    # Add these files to pybamm_data.
    # The path must be relative to the package dir (pybamm/), so
    # must process the content of list_of_files to take out the top
    # pybamm/ dir, i.e.:
    # ['input/drive_cycles/car_current.csv',
    #  'input/drive_cycles/US06.csv',
    # ...
    pybamm_data.extend(
        [os.path.join(*pathlib.Path(filename).parts[1:]) for filename in list_of_files]
    )
pybamm_data.append("./version")
pybamm_data.append("./CITATIONS.txt")

with open("MANIFEST.in", "w") as manifest:
    for data_file in pybamm_data:
        manifest.write(data_file + "\n")
