import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import json


def train_model(csvpath, pkl_filename, choice_number=None, grid_number=None):
    input_data = np.genfromtxt(csvpath, delimiter=',')

    if bool(choice_number):
        input_data = input_data[np.random.choice(input_data.shape[0], choice_number, replace=False)]

    if not bool(grid_number):
        grid_number = 1000

    regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    regressor.fit(input_data[:, 0:-1], input_data[:, -1])

    max_values = np.amax(input_data, axis=0)

    min_values = np.amin(input_data, axis=0)

    json_object = {"ranges": []}

    for i in range(input_data.shape[1] - 1):

        min_value = min_values[i]

        max_value = max_values[i]

        json_object['ranges'].append({"start": min_value, "end": max_value, "num": grid_number})

    with open(os.path.splitext(pkl_filename)[0] + '.json', 'w') as jsonfile:
        json.dump(json_object, jsonfile)

    with open(pkl_filename, 'wb') as file:
        pickle.dump(regressor, file)


if __name__ == "__main__":

    csvpath = input("Path to CSV data file: ")

    bare_path = os.path.splitext(csvpath)[0]

    default_pkl_filename = bare_path + '.pkl'

    pkl_filename = input("Path to output model (leave empty to use {0}): ".format(default_pkl_filename))

    if not bool(pkl_filename):
        pkl_filename = default_pkl_filename

    choice_number = input("Number of values from the CSV to use (leave empty to use all): ")

    grid_number = input("Number of values for each row / column of the grid (leave empty to use 1000): ")

    if not bool(choice_number):
        choice_number = None

    else:
        choice_number = int(choice_number)

    if not bool(grid_number):
        grid_number = None

    else:
        grid_number = int(grid_number)

    train_model(csvpath, pkl_filename, choice_number=choice_number, grid_number=grid_number)


