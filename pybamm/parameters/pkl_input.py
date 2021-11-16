import numpy as np
import pickle
import os
import json


class PybammStandardParameterClass:

    def __init__(self, pkl_filename):

        with open(pkl_filename, 'rb') as modelfile:
            self.model = pickle.load(modelfile)

        self.bare_path = os.path.splitext(pkl_filename)[0]

        json_path = self.bare_path + '.json'

        with open(json_path, 'r') as jsonfile:
            self.json_data = json.load(jsonfile)

    def get_interpolation_parameters(self):

        x_ = []

        for it in self.json_data["ranges"]:

            x_.append(np.linspace(
                it["start"],
                it["end"],
                it["num"]
            ))

        X = list(np.meshgrid(*x_))

        # X_f = [el.flatten() for el in X]

        x = np.column_stack([el.reshape(-1, 1) for el in X])

        y = self.model.predict(x)

        Y = y.reshape(*[len(el) for el in x_])

        return x_, Y

        # return X_f, Y


