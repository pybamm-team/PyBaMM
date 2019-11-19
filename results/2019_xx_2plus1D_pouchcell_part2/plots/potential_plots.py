import matplotlib.pyplot as plt


def plot_yz_potential(spmecc=None, reduced=None, full=None):

    num_of_subplots = 0

    if spmecc:
        num_of_subplots += 1
    if reduced:
        num_of_subplots += 1
    if full:
        num_of_subplots += 1

    fig, ax = plt.subplots(num_of_subplots)

    subplot_idx = 0

    if spmecc:
        y = spmecc[""]
        ax[subplot_idx].plot(spmecc["Negative current collector potential [V]"])

