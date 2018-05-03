import numpy as np
import matplotlib.pyplot as mpl

def plot_flux(flux, title, x_label, y_label, label_1, label_2):
    mpl.plot(flux[:, 0], label=label_1)
    mpl.plot(flux[:, 1], label=label_2)
    mpl.legend()
    mpl.title(title)
    mpl.xlabel(x_label)
    mpl.ylabel(y_label)
    mpl.show()

def plot_1d_array(flux, title, x_label, y_label, label_1):
    mpl.plot(flux[:], label=label_1)
    mpl.legend()
    mpl.title(title)
    mpl.xlabel(x_label)
    mpl.ylabel(y_label)
    mpl.show()

def pin_cell_average_flux(flux):
    pin_cell = np.reshape(flux,(16,8))
    pin_cell_avg = np.zeros(16)
    for i, x in enumerate(pin_cell):
        pin_cell_avg[i] = sum(x) / 8
    return pin_cell_avg


def flux_histogram(flux, title, x_label, y_label, label_1, label_2):
    N = len(flux)
    fast = flux[:, 0]
    thermal = flux[:, 1]
    ind = np.arange(0, N)
    width = 0.5
    p1 = mpl.bar(ind, thermal, width)
    p2 = mpl.bar(ind, fast, width, bottom=thermal)
    mpl.title(title)
    mpl.ylabel(y_label)
    mpl.xlabel(x_label)
    mpl.legend((p2[0], p1[0]), (label_1, label_2))
    mpl.show()

    return