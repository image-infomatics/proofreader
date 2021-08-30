import numpy as np
import math
from matplotlib import pyplot as plt


def make_histogram(data, bins=20, title='', xlabel='', ylabel='Counts', logscale=False):

    bins = np.linspace(math.ceil(min(data)),
                       math.floor(max(data)),
                       bins)  # fixed number of bins

    plt.xlim([min(data)-5, max(data)+5])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if logscale:
        plt.yscale('log', nonposy='clip')

    plt.show()
