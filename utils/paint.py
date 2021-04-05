import matplotlib.pyplot as plt
import numpy


def plot_window_size(y, legend = 'load', save_path=None, is_show=True):
    x = range(0, len(y))
    plt.plot(x, y)
    plt.xlabel("Relative Timestamps (s)")
    plt.ylabel("Power (W)")
    plt.legend([legend])
    if save_path is not None:
        plt.savefig(save_path)
    if is_show:
        plt.show()
    plt.clf()

if __name__ == '__main__':
    test_y = range(0, 600)
    plot_window_size(test_y, "washing machine")
