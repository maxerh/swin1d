import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import utils.helpers as hp

class Color:
    background = "#ffffff"
    text = "#000000"
    anomaly = "#fb5f5f"


class Visualizer_base:
    def __init__(self, config):
        self.config = config
        self.fig = plt.figure(figsize=(15, 10), facecolor=Color.background)
        self.ax_dict = {}
        self.draw_plot()

    def save_pgf(self, name):
        plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'  # load times roman font
        plt.rcParams['font.family'] = 'serif'  # use serif font as default
        plt.rcParams['text.usetex'] = True  # enable LaTeX rendering globally
        self.fig.savefig(f'{name}.pgf', format='pgf')

    def save_tikz(self, name):
        import tikzplotlib
        tikzplotlib.save(f"{name}.tex")

    def save_fig(self, name):
        self.fig.savefig(f"{name}.png", facecolor=self.fig.get_facecolor(), edgecolor='none')

    def save_np(self, name, data, s, c):
        file = f"{name}.npy"
        with open(file, 'wb') as f:
            for k in self.ax_dict.keys():
                ts = data[k][s:s+1024, c[k]]
                np.save(f, ts)

    def show_plot(self):
        """Show plot"""
        plt.show()

    def draw_plot(self):
        """Draw the figure"""
        plt.draw()
        plt.pause(0.1)

    def close_plot(self):
        """Close the plot"""
        plt.close(self.fig)

    def clear_axes(self):
        """Clear the axes"""
        for k, ax in self.ax_dict.items():
            ax.clear()

    def plot_legend(self):
        for _, ax in self.ax_dict.items():
            ax.legend(facecolor=Color.background, edgecolor=Color.text, labelcolor=Color.text)

    @staticmethod
    def adjust_limits(low, high):
        if low < 0:
            low = 1.1*low
        else:
            low = 0.9*low
        if high >0:
            high = 1.1*high
        else:
            high = 0.9*high
        return low, high

