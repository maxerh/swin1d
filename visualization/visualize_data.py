import numpy as np
import matplotlib.pyplot as plt

from visualization.visualizer_base import Visualizer_base, Color

class VisualizerData(Visualizer_base):
    def __init__(self, config, n_plots=5, column_names=None):
        super().__init__(config)
        self.n_plots = n_plots
        if column_names is None:
            column_names = list(np.arange(n_plots).astype(str))

        self.column_names = column_names
        self.create_subplots(n_plots)

    def create_subplots(self, n_plots):
        assert len(self.column_names) == n_plots, f"plot names and number of subplots do not match ({len(self.column_names)} vs. {n_plots})"

        for i, name in enumerate(self.column_names):
            self.ax_dict[name] = self.fig.add_subplot(n_plots, 1, i + 1,
                                                      sharex=self.ax_dict[self.column_names[0]] if i > 0 else None)

        for _, ax in self.ax_dict.items():
           ax.set_facecolor(Color.background)
           ax.tick_params(axis='x', colors=Color.text)
           ax.tick_params(axis='y', colors=Color.text)
           ax.spines['bottom'].set_color(Color.text)
           ax.spines['top'].set_color(Color.text)
           ax.spines['left'].set_color(Color.text)
           ax.spines['right'].set_color(Color.text)

    def update_subplot(self, data, colorindex=0):
        for key in data.columns:
            if key == 'time':
                continue
            c = key.split("_")[0]
            source = key.split("_")[1:]
            s = '_'.join(source) if isinstance(source, list) else source
            #self.ax_dict[key].plot(data['time']-min(data['time']), data[key], color=Color.colordict[s], label=s)
            self.ax_dict[key].plot(data['time']-min(data['time']), data[key], color=Color.colorlist[colorindex], label=s)
            self.ax_dict[key].set_title(c)
            #self.ax_dict[key].set_xlim(0,5)
        plt.subplots_adjust(hspace=0.5)

    def plot_channels_from_dict(self, data):
        """
        Plotting channels in dictionary
        :param data:
        """
        fig, ax = plt.subplots(len(data), 1, sharex=True)
        for idx, key in enumerate(data):
            ax[idx].plot(data[key]['time'], data[key][key], label=key)
            ax[idx].set_title(key)
        plt.show()

    def plot_channels_from_df(self, data):
        """
        Plotting channels in dataframe
        :param data:
        """
        fig, ax = plt.subplots(len(data.columns) - 1, 1, sharex=True)
        idx = 0
        for key in data.columns:
            if key == 'time':
                continue
            ax[idx].plot(data['time'], data[key], label=key)
            ax[idx].set_title(key)
            idx += 1
        plt.show()
