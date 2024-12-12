import math
import numpy as np
import utils.helpers as hp
from visualization.visualizer_base import Visualizer_base, Color


class VisualizerTraining(Visualizer_base):
    def __init__(self, config, n_plots=5, column_names=None, plot_loss=True):
        super().__init__(config)
        self.config = config
        self.n_plots = n_plots
        self.dataset = config['data']['dataset']
        self.entity = config['data']['entity']
        if column_names is None:
            column_names = list(np.arange(n_plots).astype(str))

        self.column_names = column_names
        self.plot_loss = plot_loss
        self.create_subplots(n_plots)

    def create_subplots(self, n_plots):
        assert len(self.column_names) == n_plots, f"plot names and number of subplots do not match ({len(self.column_names)} vs. {n_plots})"
        n_plots = n_plots + 1 if self.plot_loss else n_plots
        for i, name in enumerate(self.column_names):
            self.ax_dict[name] = self.fig.add_subplot(n_plots, 1, i+1,
                                                      sharex=self.ax_dict[self.column_names[0]] if i > 0 else None)
        if self.plot_loss:
            self.ax_dict['loss'] = self.fig.add_subplot(n_plots, 1, n_plots)

    def update_subplot(self, epoch: int, subplotcount, signal: list, style: list = ['--'], y=None):
        """


        :param epoch:
        :param signal: list with true and predicted signal
        :param subplotcount:
        :param style:
        :param y:
        :return:
        """
        self.fig.suptitle(f"{self.dataset}-{self.entity} - epoch {epoch}", c=Color.text)
        if isinstance(subplotcount, int):
            name = self.column_names[subplotcount]
        elif isinstance(subplotcount, str):
            name = subplotcount
        else:
            raise NotImplementedError(f"don't know axis {subplotcount}")
        ax = self.ax_dict[name]
        ax.clear()
        ax.set_title(f"{name}")
        # self.set_ax_lims(name, ax)
        ax.grid(color=Color.text)
        if y is not None:
            consec_ano = hp.find_segments(y.numpy())
            for i in consec_ano:
                ax.axvspan(i[0], i[1], color=Color.anomaly, alpha=0.5)
        if len(signal) > len(style):
            while len(signal) > len(style):
                style.append("-")
        elif len(signal) < len(style):
            while len(signal) < len(style):
                style.pop()
        low = 999
        high = -999
        for i, (s, sty) in enumerate(zip(signal, style)):
            ax.plot(s, sty)
            low = min(low, min(s))
            high = max(high, max(s))
        if name.startswith("loss"):
            # invert x axis
            ax.set_xlim(len(signal[0]), 0)
        else:
            ax.set_ylim(self.adjust_limits(low, high))

class VisualizerInference(Visualizer_base):
    def __init__(self, config, n_plots=5, column_names=None, plot_loss=True):
        super().__init__(config)
        self.config = config
        self.n_plots = n_plots
        self.dataset = config['data']['dataset']
        self.entity = config['data']['entity']
        if column_names is None:
            column_names = list(np.arange(n_plots).astype(str))

        self.column_names = column_names
        self.plot_loss = plot_loss
        self.create_subplots(n_plots)

    def create_subplots(self, n_plots):
        assert len(self.column_names) == n_plots, f"plot names and number of subplots do not match ({len(self.column_names)} vs. {n_plots})"
        n_plots = n_plots + 1 if self.plot_loss else n_plots
        for i, name in enumerate(self.column_names):
            self.ax_dict[name] = self.fig.add_subplot(n_plots, 1, i+1,
                                                      sharex=self.ax_dict[self.column_names[0]] if i > 0 else None)
        if self.plot_loss:
            self.ax_dict['loss'] = self.fig.add_subplot(n_plots, 1, n_plots)
            self.column_names.append("loss")


    def update_subplot(self, subplotcount, signal: list, style: list = ['--'], y=None):
        """


        :param epoch:
        :param signal: list with true and predicted signal
        :param subplotcount:
        :param style:
        :param y:
        :return:
        """
        self.fig.suptitle(f"Inference: {self.dataset}-{self.entity}", c=Color.text)
        if isinstance(subplotcount, int):
            name = self.column_names[subplotcount]
        elif isinstance(subplotcount, str):
            name = subplotcount
        else:
            raise NotImplementedError(f"don't know axis {subplotcount}")
        ax = self.ax_dict[name]
        ax.clear()
        ax.set_title(f"{name}")
        if name=="loss":
            ax.set_title(f"mean reconstruction error")
        # self.set_ax_lims(name, ax)
        ax.grid(color=Color.text)
        if y is not None:
            consec_ano = hp.find_segments(y.numpy())
            for i in consec_ano:
                ax.axvspan(i[0], i[1], color=Color.anomaly, alpha=0.5)
        if len(signal) > len(style):
            while len(signal) > len(style):
                style.append("-")
        elif len(signal) < len(style):
            while len(signal) < len(style):
                style.pop()
        low = 999
        high = -999
        for i, (s, sty) in enumerate(zip(signal, style)):
            ax.plot(s, sty)
            low = min(low, min(s))
            high = max(high, max(s))

        ax.set_ylim(self.adjust_limits(low, high))

