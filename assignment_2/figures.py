import matplotlib.pyplot as plt
import numpy as np
import copy

def append_strings(attribute, length, string=""):
    if type(attribute) is str:
        return [attribute]
    elif not attribute:
        return [string for _ in range(length)]
    else:
        return attribute


class Figure:
    def __init__(
        self,
        x_sets,
        y_sets,
        labels=[],
        colors=[],
        markers=[],
        linestyles=[],
        title="",
        xlabel=r"$x$",
        ylabel=r"$y$",
    ):
        plt.style.use("seaborn-v0_8-deep")
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        self.default_colors = prop_cycle.by_key()["color"]

        if not isinstance(x_sets[0], (np.ndarray, list)):
            x_sets = [x_sets]
        self.x_sets = x_sets

        if not isinstance(y_sets[0], (np.ndarray, list)):
            y_sets = [y_sets]
        self.y_sets = y_sets

        length = len(x_sets)

        self.labels = append_strings(labels, length)

        if not colors:
            self.current_color = 0
            self.colors = []
            for i in range(length):
                self.colors += [self.current_color % len(self.default_colors)]
                self.current_color += 1
        else:
            if type(colors) is not list:
                colors = [colors]
            self.colors = colors
            self.current_color = colors[-1]

        self.markers = append_strings(markers, length)
        self.linestyles = append_strings(linestyles, length, "-")

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def addFunctions(
        self, x_sets, y_sets, labels=[], colors=[], markers=[], linestyles=[]
    ):
        if not isinstance(x_sets[0], (np.ndarray, list)):
            x_sets = [x_sets]
        if not isinstance(y_sets[0], (np.ndarray, list)):
            y_sets = [y_sets]

        self.x_sets += x_sets
        self.y_sets += y_sets

        length = len(x_sets)

        if not colors:
            for i in range(length):
                self.current_color += 1
                self.colors += [self.current_color % len(self.default_colors)]
        else:
            if type(colors) is not list:
                colors = [colors]
            self.colors += colors
            self.current_color = colors[-1]

        self.markers += append_strings(markers, length)
        self.labels += append_strings(labels, length)
        self.linestyles += append_strings(linestyles, length, "-")

    def get_figure(self, path=""):
        # Enable Latex
        plt.rcParams["text.usetex"] = True
        plt.style.use("seaborn-v0_8-deep")
        plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)

        total_functions = len(self.x_sets)
        fig, ax = plt.subplots()
        for i in range(total_functions):
            ax.plot(
                self.x_sets[i],
                self.y_sets[i],
                color=self.default_colors[self.colors[i]],
                marker=self.markers[i],
                label=self.labels[i],
                linestyle=self.linestyles[i],
            )

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.legend()
        ax.grid(True)

        if path:
            fig.savefig(path, dpi=300)

        return fig

    def copy(self):
        return copy.deepcopy(self)

    def merge(self, figs, new_title=""):
        # Accept a single Figure or a list of Figures
        if not isinstance(figs, list):
            figs = [figs]

        for i in range(len(figs)):
            self.x_sets += figs[i].x_sets
            self.y_sets += figs[i].y_sets
            self.labels += figs[i].labels
            self.colors += figs[i].colors
            self.markers += figs[i].markers
            self.linestyles += figs[i].linestyles
            
        if new_title:
            self.title = new_title

        return self
