import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class PlotConfig:
    def __init__(self):
        self.style = "stylesheet.mplstyle"
        self.fig_width = 246.0 * (1.0 / 72.27)
        self.fig_height = self.fig_width / 1.618
        self.colors = [
            "#395470",  # soft viridis-style blue
            "#5A7A87",  # pastel twilight blue-teal
            "#A4C9A7",  # pastel sage
            "#D3C76A",  # pastel olive
            "#E9DF83",  # slightly darker pastel yellow
        ]
        self.colormap = LinearSegmentedColormap.from_list("custom_colormap", self.colors)

    def apply_style(self):
        plt.style.use(self.style)