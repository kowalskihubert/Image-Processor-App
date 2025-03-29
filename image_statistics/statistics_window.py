from PyQt5.QtWidgets import QTabWidget, QDialog, QScrollArea, QWidget, QVBoxLayout, QLabel

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from transformations.filters.grayscale import is_grayscale


class StatisticsWindow(QDialog):
    def __init__(self, parent=None, image_data=None):
        super().__init__(parent)
        self.setWindowTitle("Image Statistics")
        self.setMinimumSize(800, 600)
        self.image_data = image_data
        self.initUI()

    def initUI(self):
        # Create tab widget
        tab_widget = QTabWidget()

        # Basic Info Tab
        info_tab = QWidget()
        self.info_layout = QVBoxLayout(info_tab)
        tab_widget.addTab(info_tab, "Basic Info")

        # Histogram Tab
        histogram_tab = QWidget()
        self.histogram_layout = QVBoxLayout(histogram_tab)
        tab_widget.addTab(histogram_tab, "Histogram")

        # Projections Tab
        projections_tab = QScrollArea()
        projections_content = QWidget()
        self.projections_layout = QVBoxLayout(projections_content)
        projections_tab.setWidget(projections_content)
        projections_tab.setWidgetResizable(True)
        tab_widget.addTab(projections_tab, "Projections")

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(tab_widget)

        # Populate data if image exists
        if self.image_data is not None:
            self.populate_stats()

    def populate_stats(self):
        # Basic Info
        self.add_basic_info()

        # Histogram
        self.create_histogram()

        # Projections
        self.create_projections()

    def add_basic_info(self):
        # Clear previous content
        while self.info_layout.count():
            child = self.info_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add basic stats
        height, width, _ = self.image_data.shape
        self.info_layout.addWidget(QLabel(f"Resolution: {width}x{height}"))
        if is_grayscale(self.image_data):
            self.info_layout.addWidget(QLabel("Color Channels: 3 (Grayscale)"))
        else:
            self.info_layout.addWidget(QLabel(f"Color Channels: 3 (RGB)"))
        gray_image = np.mean(self.image_data, axis=2)
        self.info_layout.addWidget(QLabel(f"Mean Intensity: {gray_image.mean():.2f}"))
        self.info_layout.addStretch()

    def create_histogram(self):
        # Clear previous content
        while self.histogram_layout.count():
            child = self.histogram_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create matplotlib figure
        fig = plt.figure()
        canvas = FigureCanvas(fig)

        if is_grayscale(self.image_data):
            # Grayscale image
            plt.hist(self.image_data[..., 0].ravel(), bins=256, color='gray', alpha=0.8)
            plt.title('Grayscale Histogram')
        else:
            # Plot histograms for each channel
            for i, color in enumerate(['red', 'green', 'blue']):
                plt.subplot(3, 1, i + 1)
                plt.hist(self.image_data[:, :, i].ravel(), bins=256, color=color)
                plt.title(f'{color.capitalize()} Channel Histogram')

        plt.tight_layout()
        self.histogram_layout.addWidget(canvas)

    def create_projections(self):
        # Clear previous content
        while self.projections_layout.count():
            child = self.projections_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Convert to grayscale for projections
        gray_image = np.mean(self.image_data, axis=2)

        # Create matplotlib figures
        fig_h = plt.figure()
        canvas_h = FigureCanvas(fig_h)
        plt.plot(gray_image.sum(axis=1))
        plt.title("Horizontal Projection")

        fig_v = plt.figure()
        canvas_v = FigureCanvas(fig_v)
        plt.plot(gray_image.sum(axis=0))
        plt.title("Vertical Projection")

        self.projections_layout.addWidget(canvas_h)
        self.projections_layout.addWidget(canvas_v)