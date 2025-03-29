import sys

import PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QStatusBar, QWidget, QMenuBar, QAction, QSizePolicy, QComboBox,
    QSlider, QSpinBox, QGridLayout, QCheckBox, QWidgetItem
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QSize

import numpy as np
from PIL import Image

from adv_img_operations import color_thief
from image_statistics import statistics_window
from transformations.convolution import smoothing, sharpening, custom_filter
from transformations.filters import grayscale, brightness, contrast, negative
from transformations.edge_detection import sobel, roberts, canny


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set relevant fields:
        # TODO: Przenieść do osobnej klasy typu ImageProcessor
        self.image_path = None
        self.rgb_image_cache = None
        self.working_image = None
        self.saved_image = None
        self.cached_saved_image = None
        self.is_gray = False
        self.is_original_gray = False

        self.default_font = QFont("Helvetica", 14)
        self.big_label_font = QFont("Helvetica", 18)
        self.title_font = QFont("Helvetica Neue", 24)
        self.button_font = QFont("Helvetica", 14)

        self.initUI()

    def initUI(self):
        # Main window settings
        self.setWindowTitle('Image Processor')
        self.setGeometry(100, 100, 1200, 800)  # Larger window

        # Main horizontal layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_panel.setObjectName("leftPanel")

        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setObjectName("leftLayout")
        left_layout.setContentsMargins(20, 50, 20, 50)

        # Transformation selection
        self.transformation_combo = QComboBox()
        self.transformation_combo.addItems([
            "Select Transformation",
            "Grayscale",
            "Brightness Adjustment",
            "Contrast Adjustment",
            "Gamma Adjustment",
            "Negative",
            "Binarization",
            "Kernel Filters",
            "Edge Detection",
            "Canny"
        ])
        self.transformation_combo.currentIndexChanged.connect(self.show_parameters)
        left_layout.addWidget(QLabel("Select Transformation:"))
        left_layout.addWidget(self.transformation_combo)

        # Parameters container
        self.parameters_container = QWidget()
        self.parameters_layout = QVBoxLayout(self.parameters_container)
        left_layout.addWidget(self.parameters_container)

        # Apply button
        self.apply_btn = QPushButton("Apply Transformation")
        self.apply_btn.clicked.connect(self.apply_transformation)
        left_layout.addWidget(self.apply_btn)

        left_layout.addStretch()
        self.stats_btn = QPushButton("Show Statistics")
        self.stats_btn.clicked.connect(self.show_statistics)
        left_layout.addWidget(self.stats_btn)

        # Right panel for image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        right_layout.addWidget(self.image_label)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Configure menu bar
        self.createToolbar()
        self.create_menu()
        self.center()
        self.setStyleSheet(open('static/css/styles.css').read())

    def createToolbar(self):
        toolbar = self.addToolBar('Main Toolbar')
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        toolbar.setIconSize(QSize(20,20))  # Set the icon size to 32x32 pixels

        # Add a spacer between icons
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        upload_action = QAction(QIcon('static/img/upload.png'), 'Upload', self)
        upload_action.triggered.connect(self.open_image)
        upload_action.setObjectName('Upload')
        toolbar.addAction(upload_action)

        undo_action = QAction(QIcon('static/img/undo.png'), 'Undo', self)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        revert_action = QAction(QIcon('static/img/revert2.png'), 'Revert', self)
        revert_action.triggered.connect(self.revert_to_original)
        toolbar.addAction(revert_action)

        save_action = QAction(QIcon('static/img/save.png'), 'Save', self)
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)

    def create_menu(self):
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu('File')
        file_menu.addAction('Open').triggered.connect(self.open_image)
        file_menu.addAction('Exit').triggered.connect(self.close)
        self.setMenuBar(menu_bar)

    def center(self):
        screen = QApplication.primaryScreen().availableGeometry()
        window = self.frameGeometry()
        window.moveCenter(screen.center())
        self.move(window.topLeft())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.showNormal()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._display_image(self.working_image)

    def show_parameters(self):
        """Show parameters for selected transformation"""
        # Clear previous parameters
        while self.parameters_layout.count():
            child = self.parameters_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif isinstance(child, QGridLayout):
                for j in range(child.count()):
                    sub_widget = child.itemAt(j).widget()
                    if sub_widget:
                        sub_widget.deleteLater()

        transformation = self.transformation_combo.currentText()

        if transformation == "Grayscale":
            self.add_grayscale_parameters()
        elif transformation == "Brightness Adjustment":
            self.add_brightness_parameters()
        elif transformation == "Contrast Adjustment":
            self.add_contrast_parameters()
        elif transformation == "Kernel Filters":
            self.add_kernel_parameters()
        elif transformation == "Gamma Adjustment":
            self.add_gamma_parameters()
        elif transformation == "Negative":
            pass
        elif transformation == "Binarization":
            self.add_binarization_parameters()
        elif transformation == "Edge Detection":
            self.add_edge_detection_parameters()
        elif transformation == "Canny":
            self.add_canny_parameters()

        self.cached_saved_image = self.saved_image
        self.saved_image = self.working_image

    def add_grayscale_parameters(self):
        """Parameters for grayscale conversion"""
        self.grayscale_method = QComboBox()
        self.grayscale_method.addItems([
            "Custom Weights",
            "Luminosity",
            "Average"
        ])
        self.grayscale_method.currentIndexChanged.connect(self.show_grayscale_weights)
        self.parameters_layout.addWidget(QLabel("Method:"))
        self.parameters_layout.addWidget(self.grayscale_method)
        if self.grayscale_method.currentText() == "Custom Weights":
            self.show_grayscale_weights()

    def show_grayscale_weights(self):
        """Show custom weights for grayscale conversion if custom method is selected"""
        method = self.grayscale_method.currentText()
        if method == "Custom Weights":
            self.parameters_layout.addWidget(QLabel("Choose custom weights:"))
            self.grayscale_weight_red = QSpinBox()
            self.grayscale_weight_red.setRange(0, 100)
            self.grayscale_weight_red.setValue(3)

            self.grayscale_weight_green = QSpinBox()
            self.grayscale_weight_green.setRange(0, 100)
            self.grayscale_weight_green.setValue(6)

            self.grayscale_weight_blue = QSpinBox()
            self.grayscale_weight_blue.setRange(0, 100)
            self.grayscale_weight_blue.setValue(1)

            self.parameters_layout.addWidget(QLabel("Red Weight:"))
            self.parameters_layout.addWidget(self.grayscale_weight_red)
            self.parameters_layout.addWidget(QLabel("Green Weight:"))
            self.parameters_layout.addWidget(self.grayscale_weight_green)
            self.parameters_layout.addWidget(QLabel("Blue Weight:"))
            self.parameters_layout.addWidget(self.grayscale_weight_blue)
        else:
            for i in reversed(range(self.parameters_layout.count())):
                widget = self.parameters_layout.itemAt(i).widget()
                if isinstance(widget, QLabel) and widget.text() in ["Red Weight:", "Green Weight:", "Blue Weight:", "Choose custom weights:"]:
                    widget.deleteLater()
                elif isinstance(widget, QSpinBox):
                    widget.deleteLater()

    def add_brightness_parameters(self):
        """Parameters for brightness adjustment"""
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_value = QLabel("0")
        self.brightness_value.setAlignment(Qt.AlignCenter)
        self.brightness_slider.valueChanged.connect(
            lambda: self.brightness_value.setText(str(self.brightness_slider.value())))

        self.parameters_layout.addWidget(QLabel("Brightness (-100 to +100):"))
        self.parameters_layout.addWidget(self.brightness_slider)
        self.parameters_layout.addWidget(self.brightness_value)

    def add_contrast_parameters(self):
        """Parameters for contrast adjustment"""
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 50)
        self.contrast_slider.setValue(20)
        self.contrast_value = QLabel("2.0")
        self.contrast_value.setAlignment(Qt.AlignCenter)
        self.contrast_slider.valueChanged.connect(
            lambda: self.contrast_value.setText(f"{self.contrast_slider.value() / 10:.1f}"))

        self.parameters_layout.addWidget(QLabel("Contrast (0.0 to 5.0):"))
        self.parameters_layout.addWidget(self.contrast_slider)
        self.parameters_layout.addWidget(self.contrast_value)

    def add_gamma_parameters(self):
        """Parameters for gamma correction"""
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(0, 50)
        self.gamma_slider.setValue(20)
        self.gamma_value = QLabel("2.0")
        self.gamma_value.setAlignment(Qt.AlignCenter)
        self.gamma_slider.valueChanged.connect(
            lambda: self.gamma_value.setText(f"{self.gamma_slider.value() / 10:.1f}"))

        self.parameters_layout.addWidget(QLabel("Gamma (0.0 to 5.0):"))
        self.parameters_layout.addWidget(self.gamma_slider)
        self.parameters_layout.addWidget(self.gamma_value)

    def add_kernel_parameters(self):
        """Parameters for kernel filters"""
        self.kernel_type = QComboBox()
        self.kernel_type.addItems([
            "Averaging",
            "Gaussian Blur",
            "Sharpen",
            "Custom Filter"
        ])
        self.kernel_type.currentIndexChanged.connect(self.show_kernel_specific_parameters)
        self.parameters_layout.addWidget(QLabel("Kernel Type:"))
        self.parameters_layout.addWidget(self.kernel_type)

        self.kernel_size = QComboBox()
        self.kernel_size.addItems([str(i) for i in range(3, 12, 2)])
        self.kernel_size.setObjectName("kernelSizeCombo")
        self.parameters_layout.addWidget(QLabel("Kernel Size:"))
        self.parameters_layout.addWidget(self.kernel_size)

    def show_kernel_specific_parameters(self):
        """Show additional parameters based on selected kernel type"""
        for i in reversed(range(self.parameters_layout.count())):
            widget = self.parameters_layout.itemAt(i)
            if isinstance(widget, QWidgetItem):
                widget = widget.widget()
            if isinstance(widget, QSlider) or (isinstance(widget, QLabel) and widget.text() in ["Sigma:", "Alpha:", "Kernel Size:"])\
                    or (isinstance(widget, QLabel) and widget.objectName() in ["sigmaValue", "alphaValue", "customKernelLabel"])\
                    or (isinstance(widget, QComboBox) and widget.objectName() == "kernelSizeCombo"):
                widget.deleteLater()
            elif isinstance(widget, QGridLayout):
                for j in range(widget.count()):
                    sub_widget = widget.itemAt(j).widget()
                    sub_widget.deleteLater()

        kernel_type = self.kernel_type.currentText()
        if kernel_type == "Gaussian Blur":
            self.kernel_size = QComboBox()
            self.kernel_size.addItems([str(i) for i in range(3, 12, 2)])
            self.kernel_size.setObjectName("kernelSizeCombo")
            self.parameters_layout.addWidget(QLabel("Kernel Size:"))
            self.parameters_layout.addWidget(self.kernel_size)

            self.gaussian_sigma_slider = QSlider(Qt.Horizontal)
            self.gaussian_sigma_slider.setRange(1, 50)
            self.gaussian_sigma_slider.setValue(10)
            self.sigma_value = QLabel("1.0")
            self.sigma_value.setObjectName("sigmaValue")
            self.sigma_value.setAlignment(Qt.AlignCenter)
            self.gaussian_sigma_slider.valueChanged.connect(
                lambda: self.sigma_value.setText(f"{self.gaussian_sigma_slider.value() / 10:.1f}"))
            self.parameters_layout.addWidget(QLabel("Sigma:"))
            self.parameters_layout.addWidget(self.gaussian_sigma_slider)
            self.parameters_layout.addWidget(self.sigma_value)

        elif kernel_type == "Sharpen":
            self.kernel_size = QComboBox()
            self.kernel_size.addItems([str(i) for i in range(3, 12, 2)])
            self.kernel_size.setObjectName("kernelSizeCombo")
            self.parameters_layout.addWidget(QLabel("Kernel Size:"))
            self.parameters_layout.addWidget(self.kernel_size)

            self.sharpen_alpha_slider = QSlider(Qt.Horizontal)
            self.sharpen_alpha_slider.setRange(1, 50)
            self.sharpen_alpha_slider.setValue(10)
            self.alpha_value = QLabel("1.0")
            self.alpha_value.setObjectName("alphaValue")
            self.alpha_value.setAlignment(Qt.AlignCenter)
            self.sharpen_alpha_slider.valueChanged.connect(
                lambda: self.alpha_value.setText(f"{self.sharpen_alpha_slider.value() / 10:.1f}"))
            self.parameters_layout.addWidget(QLabel("Alpha:"))
            self.parameters_layout.addWidget(self.sharpen_alpha_slider)
            self.parameters_layout.addWidget(self.alpha_value)

        elif kernel_type == "Custom Filter":
            self.add_custom_filter_parameters()

        else:
            self.kernel_size = QComboBox()
            self.kernel_size.addItems([str(i) for i in range(3, 12, 2)])
            self.kernel_size.setObjectName("kernelSizeCombo")
            self.parameters_layout.addWidget(QLabel("Kernel Size:"))
            self.parameters_layout.addWidget(self.kernel_size)

    def add_custom_filter_parameters(self):
        """Parameters for custom kernel filter"""
        self.custom_kernel_values = []
        grid_layout = QGridLayout()
        grid_layout.setObjectName("customKernelGridLayout")
        custom_kernel_label = QLabel("Enter 3x3 Kernel Values:")
        custom_kernel_label.setObjectName("customKernelLabel")
        self.parameters_layout.addWidget(custom_kernel_label)
        for i in range(3):
            for j in range(3):
                spin_box = QSpinBox()
                spin_box.setRange(-255, 255)
                spin_box.setValue(0)
                self.custom_kernel_values.append(spin_box)
                grid_layout.addWidget(spin_box, i, j)
        self.parameters_layout.addLayout(grid_layout)

    def add_binarization_parameters(self):
        self.binarization_slider = QSlider(Qt.Horizontal)
        self.binarization_slider.setRange(0, 255)
        self.binarization_slider.setValue(128)
        self.binarization_value = QLabel("128")
        self.binarization_value.setAlignment(Qt.AlignCenter)
        self.binarization_slider.valueChanged.connect(
            lambda: self.binarization_value.setText(str(self.binarization_slider.value()))
        )

        self.parameters_layout.addWidget(QLabel("Binarization Threshold (0-255):"))
        self.parameters_layout.addWidget(self.binarization_slider)
        self.parameters_layout.addWidget(self.binarization_value)

    def add_edge_detection_parameters(self):
        """Parameters for edge detection"""
        self.edge_detection_method = QComboBox()
        self.edge_detection_method.addItems([
            "Sobel",
            "Robert's Cross"
        ])
        self.parameters_layout.addWidget(QLabel("Method:"))
        self.parameters_layout.addWidget(self.edge_detection_method)

        self.edge_detection_threshold = QSlider(Qt.Horizontal)
        self.edge_detection_threshold.setRange(0, 255)
        self.edge_detection_threshold.setValue(128)
        self.edge_detection_threshold_value = QLabel("128")
        self.edge_detection_threshold_value.setAlignment(Qt.AlignCenter)
        self.edge_detection_threshold.valueChanged.connect(
            lambda: self.edge_detection_threshold_value.setText(str(self.edge_detection_threshold.value()))
        )

        self.parameters_layout.addWidget(QLabel("Threshold (0-255):"))
        self.parameters_layout.addWidget(self.edge_detection_threshold)
        self.parameters_layout.addWidget(self.edge_detection_threshold_value)

    def add_canny_parameters(self):
        # Add 2 thresholds for Canny edge detection
        self.canny_low_threshold = QSlider(Qt.Horizontal)
        self.canny_low_threshold.setRange(0, 255)
        self.canny_low_threshold.setValue(50)
        self.canny_low_threshold_value = QLabel("50")
        self.canny_low_threshold_value.setAlignment(Qt.AlignCenter)
        self.canny_low_threshold.valueChanged.connect(
            lambda: self.canny_low_threshold_value.setText(str(self.canny_low_threshold.value())))
        self.parameters_layout.addWidget(QLabel("Canny Low Threshold (0-255):"))
        self.parameters_layout.addWidget(self.canny_low_threshold)
        self.parameters_layout.addWidget(self.canny_low_threshold_value)

        self.canny_high_threshold = QSlider(Qt.Horizontal)
        self.canny_high_threshold.setRange(0, 255)
        self.canny_high_threshold.setValue(150)
        self.canny_high_threshold_value = QLabel("150")
        self.canny_high_threshold_value.setAlignment(Qt.AlignCenter)
        self.canny_high_threshold.valueChanged.connect(
            lambda: self.canny_high_threshold_value.setText(str(self.canny_high_threshold.value())))
        self.parameters_layout.addWidget(QLabel("Canny High Threshold (0-255):"))
        self.parameters_layout.addWidget(self.canny_high_threshold)
        self.parameters_layout.addWidget(self.canny_high_threshold_value)


    def apply_transformation(self):
        """Handle transformation application"""
        transformation = self.transformation_combo.currentText()

        if transformation == "Grayscale":
            method = self.grayscale_method.currentText()
            if method == "Custom Weights":
                red_weight = self.grayscale_weight_red.value()
                green_weight = self.grayscale_weight_green.value()
                blue_weight = self.grayscale_weight_blue.value()
                print(f"Applying grayscale with custom weights: {red_weight}, {green_weight}, {blue_weight}")
                self.apply_grayscale(method, (red_weight, green_weight, blue_weight))
            else:
                self.apply_grayscale(method)

        elif transformation == "Brightness Adjustment":
            value = self.brightness_slider.value()
            self.adjust_brightness(value)

        elif transformation == "Contrast Adjustment":
            value = self.contrast_slider.value() / 10
            self.adjust_contrast(value)

        elif transformation == "Gamma Adjustment":
            value = self.gamma_slider.value() / 10
            self.apply_gamma(value)

        elif transformation == "Negative":
            self.apply_negative()

        elif transformation == "Binarization":
            threshold = self.binarization_slider.value()
            self.apply_binarization(threshold)

        elif transformation == "Edge Detection":
            method = self.edge_detection_method.currentText()
            threshold = self.edge_detection_threshold.value()
            if method == "Sobel":
                print(f"Applying Sobel edge detection with threshold: {threshold}")
                self.apply_sobel_edge_detection(threshold)
            elif method == "Robert's Cross":
                print(f"Applying Robert's Cross edge detection with threshold: {threshold}")
                self.apply_roberts_cross_edge_detection(threshold)
            else:
                print("Invalid edge detection method selected")

        elif transformation == "Kernel Filters":
            method = self.kernel_type.currentText()
            if method == "Gaussian Blur":
                size = int(self.kernel_size.currentText())
                sigma = self.gaussian_sigma_slider.value() / 10
                print(f"Applying Gaussian Blur with sigma: {sigma}")
                self.apply_gaussian_blur(size, sigma)
            elif method == "Sharpen":
                size = int(self.kernel_size.currentText())
                alpha = self.sharpen_alpha_slider.value() / 10
                print(f"Applying Sharpen with alpha: {alpha}")
                self.apply_sharpen(size, alpha)
            elif method == "Custom Filter":
                kernel_values = [spin_box.value() for spin_box in self.custom_kernel_values]
                print(f"Applying Custom Filter with kernel values: {kernel_values}")
                self.apply_custom_filter(kernel_values)
            else:
                size = int(self.kernel_size.currentText())
                print(f"Applying Averaging filter with size: {size}")
                self.apply_averaging(size)

        elif transformation == "Canny":
            low_threshold = self.canny_low_threshold.value()
            high_threshold = self.canny_high_threshold.value()
            print(f"Applying Canny edge detection with low: {low_threshold}, high: {high_threshold}")
            self.apply_canny(low_threshold, high_threshold)

    def apply_grayscale(self, method, weights=(0.3, 0.6, 0.1)):
        print(f"Applying grayscale with method: {method}")
        self.convert_to_grayscale(method, weights)

    def adjust_brightness(self, value):
        print(f"Adjusting brightness by {value}")
        if self.saved_image is not None:
            try:
                adj_image = brightness.adjust_brightness(self.saved_image, value)
                self.working_image = adj_image
                self._display_image(adj_image)
                self.status_bar.showMessage("✅ Brightness adjusted")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Brightness correction error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to correct brightness")

    def adjust_contrast(self, value):
        print(f"Adjusting contrast to {value}%")
        if self.saved_image is not None:
            try:
                adj_image = contrast.adjust_contrast(self.saved_image, value)
                self.working_image = adj_image
                self._display_image(adj_image)
                self.status_bar.showMessage("✅ Contrast adjusted")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Contrast correction error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to correct contrast")

    def apply_gamma(self, value):
        print(f"Applying gamma correction with value: {value}")
        if self.saved_image is not None:
            try:
                adj_image = brightness.adjust_gamma(self.saved_image, value)
                self.working_image = adj_image
                self._display_image(adj_image)
                self.status_bar.showMessage("✅ Gamma correction applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Gamma correction error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply gamma correction")

    def apply_negative(self):
        print("Applying negative transformation")
        if self.saved_image is not None:
            try:
                neg_image = negative.apply_negative(self.saved_image)
                self.working_image = neg_image
                self._display_image(neg_image)
                self.status_bar.showMessage("✅ Negative transformation applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Negative transformation error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply negative transformation")

    def apply_binarization(self, threshold):
        print(f"Applying binarization with threshold {threshold}")
        if self.saved_image is not None:
            try:
                if not self.is_gray:
                    gray_image = grayscale.rgb_to_grayscale_luminosity(self.saved_image)
                    bin_image = grayscale.binarize(gray_image, threshold)
                else:
                    bin_image = grayscale.binarize(self.saved_image, threshold)
                self.working_image = bin_image
                self.is_gray = True
                self._display_image(bin_image)
                self.status_bar.showMessage("✅ Binarization applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Binarization error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply binarization")

    def apply_averaging(self, size):
        print(f"Applying averaging filter with kernel size: {size}")
        if self.saved_image is not None:
            try:
                filtered_image = smoothing.apply_average_filter(self.saved_image, size)
                self.working_image = filtered_image
                self._display_image(filtered_image)
                self.status_bar.showMessage("✅ Averaging filter applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Averaging filter error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply averaging filter")

    def apply_gaussian_blur(self, size, sigma):
        print(f"Applying Gaussian Blur with kernel size: {size} and sigma: {sigma}")
        if self.saved_image is not None:
            try:
                filtered_image = smoothing.apply_gaussian_blur(self.saved_image, size, sigma)
                self.working_image = filtered_image
                self._display_image(filtered_image)
                self.status_bar.showMessage("✅ Gaussian Blur applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Gaussian Blur error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply Gaussian Blur")

    def apply_sharpen(self, size, alpha):
        print(f"Applying Sharpen filter with kernel size: {size} and alpha: {alpha}")
        if self.saved_image is not None:
            try:
                filtered_image = sharpening.apply_sharpen_image(self.saved_image, size, alpha)
                self.working_image = filtered_image
                self._display_image(filtered_image)
                self.status_bar.showMessage("✅ Sharpen filter applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Sharpen filter error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply Sharpen filter")

    def apply_custom_filter(self, kernel_values):
        print(f"Applying Custom Filter with kernel values: {kernel_values}")
        if self.saved_image is not None:
            try:
                filtered_image = custom_filter.apply_custom_filter(self.saved_image, kernel_values)
                self.working_image = filtered_image
                self._display_image(filtered_image)
                self.status_bar.showMessage("✅ Custom Filter applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Custom Filter error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply Custom Filter")

    def apply_sobel_edge_detection(self, threshold):
        print(f"Applying Sobel edge detection with threshold: {threshold}")
        if self.saved_image is not None:
            try:
                if not self.is_gray:
                    gray_image = grayscale.rgb_to_grayscale_luminosity(self.saved_image)
                    edge_image = sobel.sobel_edge_detection(gray_image, threshold)
                else:
                    edge_image = sobel.sobel_edge_detection(self.saved_image, threshold)
                self.working_image = edge_image
                self.is_gray = True
                self._display_image(edge_image)
                self.status_bar.showMessage("✅ Sobel edge detection applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Sobel edge detection error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply Sobel edge detection")

    def apply_roberts_cross_edge_detection(self, threshold):
        print(f"Applying Robert's Cross edge detection with threshold: {threshold}")
        if self.saved_image is not None:
            try:
                if not self.is_gray:
                    gray_image = grayscale.rgb_to_grayscale_luminosity(self.saved_image)
                    edge_image = roberts.roberts_edge_detection(gray_image, threshold)
                else:
                    edge_image = roberts.roberts_edge_detection(self.saved_image, threshold)
                self.working_image = edge_image
                self.is_gray = True
                self._display_image(edge_image)
                self.status_bar.showMessage("✅ Robert's Cross edge detection applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Robert's Cross edge detection error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply Robert's Cross edge detection")

    def apply_canny(self, low_threshold, high_threshold):
        print(f"Applying Canny edge detection with low: {low_threshold}, high: {high_threshold}")
        if self.saved_image is not None:
            try:
                if not self.is_gray:
                    gray_image = grayscale.rgb_to_grayscale_luminosity(self.saved_image)
                    edge_image = canny.canny_edge_detection(gray_image, low_threshold, high_threshold)
                else:
                    edge_image = canny.canny_edge_detection(self.saved_image, low_threshold, high_threshold)
                self.working_image = edge_image
                self.is_gray = True
                self._display_image(edge_image)
                self.status_bar.showMessage("✅ Canny edge detection applied")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Canny edge detection error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to apply Canny edge detection")

    def array_to_pixmap(self, array):
        """Convert numpy array to QPixmap
        Numpy array image is assumed to be in RGB format and shape (H, W, 3-channels)"""
        height, width, _ = array.shape
        bytes_per_line = 3 * width
        qimage = QImage(
            array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        return QPixmap.fromImage(qimage)

    def open_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options
        )

        if file_path:
            try:
                # Load image as numpy array using PIL
                pil_image = Image.open(file_path).convert('RGB')
                # Always (H, W, C) format
                self.rgb_image_cache = np.array(pil_image)
                self.working_image = self.rgb_image_cache.copy()
                self.saved_image = self.rgb_image_cache.copy()
                self.cached_saved_image = self.rgb_image_cache.copy()

                is_gray = grayscale.is_grayscale(self.rgb_image_cache)
                self.is_original_gray = is_gray
                self.is_gray = is_gray

                self._display_image(self.working_image)
                self.status_bar.showMessage(f"✅ Loaded: {file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Error: {str(e)}")

    def _display_image(self, array):
        """Helper to display any numpy array"""
        if array is not None:
            pixmap = self.array_to_pixmap(array)
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            dominant_color = color_thief.get_dominant_color(array)
            self.image_label.setStyleSheet(f"background-color: {dominant_color};")
            self.image_label.setPixmap(scaled_pixmap)

    def convert_to_grayscale(self, method="Average", weights=(0.3, 0.6, 0.1)):
        if self.saved_image is not None:
            try:
                # Apply custom grayscale conversion
                if method == "Custom Weights":
                    gray_array = grayscale.rgb_to_grayscale_custom(self.saved_image, weights)
                elif method == "Luminosity":
                    gray_array = grayscale.rgb_to_grayscale_luminosity(self.saved_image)
                elif method == "Average":
                    gray_array = grayscale.rgb_to_grayscale_average(self.saved_image)
                else:
                    raise ValueError("Invalid method selected")
                self.working_image = gray_array
                self.is_gray = True

                # Update display
                self._display_image(gray_array)
                self.status_bar.showMessage("✅ Converted to grayscale")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Conversion error: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image loaded to convert")

    def revert_to_original(self):
        """New function to revert to original image"""
        if self.rgb_image_cache is not None:
            self.saved_image = self.rgb_image_cache.copy()
            self.cached_saved_image = self.rgb_image_cache.copy()
            self.working_image = self.rgb_image_cache.copy()
            self.is_gray = self.is_original_gray
            self._display_image(self.working_image)
            self.status_bar.showMessage("✅ Reverted to original image")
        else:
            self.status_bar.showMessage("⚠️ No original image to revert to")

    def undo(self):
        """New function to undo last transformation"""
        if self.cached_saved_image is not None:
            self.saved_image = self.cached_saved_image.copy()
            self.working_image = self.cached_saved_image.copy()
            self.is_gray = grayscale.is_grayscale(self.cached_saved_image)
            self._display_image(self.working_image)
            self.status_bar.showMessage("✅ Undone last transformation")
        else:
            self.status_bar.showMessage("⚠️ No transformation to undo")

    def save_image(self):
        if self.working_image is not None:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image File", "",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;BMP Files (*.bmp);;All Files (*)",
                options=options
            )
            if file_path:
                try:
                    pil_image = Image.fromarray(self.working_image)
                    pil_image.save(file_path)
                    self.status_bar.showMessage(f"✅ Image saved: {file_path}")
                except Exception as e:
                    self.status_bar.showMessage(f"❌ Error saving image: {str(e)}")
        else:
            self.status_bar.showMessage("⚠️ No image to save")

    def show_statistics(self):
        if self.working_image is not None:
            self.stats_window = statistics_window.StatisticsWindow(self, self.working_image)
            self.stats_window.exec_()
        else:
            self.status_bar.showMessage("⚠️ No image loaded to show image_statistics")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('static/img/logo.png'))
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())