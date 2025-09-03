import sys
import os
import numpy
from typing import List

from ..distortion_objects.zernike_distortion import ZernikeDistortion
from ..read_transform import read_transform

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QListWidget, QLabel, QSplitter, QPushButton, QFileDialog,
    QMessageBox, QHBoxLayout, QComboBox, QListWidgetItem
)
from PyQt5.QtCore import Qt, QUrl

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# ============================================================
# Canvas Matplotlib
# ============================================================
class MplCanvas(FigureCanvas):
    """Canvas matplotlib integrated into PyQt."""
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 4), dpi=100)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

# ============================================================
# Bar plot on an axes
# ============================================================
def plot_bar_chart(mpl_canvas: MplCanvas, distortions: List[ZernikeDistortion], labels: List[str], mode: str = "x") -> MplCanvas:
    """Plot a bar chart of the Zernike coefficients for various models"""
    # Check the inputs
    if not isinstance(mpl_canvas, MplCanvas):
        raise ValueError("Invalid MplCanvas object")

    if not isinstance(distortions, list) or len(distortions) == 0:
        raise ValueError("Invalid distortions list")

    if not all(isinstance(d, ZernikeDistortion) for d in distortions):
        raise ValueError("Invalid distortion object")

    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("Invalid labels list")

    if not all(isinstance(l, str) for l in labels):
        raise ValueError("Invalid label object")
    
    if not len(labels) == len(distortions):
        raise ValueError("Labels and distortions must have the same length")

    if not mode in ["x", "y"]:
        raise ValueError("Invalid mode, must be 'x' or 'y'")

    # Clear the axes
    mpl_canvas.axes.clear()

    # Extract the maximum order
    Nzer = max(distortion.Nzer for distortion in distortions)

    # Constructs the labels and parameters
    ticklabels = [f"C{mode}({n}, {m})" for n in range(Nzer + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0]
    x = numpy.arange(len(ticklabels))  # Position of the bars on the x-axis
    x_width = 0.8 / len(distortions)  # Width of each bar
    x_shift = -0.4 + x_width / 2  # Center the bars
    colormap = plt.get_cmap('viridis')

    for i, distortion in enumerate(distortions):
        parameters = distortion.parameters_x if mode == "x" else distortion.parameters_y
        parameters = numpy.concatenate([parameters, numpy.zeros(len(ticklabels) - len(parameters))])
        mpl_canvas.axes.bar(x + x_shift + i * x_width, parameters, width=x_width, label=labels[i], color=colormap(i / len(distortions)))

    mpl_canvas.axes.set_xticks(x + x_shift + x_width / 2)
    mpl_canvas.axes.set_xticklabels(ticklabels, rotation=75)
    mpl_canvas.axes.set_title(f"Zernike Coefficients [{mode}-axis]")
    mpl_canvas.axes.legend()
    mpl_canvas.draw()
    return mpl_canvas

# ============================================================
# Filename truncate
# ============================================================
def truncate_filename(file_path, max_len=30):
    r"""
    Truncate the filename to a maximum length, preserving the file extension.
    """
    name = os.path.basename(file_path)
    if len(name) <= max_len:
        return name
    part_len = (max_len - 3) // 2
    return f"{name[:part_len]}...{name[-part_len:]}"


# ============================================================
# DropWidget
# ============================================================
class DropWidget(QWidget):
    """
    Drag-and-drop widget for loading files into the application.

    Supports multiple file selection and updates a given QListWidget and file_data dictionary.
    """

    def __init__(self, file_list_widget, file_data, statusbar):
        super().__init__()
        self.setAcceptDrops(True)
        self.file_list_widget = file_list_widget
        self.file_data = file_data
        self.statusbar = statusbar

        # Instruction label
        self.label = QLabel("Drag your files here", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self._reset_style()

    def _highlight_style(self):
        self.setStyleSheet("background-color: #e8ffe8; border: 2px dashed #66aa66;")
        self.label.setText("Release to drop")

    def _reset_style(self):
        self.setStyleSheet("background-color: none; border: 2px dashed #cccccc;")
        self.label.setText("Drag your files here")

    def _extract_urls(self, mime):
        if mime.hasUrls():
            return list(mime.urls())
        if mime.hasText():
            text = mime.text().strip()
            return [QUrl.fromLocalFile(p.strip()) for p in text.splitlines() if p.strip()]
        return []

    # --- Drag & Drop Events ---
    def dragEnterEvent(self, event):
        urls = self._extract_urls(event.mimeData())
        if urls:
            event.acceptProposedAction()
            self._highlight_style()
            self.statusbar.showMessage(f"{len(urls)} file(s) detected")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._reset_style()
        self.statusbar.clearMessage()

    def dropEvent(self, event):
        urls = self._extract_urls(event.mimeData())
        if not urls:
            event.ignore()
            self._reset_style()
            return

        event.acceptProposedAction()
        self._reset_style()

        loaded_count = 0
        for qurl in urls:
            file_path = qurl.toLocalFile()
            if not file_path or file_path in self.file_data:
                continue  # Skip already loaded files

            try:
                distortion = read_transform(file_path, ZernikeDistortion)
                self.file_data[file_path] = distortion
                self.file_list_widget.addItem(file_path)
                loaded_count += 1
            except Exception as e:
                QMessageBox.warning(
                    self, "File Read Error",
                    f"Unable to read {file_path}:\n{e}"
                )

        # Select all newly added files
        for qurl in urls:
            file_path = qurl.toLocalFile()
            if not file_path:
                continue
            items = self.file_list_widget.findItems(file_path, Qt.MatchExactly)
            for item in items:
                item.setSelected(True)

        self.statusbar.showMessage(f"{loaded_count} file(s) loaded")





# ============================================================
# Main Window
# ============================================================
class ZernikeDistortionVisualizerUI(QMainWindow):
    """
    Main application window for visualizing Zernike distortion files.

    Supports multiple file selection for superimposed plotting and deletion.
    """

    def __init__(self):
        super().__init__()
        self.file_data = {}

        splitter = QSplitter()

        # --- File list ---
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)  # Standard multi-selection behavior
        self.file_list.itemSelectionChanged.connect(self.update_plot)

        # --- Figure selector ---
        self.figure_selector = QComboBox()
        self.figure_selector.addItems(["Figure 1 : parameters (X)", "Figure 2 : parameters (Y)"])
        self.figure_selector.currentTextChanged.connect(self.update_plot)

        # --- Status bar ---
        self.statusbar = self.statusBar()

        # --- Drop area ---
        self.drop_area = DropWidget(self.file_list, self.file_data, self.statusbar)

        # --- Left panel (buttons + drop area + list + combo) ---
        left_panel = QWidget()
        left_panel.setMinimumWidth(100)
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)

        btn_layout = QHBoxLayout()
        open_btn = QPushButton("Open...")
        open_btn.clicked.connect(self.open_files)
        remove_btn = QPushButton("Delete")
        remove_btn.clicked.connect(self.remove_selected_files)
        btn_layout.addWidget(open_btn)
        btn_layout.addWidget(remove_btn)
        left_layout.addLayout(btn_layout)

        left_layout.addWidget(self.drop_area)
        left_layout.addWidget(QLabel("Loaded files:"))
        left_layout.addWidget(self.file_list)
        left_layout.addWidget(QLabel("Select figure:"))
        left_layout.addWidget(self.figure_selector)

        splitter.addWidget(left_panel)

        # --- Matplotlib canvas ---
        self.canvas = MplCanvas(self)
        splitter.addWidget(self.canvas)

        self.setCentralWidget(splitter)
        self.setWindowTitle("Zernike Distortion Visualizer")

    # --- File management ---
    def remove_selected_files(self):
        """Delete all selected files from the list and internal data."""
        selected_items = self.file_list.selectedItems()
        for item in selected_items:
            file_path = item.text()
            if file_path in self.file_data:
                del self.file_data[file_path]
            self.file_list.takeItem(self.file_list.row(item))
        self.update_plot()

    def open_files(self):
        """
        Open files via a QFileDialog and load them as ZernikeDistortion.
        Avoids crashes when .json files are present in the folder by using
        the Qt internal file dialog instead of the native one.
        Already loaded files are not duplicated, only selected.
        """
        dialog = QFileDialog(self, "Select files")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilters(["JSON Files (*.json)", "All Files (*)"])
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # Avoid native dialog crash

        if dialog.exec_():
            paths = dialog.selectedFiles()
            for p in paths:
                if p in self.file_data:
                    # File already loaded → select it in the list
                    items = self.file_list.findItems(p, Qt.MatchExactly)
                    if items:
                        self.file_list.setCurrentItem(items[0])
                    continue

                try:
                    distortion = read_transform(p, ZernikeDistortion)
                    self.file_data[p] = distortion
                    self.file_list.addItem(p)
                except Exception as e:
                    QMessageBox.warning(self, "File Read Error", f"Unable to read {p}:\n{e}")

            # Select all newly added files
            for p in paths:
                items = self.file_list.findItems(p, Qt.MatchExactly)
                if items:
                    items[0].setSelected(True)

            self.update_plot()

    # --- Plot update ---
    def update_plot(self):
        """Update the matplotlib figure with currently selected files."""
        selected_items = self.file_list.selectedItems()
        self.canvas.axes.clear()

        if not selected_items:
            self.canvas.draw()
            return

        fig_choice = self.figure_selector.currentText()

        distortions = []
        labels = []

        for item in selected_items:
            file_path = item.text()
            distortion = self.file_data.get(file_path)
            if distortion:
                distortions.append(distortion)
                labels.append(truncate_filename(file_path))

        if not distortions:
            self.canvas.draw()
            return

        if fig_choice.startswith("Figure 1"):
            self.canvas = plot_bar_chart(self.canvas, distortions, labels, mode="x")
        elif fig_choice.startswith("Figure 2"):
            self.canvas = plot_bar_chart(self.canvas, distortions, labels, mode="y")
