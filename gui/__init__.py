#!/usr/bin/env python3
import sys
import json
import threading
import socket

from gui.tabs.helpers import HelperFunctions
from gui.widgets import Widgets
from core.hardware.filter_wheel_fratio import FilterWheel

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QFileDialog, QVBoxLayout
                             )
from PyQt6.QtCore import pyqtSignal

from gui.main_window import MainWindowInit
from gui.widgets import FiberDataWindow


if sys.platform.startswith("linux"):
    print("Linux")
    BASE_PATH = r"/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements"
elif sys.platform.startswith("win"):
    hostname = socket.gethostname()
    if hostname == "DESKTOP-HEBN59N":
        BASE_PATH = r"D:\Vincent"
    else:
        BASE_PATH = r"\\srv4\labshare\raw_data\fibers\Measurements"
else:
    raise OSError("Unsupported OS")



class MainWindow(QMainWindow, HelperFunctions, Widgets):
    progress_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.stop_event = threading.Event()

        self.base_directory = BASE_PATH
        #print(self.base_directory)
        self.inputs_locked = False
        self.experiment_running = False

        self.setWindowTitle("Fiber Measurement and Analysis")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.fiber_diameter = ""
        self.fiber_width = ""
        self.fiber_height = ""
        self.fiber_shape = ""
        self.folder_name = ""
        self.fiber_dimension = ""

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: red; font-weight: bold;")

        self.filter_wheel_ready = False
        self.filter_wheel_initiated = False

        self.main_window_init = MainWindowInit(self)

        self.fiber_data_window = FiberDataWindow(self)

    def initialize_filter_wheel(self):
        import serial.tools.list_ports
        available_ports = [port.device for port in serial.tools.list_ports.comports()]
        if 'COM5' in available_ports:
            self.filter_wheel_initiated = True
            self.filter_wheel = FilterWheel('COM5')
            self.filter_wheel_ready = True
            self.main_window_init.general_tab_init.update_general_tab_buttons()
        else:
            self.main_window_init.show_message("COM5 is not available.")

    def update_fiber_shape_inputs(self):
        fiber_shape = self.fiber_shape_combo.currentText()
        if fiber_shape == "rectangular":
            self.fiber_diameter_label.hide()
            self.fiber_diameter_input.hide()
            self.fiber_width_label.show()
            self.fiber_width_input.show()
            self.fiber_height_label.show()
            self.fiber_height_input.show()
        else:
            self.fiber_diameter_label.show()
            self.fiber_diameter_input.show()
            self.fiber_width_label.hide()
            self.fiber_width_input.hide()
            self.fiber_height_label.hide()
            self.fiber_height_input.hide()

    def choose_calibration_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Calibration Folder",
                                                       self.base_directory + "/Calibration")
        if folder_path:
            self.calibration_folder_input.setText(folder_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())