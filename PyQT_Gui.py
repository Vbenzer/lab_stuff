#!/usr/bin/env python3
import sys
import os
import json
import threading
import time
import subprocess
from qhycfw3_filter_wheel_control import FilterWheel
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QUrl, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator


if sys.platform.startswith("linux"):
    print("Linux")
    BASE_PATH = "/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements"
elif sys.platform.startswith("win"):
    print("Windows")
    BASE_PATH = r"\\srv4\labshare\raw_data\fibers\Measurements"
    #BASE_PATH = r"D:\Vincent"
else:
    raise OSError("Unsupported OS")

def save_recent_folders(recent_folders:str, file_path:str):
    """
    Save the recent folders to a JSON file.
    Args:
        recent_folders: Folder names to save.
        file_path: Path of the JSON file to save to.
    """
    with open(file_path, 'w') as file:
        json.dump(recent_folders, file)

def load_recent_folders(file_path:str):
    """
    Load the recent folders from a JSON file.
    Args:
        file_path: Path of the JSON file to load from.

    Returns: List of recent folders.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def update_recent_folders(folder:str, recent_folders:list[str], max_recent=2, base_directory:str=None):
    """
    Update the list of recent folders.
    Args:
        folder: New folder to add.
        recent_folders: List of recent folders.
        max_recent: Maximum number of recent folders to keep.
    """
    if folder in recent_folders:
        recent_folders.remove(folder)
    recent_folders.insert(0, folder)
    if len(recent_folders) > max_recent:
        recent_folders.pop()
    save_recent_folders(recent_folders, file_path=base_directory + r'\recent_folders.json')

class MainWindow(QMainWindow):
    progress_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.stop_event = threading.Event()

        self.base_directory = BASE_PATH
        print(self.base_directory)
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

        self.init_ui()

    def init_ui(self):
        self.folder_name_label = QLabel("Fiber Name:")
        self.folder_name_input = QLineEdit()
        self.folder_name_input.setFixedWidth(700)
        self.folder_name_input.setReadOnly(True)
        self.folder_name_input.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.folder_name_input.textChanged.connect(self.update_working_dir)
        self.folder_name_input.textChanged.connect(self.update_run_button_state)

        self.open_fiber_data_button = QPushButton("Open Fiber Data")
        self.open_fiber_data_button.clicked.connect(self.open_fiber_data_window)

        self.working_dir_label = QLabel("Working Directory:")
        self.working_dir_display = QLabel("")

        self.choose_folder_button = QPushButton("Choose Existing Folder")
        self.choose_folder_button.clicked.connect(self.choose_folder)

        self.comments_button = QPushButton("Add Comments")
        self.comments_button.clicked.connect(self.access_comments_file)
        self.comments_button.setDisabled(True)

        """self.lock_button = QPushButton("Lock In")
        #self.lock_button.clicked.connect(self.lock_inputs)

        self.unlock_button = QPushButton("Unlock")
        self.unlock_button.clicked.connect(self.unlock_inputs)
        self.unlock_button.setDisabled(True)"""

        self.recent_folders = load_recent_folders(file_path=self.base_directory + r'\recent_folders.json')
        self.recent_folders_combo = QComboBox()
        self.update_recent_folders_combo()
        self.recent_folders_combo.currentIndexChanged.connect(self.select_recent_folder)

        self.progress_label = QLabel("")
        self.progress_text_edit = QTextEdit()
        self.progress_text_edit.setReadOnly(True)
        self.progress_text_edit.hide()  # Initially hidden

        self.layout.addLayout(self.create_hbox_layout(self.folder_name_label, self.folder_name_input))
        self.layout.addLayout(self.create_hbox_layout(self.working_dir_label, self.working_dir_display))

        self.layout.addWidget(self.open_fiber_data_button)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_folder_button)
        button_layout.addWidget(self.comments_button)
        #button_layout.addWidget(self.lock_button)
        #button_layout.addWidget(self.unlock_button)
        self.layout.addLayout(button_layout)

        self.recent_folders_label = QLabel("Recent Folders:")
        self.layout.addWidget(self.recent_folders_label)

        self.layout.addWidget(self.recent_folders_combo)
        self.layout.addWidget(self.message_label)
        self.layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_text_edit)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.update_input_visibility)
        self.layout.addWidget(self.tabs)

        self.measure_tab = QWidget()
        self.analyse_tab = QWidget()
        self.general_tab = QWidget()
        self.camera_tab = QWidget()

        self.tabs.addTab(self.measure_tab, "Measure")
        self.tabs.addTab(self.analyse_tab, "Analyse")
        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.camera_tab, "Cameras")

        self.init_measure_tab()
        self.init_analyse_tab()
        self.init_general_tab()
        self.init_camera_tab()

        self.tabs.currentChanged.connect(self.update_ui_state)
        self.tabs.currentChanged.connect(self.update_run_button_state)

        self.progress_signal.connect(self.update_progress)

        self.fiber_data_window = FiberDataWindow(self)

    def update_camera_tab_buttons(self):
        if self.tabs.currentWidget() != self.camera_tab:
            return

        self.update_run_button_state()

        selected_function = self.camera_function_combo.currentText()

        if selected_function in ["Thorlabs Camera Live", "Thorlabs Camera Single"]:
            self.camera_chooser_label.show()
            self.camera_chooser_combo.show()
        else:
            self.camera_chooser_label.hide()
            self.camera_chooser_combo.hide()

    def init_camera_tab(self):
        self.update_input_visibility()

        layout = QVBoxLayout()

        self.camera_function_label = QLabel("Select Function:")
        self.camera_function_combo = QComboBox()
        self.camera_function_combo.addItems(["Thorlabs Camera Live", "Thorlabs Camera Single", "Qhyccd Camera Single"])
        self.camera_function_combo.currentIndexChanged.connect(self.update_camera_tab_buttons)

        # Create a widget for the function chooser and set its position
        function_widget = QWidget()
        function_layout = QHBoxLayout(function_widget)
        function_layout.addWidget(self.camera_function_label)
        function_layout.addWidget(self.camera_function_combo)
        function_layout.addStretch()

        layout.addWidget(function_widget)

        self.camera_chooser_label = QLabel("Camera:")
        self.camera_chooser_combo = QComboBox()
        self.camera_chooser_combo.addItems(["Entrance Cam", "Exit Cam"])
        layout.addLayout(self.create_hbox_layout(self.camera_chooser_label, self.camera_chooser_combo))

        self.exposure_time_label = QLabel("Exposure Time:")
        self.exposure_time_input = QLineEdit()
        self.exposure_time_input.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+(\.\d+)?(ms|s|us)$")))
        self.exposure_time_input.setFixedWidth(100)
        self.exposure_time_input.setText("1ms")
        self.exposure_time_input.textChanged.connect(self.check_valid_exp_input)
        layout.addLayout(self.create_hbox_layout(self.exposure_time_label, self.exposure_time_input))

        # Add a spacer item to push the button to the bottom
        layout.addStretch()

        # Add the Run button to the Camera tab
        self.run_button = QPushButton("Run")
        self.run_button.setDisabled(True)  # Initially disabled
        self.run_button.clicked.connect(self.run_camera_function)
        layout.addWidget(self.run_button)

        self.camera_tab.setLayout(layout)

    def check_valid_exp_input(self):
        if not self.exposure_time_input.hasAcceptableInput():
            self.show_message("Invalid exposure time input. Please enter a valid exposure time. E.g.: 1ms, 1s, 1us")
        else:
            self.show_message("")

    def run_camera_function(self):
        selected_function = self.camera_function_combo.currentText()
        folder_name = self.folder_name_input.text()

        if selected_function in ["Thorlabs Camera", "Qhyccd Camera"] and folder_name != "":
            self.show_message("Please enter folder name before running the function.")
            return

        if not self.exposure_time_input.hasAcceptableInput():
            return

        self.experiment_running = True
        self.update_ui_state()

        working_dir = self.working_dir_display.text()
        threading.Thread(target=self.run_camera_function_thread, args=(selected_function, working_dir)).start()

    def run_camera_function_thread(self, selected_function, working_dir):
        self.progress_signal.emit(f"Running {selected_function}...")
        if selected_function == "Thorlabs Camera Live":
            import thorlabs_cam_control
            thorlabs_cam_control.open_thorcam()
        elif selected_function == "Thorlabs Camera Single":
            import thorlabs_cam_control
            if self.camera_chooser_combo.currentText() == "Entrance Cam":
                cam_type = "entrance_cam"
                exp_time = self.exposure_time_input.text()
                image_name_path = os.path.join(working_dir, "entrance_image.fits")
                thorlabs_cam_control.take_image(cam_type, image_name_path, wait=True, exposure_time=exp_time, info=True, save_fits=True)
            elif self.camera_chooser_combo.currentText() == "Exit Cam":
                cam_type = "exit_cam"
                exp_time = self.exposure_time_input.text()
                image_name_path = os.path.join(working_dir, "exit_image.fits")
                thorlabs_cam_control.take_image(cam_type, image_name_path, wait=True, exposure_time=exp_time, info=True, save_fits=True)

        elif selected_function == "Qhyccd Camera Single":
            import qhy_ccd_take_image


            self.qhyccd_cam = qhy_ccd_take_image.Camera(1000)

            exposure_time_us = qhy_ccd_take_image.convert_to_us(self.exposure_time_input.text())
            self.qhyccd_cam.change_exposure_time(exposure_time_us)
            image_name = "qhyccd_image"
            self.qhyccd_cam.take_single_frame(working_dir, image_name, show=True)
            self.qhyccd_cam.close()

        self.progress_signal.emit(f"{selected_function} complete.")
        self.experiment_running = False
        self.update_ui_state()

    def initialize_filter_wheel(self):
        import serial.tools.list_ports
        available_ports = [port.device for port in serial.tools.list_ports.comports()]
        if 'COM5' in available_ports:
            self.filter_wheel_initiated = True
            self.filter_wheel = FilterWheel('COM5')
            self.filter_wheel_ready = True
            self.update_general_tab_buttons()
        else:
            self.show_message("COM5 is not available.")

    def open_fiber_data_window(self):
        self.fiber_data_window.fiberDataChanged.connect(self.update_fiber_data)
        self.fiber_data_window.update_window()
        self.fiber_data_window.show()

    @pyqtSlot(str, object, str)
    def update_fiber_data(self, name, dimension, shape):
        self.folder_name = name
        self.fiber_dimension = dimension
        self.fiber_shape = shape

        self.folder_name_input.setText(name)
        self.update_input_visibility()

    def insert_spacer(self, height):
        self.placeholder_spacer = QSpacerItem(20, height)
        self.layout.insertItem(self.layout.count() - 1, self.placeholder_spacer)
        self.layout.update()

    def update_input_visibility(self):
        """
        Update the visibility of the input fields based on the selected tab.
        """
        if self.folder_name != "":
            self.comments_button.setDisabled(False)
        else:
            self.comments_button.setDisabled(True)

        if self.tabs.currentWidget() == self.general_tab:
            self.open_fiber_data_button.hide()
            self.folder_name_input.setReadOnly(False)
            self.folder_name_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.recent_folders_label.hide()
            self.recent_folders_combo.hide()
            self.choose_folder_button.hide()
            self.folder_name_label.setText("Folder Name:")
            if hasattr(self, 'placeholder_spacer'):
                self.layout.removeItem(self.placeholder_spacer)
                del self.placeholder_spacer
                self.insert_spacer(82)
            else:
                self.insert_spacer(82)
            self.update_general_tab_buttons()  # Ensure buttons are correctly updated

        elif self.tabs.currentWidget() == self.camera_tab:
            self.open_fiber_data_button.hide()
            self.folder_name_input.setReadOnly(False)
            self.folder_name_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.recent_folders_label.hide()
            self.recent_folders_combo.hide()
            self.choose_folder_button.hide()
            self.folder_name_label.setText("Folder Name:")

            if hasattr(self, 'placeholder_spacer'):
                self.layout.removeItem(self.placeholder_spacer)
                del self.placeholder_spacer
                self.insert_spacer(82)
            else:
                self.insert_spacer(82)

        else:
            if hasattr(self, 'placeholder_spacer'):
                self.layout.removeItem(self.placeholder_spacer)
                del self.placeholder_spacer
                self.layout.update()

            self.folder_name_label.setText("Fiber Name:")
            self.folder_name_input.setReadOnly(True)
            self.folder_name_input.setFocusPolicy(Qt.FocusPolicy.NoFocus)

            # Show the buttons that were hidden for some functions in the general tab
            self.folder_name_label.show()
            self.folder_name_input.show()
            self.choose_folder_button.show()
            self.open_fiber_data_button.show()
            #self.lock_button.show()
            #self.unlock_button.show()
            self.comments_button.show()
            self.recent_folders_combo.show()
            self.working_dir_label.show()
            self.recent_folders_label.show()

    def create_hbox_layout(self, label, widget):
        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(widget)
        return hbox

    def update_recent_folders_combo(self):
        self.recent_folders_combo.clear()
        self.recent_folders_combo.addItem("None Selected")
        self.recent_folders_combo.addItems(self.recent_folders)
        self.recent_folders_combo.setCurrentIndex(0)

    def select_recent_folder(self, index):
        if 0 < index <= len(self.recent_folders):
            folder = self.recent_folders[index - 1]
            self.working_dir_display.setText(folder)
            self.folder_name_input.setText(os.path.basename(folder))
            self.open_fiber_data_window()
            self.fiber_data_window.load_fiber_data(folder)
            self.update_comments_button()

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

    def access_comments_file(self):
        working_dir = self.working_dir_display.text()
        if not working_dir:
            self.show_message("Please select a working directory first.")
            return

        comments_file_path = os.path.join(working_dir, "comments.txt")
        if not os.path.exists(comments_file_path):
            with open(comments_file_path, "w") as file:
                file.write("Comments:\n")
            self.comments_button.setText("Access Comments")
        else:
            if sys.platform.startswith("win"):
                os.startfile(comments_file_path)
            elif sys.platform.startswith("linux"):
                subprocess.call(["xdg-open", comments_file_path])

    def update_comments_button(self):
        working_dir = self.working_dir_display.text()
        comments_file_path = os.path.join(working_dir, "comments.txt")
        if os.path.exists(comments_file_path) and os.path.getsize(comments_file_path) > 0:
            self.comments_button.setText("Access Comments")
        else:
            self.comments_button.setText("Add Comments")

    def update_progress(self, message):
        if not self.progress_text_edit.isVisible():
            self.progress_text_edit.show()
        self.progress_text_edit.append(message)

        # Calculate the height based on the number of lines
        line_height = self.progress_text_edit.fontMetrics().height()
        num_lines = self.progress_text_edit.document().blockCount()
        max_lines = 10
        new_height = min(num_lines, max_lines) * line_height + 10  # Add some padding

        # Set the new height
        self.progress_text_edit.setFixedHeight(new_height)

    def update_working_dir(self):
        fiber_name = self.folder_name_input.text()
        if fiber_name:
            working_dir = os.path.join(self.base_directory, fiber_name)
            self.working_dir_display.setText(working_dir)
        else:
            self.working_dir_display.setText("")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.base_directory)
        if folder:
            self.working_dir_display.setText(folder)
            self.folder_name_input.setText(os.path.basename(folder))

            self.open_fiber_data_window()
            self.fiber_data_window.load_fiber_data(folder)

            """if self.inputs_locked:
                self.check_existing_measurements(folder)"""

            """# Check if all required inputs are filled before locking
            if self.folder_name and self.fiber_dimension and self.fiber_shape != "None":
                return
                #self.lock_inputs()
            else:
                self.show_message("Please enter fiber name, diameter, length, and shape before locking inputs.")"""

            self.update_comments_button()
            update_recent_folders(folder, self.recent_folders, base_directory=self.base_directory)
            self.update_recent_folders_combo()

    def save_fiber_data(self, folder, fiber_diameter, fiber_shape, fiber_length):
        file_path = os.path.join(folder, "fiber_data.json")
        with open(file_path, "w") as file:
            # noinspection PyTypeChecker
            json.dump({"fiber_diameter": fiber_diameter, "fiber_shape": fiber_shape,
                       "fiber_length": fiber_length}, file)

    def show_message(self, message):
        self.message_label.setText(message)

    def lock_inputs(self):
        if self.tabs.currentWidget() == self.general_tab:
            folder_name = self.folder_name_input.text()
            if not folder_name:
                self.show_message("Please enter a folder name.")
                return
            working_dir = os.path.join(self.base_directory, folder_name)
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
            self.working_dir_display.setText(working_dir)
        else:
            if self.fiber_shape == "None":
                self.show_message("Fiber data not set. Please open the fiber data window and set the fiber data.")
                return

            if any(value == "" for value in [self.fiber_name, self.fiber_dimension]):
                self.show_message("Fiber data not set. Please open the fiber data window and set the fiber data.")
                return

            working_dir = self.working_dir_display.text()
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)

        self.folder_name_input.setDisabled(True)
        self.inputs_locked = True
        self.show_message("Inputs locked.")
        self.run_measurement_button.setDisabled(True)
        self.run_analysis_button.setDisabled(True)
        self.check_existing_measurements(working_dir)
        self.update_checklist()

        # Save fiber data to JSON


        # Update all run buttons
        self.update_measurement_button_state()
        self.update_run_button_state()

        self.comments_button.setDisabled(False)
        self.update_comments_button()

        # Update the UI state
        self.update_ui_state()

        self.unlock_button.setDisabled(False)
        self.lock_button.setDisabled(True)
        self.update_run_button_state()

    def unlock_inputs(self):
        self.folder_name_input.setDisabled(False)
        self.fiber_diameter_input.setDisabled(False)
        self.fiber_height_input.setDisabled(False)
        self.fiber_width_input.setDisabled(False)
        self.fiber_length_input.setDisabled(False)
        self.fiber_shape_combo.setDisabled(False)
        self.inputs_locked = False
        self.show_message("Inputs unlocked.")
        self.run_measurement_button.setDisabled(True)
        self.run_analysis_button.setDisabled(True)
        self.existing_measurements_label.setText("")
        self.comments_button.setDisabled(True)
        self.lock_button.setDisabled(False)
        self.unlock_button.setDisabled(True)
        self.update_run_button_state()

    def update_run_button_state(self):
        if self.tabs.currentWidget() == self.general_tab:
            return

        if self.tabs.currentWidget() == self.camera_tab:
            if self.folder_name_input.text() == "":
                self.run_button.setDisabled(True)
            else:
                self.run_button.setDisabled(False)
            return

        if self.folder_name and self.fiber_shape and self.fiber_dimension != "":
            if self.analysis_type_combo.currentText() == "Throughput":
                self.run_analysis_button.setDisabled(False)

            elif self.analysis_type_combo.currentText() == "SG" or self.analysis_type_combo.currentText() == "FRD":
                if (self.plot_sg_checkbox.isChecked() or self.calc_sg_checkbox.isChecked()
                        or self.plot_coms_checkbox.isChecked() or self.get_params_checkbox.isChecked()
                        or self.plot_masks_checkbox.isChecked() or self.make_video_checkbox.isChecked()
                        or self.sg_new_checkbox.isChecked() or self.calc_frd_checkbox.isChecked()
                        or self.plot_sutherland_checkbox.isChecked() or self.plot_f_ratio_circles_on_raw_checkbox.isChecked()
                ):
                    self.run_analysis_button.setDisabled(False)
                else:
                    self.run_analysis_button.setDisabled(True)
            else:
                self.run_analysis_button.setDisabled(True)

        else:
            self.run_analysis_button.setDisabled(True)

    def init_general_tab(self):
        layout = QVBoxLayout()

        self.general_function_label = QLabel("Select Function:")
        self.general_function_combo = QComboBox()
        self.general_function_combo.addItems(
            ["Measure System F-ratio", "Make Throughput Calibration", "Adjust Tip/Tilt",
             "Motor Controller: Reference", "Motor Controller: Move to Position",
             "Measure Eccentricity", "FF with each Filter", "Change Color Filter", "Change System F-ratio"])

        # Create a widget for the function chooser and set its position
        function_widget = QWidget()
        function_layout = QHBoxLayout(function_widget)
        function_layout.addWidget(self.general_function_label)
        function_layout.addWidget(self.general_function_combo)
        function_layout.addStretch()

        layout.addWidget(function_widget)

        # Number input for the motor controller
        self.number_input_label = QLabel("Position to move to (0 - 9.9) in mm:")
        self.number_input = QLineEdit()
        self.number_input.setText("0")
        self.number_input.setFixedWidth(100)

        layout.addLayout(self.create_hbox_layout(self.number_input_label, self.number_input))
        # Initially hidden
        self.number_input_label.hide()
        self.number_input.hide()

        # Filter input for color filter wheel
        self.filter_input_label = QLabel("Color Filter:")
        self.filter_input_combo = QComboBox()
        self.filter_input_combo.addItems(["Open", "Closed", "400", "450", "500", "600", "700", "800"])
        layout.addLayout(self.create_hbox_layout(self.filter_input_label, self.filter_input_combo))
        # Initially hidden
        self.filter_input_label.hide()
        self.filter_input_combo.hide()

        # Exposure time input for the camera
        self.exposure_time_label_gt = QLabel("Exposure Time:") # gt = General Tab
        self.exposure_time_input_gt = QLineEdit()
        self.exposure_time_input_gt.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+(\.\d+)?(ms|s|us)$")))
        self.exposure_time_input_gt.setFixedWidth(100)
        self.exposure_time_input_gt.setText("70ms")
        layout.addLayout(self.create_hbox_layout(self.exposure_time_label_gt, self.exposure_time_input_gt))
        # Initially hidden
        self.exposure_time_label_gt.hide()
        self.exposure_time_input_gt.hide()

        # F-ratio input for the system
        self.fratio_input_label = QLabel("F-ratio:")
        self.fratio_input_combo = QComboBox()
        self.fratio_input_combo.addItems(["2.5", "3.5", "4.0", "4.5", "5.0", "6.0"])
        layout.addLayout(self.create_hbox_layout(self.fratio_input_label, self.fratio_input_combo))
        # Initially hidden
        self.fratio_input_label.hide()
        self.fratio_input_combo.hide()

        # Add a spacer item to push the button to the bottom
        layout.addStretch()

        self.stop_button = QPushButton("Stop")
        self.stop_button.setDisabled(True)
        self.stop_button.clicked.connect(self.stop_general_function)
        layout.addWidget(self.stop_button)

        # Add the Run button to the General tab
        self.run_button = QPushButton("Run")
        self.run_button.setDisabled(False)  # Initially disabled
        self.run_button.clicked.connect(self.run_general_function)
        layout.addWidget(self.run_button)

        self.general_tab.setLayout(layout)

        # Connect the signal and update the stop button visibility
        self.general_function_combo.currentIndexChanged.connect(self.update_general_tab_buttons)
        self.folder_name_input.textChanged.connect(self.update_general_tab_buttons)
        self.update_general_tab_buttons()

    def update_general_tab_buttons(self):
        if self.tabs.currentWidget() != self.general_tab:
            return

        selected_function = self.general_function_combo.currentText()

        if selected_function == "Measure System F-ratio":
            self.exposure_time_label_gt.show()
            self.exposure_time_input_gt.show()
        else:
            self.exposure_time_label_gt.hide()
            self.exposure_time_input_gt.hide()

        if selected_function in ["Adjust Tip/Tilt", "Measure Eccentricity"]:
            self.stop_button.show()
        else:
            self.stop_button.hide()

        if selected_function == "Motor Controller: Move to Position":
            self.number_input_label.show()
            self.number_input.show()
        else:
            self.number_input_label.hide()
            self.number_input.hide()

        if selected_function == "Change Color Filter":
            self.filter_input_label.show()
            self.filter_input_combo.show()
        else:
            self.filter_input_label.hide()
            self.filter_input_combo.hide()

        if selected_function == "Change System F-ratio":
            # Initialize the filter wheel here so that it is only initialized once
            if not self.filter_wheel_initiated:
                threading.Thread(target=self.initialize_filter_wheel).start()
            self.fratio_input_label.show()
            self.fratio_input_combo.show()

            if self.filter_wheel_ready:
                self.fratio_input_combo.setDisabled(False)
                self.run_button.setDisabled(False)
            else:
                self.fratio_input_combo.setDisabled(True)
                self.run_button.setDisabled(True)

        else:
            self.fratio_input_label.hide()
            self.fratio_input_combo.hide()

        if selected_function in ["Motor Controller: Reference", "Motor Controller: Move to Position",
                                 "Measure Eccentricity", "Adjust Tip/Tilt", "Change Color Filter",
                                 "Change System F-ratio"]:
            if hasattr(self, 'placeholder_spacer'):
                self.layout.removeItem(self.placeholder_spacer)
                del self.placeholder_spacer
                self.insert_spacer(140)
            else:
                self.insert_spacer(140)

            self.folder_name_label.hide()
            self.folder_name_input.hide()
            #self.lock_button.hide()
            #self.unlock_button.hide()
            self.comments_button.hide()
            self.working_dir_label.hide()
            if selected_function != "Change System F-ratio":
                self.run_button.setDisabled(False)

        else:
            if hasattr(self, 'placeholder_spacer'):
                self.layout.removeItem(self.placeholder_spacer)
                del self.placeholder_spacer
                self.insert_spacer(82)
            else:
                self.insert_spacer(82)

            self.folder_name_label.show()
            self.folder_name_input.show()
            #self.lock_button.show()
            #self.unlock_button.show()
            self.comments_button.show()
            self.working_dir_label.show()

            self.run_button.setDisabled(self.folder_name_input.text() == "")

        if selected_function == "Make Throughput Calibration":
            # Change the folder name label to calibration name
            self.folder_name_label.setText("Calibration Name:")
        else:
            self.folder_name_label.setText("Folder Name:")

        self.run_button.setEnabled(True)


    def stop_general_function(self):
        self.stop_event.set()
        self.progress_signal.emit("Stopping function...")

    def run_general_function(self):
        selected_function = self.general_function_combo.currentText()

        if (selected_function in ["Measure Eccentricity", "Adjust Tip/Tilt", "FF with each Filter"]
                and self.folder_name != ""):
            self.show_message("Please enter folder name before running the function.")
            return


        self.experiment_running = True
        self.update_ui_state()

        working_dir = self.working_dir_display.text()
        threading.Thread(target=self.run_general_function_thread, args=(selected_function, working_dir)).start()

    def run_general_function_thread(self, selected_function, working_dir):
        self.progress_signal.emit(f"Running {selected_function}...")
        self.stop_event = threading.Event()
        if selected_function == "Measure System F-ratio":
            import fiber_frd_measurements as frd
            #import qhy_ccd_take_image as qhy
            #exposure_time = qhy.convert_to_us(self.exposure_time_input_gt.text())
            #import analyse_main as am
            #am.main_measure_new(working_dir, progress_signal=self.progress_signal, exposure_time)
            frd.main_measure_all_filters(working_dir, progress_signal=self.progress_signal, base_directory=self.base_directory)
            frd.main_analyse_all_filters(working_dir, progress_signal=self.progress_signal)
        elif selected_function == "Make Throughput Calibration":
            import throughput_analysis as ta
            calibration_folder_name = os.path.basename(working_dir)
            ta.measure_all_filters(working_dir, progress_signal=self.progress_signal, calibration=calibration_folder_name,
                                   base_directory=self.base_directory)
        elif selected_function == "Adjust Tip/Tilt":
            import qhyccd_cam_control
            qhyccd_cam_control.use_camera("tiptilt", self.stop_event)
        elif selected_function == "Motor Controller: Reference":
            import step_motor_control as smc
            smc.make_reference_move()
            smc.close_connection()
        elif selected_function == "Motor Controller: Move to Position":
            import step_motor_control as smc
            position = float(self.number_input.text())
            smc.move_motor_to_position(position)
            smc.close_connection()
        elif selected_function == "Measure Eccentricity":
            import qhyccd_cam_control
            qhyccd_cam_control.use_camera("eccentricity", self.stop_event)
        elif selected_function == "FF with each Filter":
            import general_functions
            general_functions.get_ff_with_all_filters(working_dir)
        elif selected_function == "Change Color Filter":
            import move_to_filter
            filter_name = self.filter_input_combo.currentText()
            print(filter_name)
            move_to_filter.move(filter_name)
        elif selected_function == "Change System F-ratio":
            # Check if the filter wheel is available
            if self.filter_wheel:
                f_ratio = self.fratio_input_combo.currentText()
                self.filter_wheel.move_to_filter(f_ratio)

        self.progress_signal.emit(f"{selected_function} complete.")
        self.experiment_running = False
        self.update_ui_state()

    def init_measure_tab(self):
        layout = QVBoxLayout()

        self.measurement_type_label = QLabel("Measurement Type:")
        self.measurement_type_combo = QComboBox()
        self.measurement_type_combo.addItems(["SG", "FRD", "Throughput"])
        self.measurement_type_combo.currentIndexChanged.connect(self.update_checklist)

        self.exposure_time_label_mt = QLabel("Exposure Time:") # mt = measurement tab
        self.exposure_time_input_mt = QLineEdit()
        self.exposure_time_input_mt.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+(\.\d+)?(ms|s|us)$")))
        self.exposure_time_input_mt.setFixedWidth(100)
        self.exposure_time_input_mt.setText("70ms")

        # Create a widget for the measurement type chooser and set its position
        measurement_type_widget = QWidget()
        measurement_type_layout = QHBoxLayout(measurement_type_widget)
        measurement_type_layout.addWidget(self.measurement_type_label)
        measurement_type_layout.addWidget(self.measurement_type_combo)
        measurement_type_layout.addStretch()
        measurement_type_layout.addWidget(self.exposure_time_label_mt)
        measurement_type_layout.addWidget(self.exposure_time_input_mt)

        layout.addWidget(measurement_type_widget)

        self.checklist_label = QLabel("Checklist:")
        self.checklist_label.setStyleSheet("font-weight: bold;")
        self.check1 = QCheckBox("Fiber in place: Output at small camera")
        self.check2 = QCheckBox("Input spot in center and in focus. Exit camera fiber also in focus")
        self.check3 = QCheckBox("ThorCam software closed")
        self.check4 = QCheckBox("Lights Out")
        self.check5 = QCheckBox("Check 5")

        self.check1.stateChanged.connect(self.update_measurement_button_state)
        self.check2.stateChanged.connect(self.update_measurement_button_state)
        self.check3.stateChanged.connect(self.update_measurement_button_state)
        self.check4.stateChanged.connect(self.update_measurement_button_state)
        self.check5.stateChanged.connect(self.update_measurement_button_state)

        layout.addWidget(self.checklist_label)
        layout.addWidget(self.check1)
        layout.addWidget(self.check2)
        layout.addWidget(self.check3)
        layout.addWidget(self.check4)
        layout.addWidget(self.check5)

        self.existing_measurements_label = QLabel("")
        self.existing_measurements_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.existing_measurements_label)

        # Add a spacer item to push the run_measurement_button to the bottom
        layout.addStretch()

        self.run_measurement_button = QPushButton("Run Measurement")
        self.run_measurement_button.setDisabled(True)
        self.run_measurement_button.clicked.connect(self.run_measurement)
        layout.addWidget(self.run_measurement_button)

        self.measure_tab.setLayout(layout)

        # Update the checklist based on the default selected measurement type
        self.update_checklist()

    def init_analyse_tab(self):
        layout = QVBoxLayout()

        self.analysis_type_label = QLabel("Analysis Type:")
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["SG", "FRD", "Throughput"])
        self.analysis_type_combo.currentIndexChanged.connect(self.update_analysis_tab)

        analysis_type_widget = QWidget()
        analysis_type_layout = QHBoxLayout(analysis_type_widget)
        analysis_type_layout.addWidget(self.analysis_type_label)
        analysis_type_layout.addWidget(self.analysis_type_combo)
        analysis_type_layout.addStretch()

        layout.addWidget(analysis_type_widget)

        self.plot_sg_checkbox = QCheckBox("Plot SG (Deprecated)")
        self.calc_sg_checkbox = QCheckBox("Calc SG (Deprecated)")
        self.plot_coms_checkbox = QCheckBox("Plot COMs")
        self.get_params_checkbox = QCheckBox("Get Parameters (Must have been run for everything else to work)")
        self.plot_masks_checkbox = QCheckBox("Plot Masks")
        self.make_video_checkbox = QCheckBox("Make Video")
        self.sg_new_checkbox = QCheckBox("Plot and Calculate SG")
        self.calc_frd_checkbox = QCheckBox("Calculate FRD (Must have been run for everything else to work)")
        self.plot_sutherland_checkbox = QCheckBox("Make Sutherland Plot")
        self.plot_f_ratio_circles_on_raw_checkbox = QCheckBox("Plot F-ratio Circles on Raw Image")

        self.plot_sg_checkbox.stateChanged.connect(self.update_run_button_state)
        self.calc_sg_checkbox.stateChanged.connect(self.update_run_button_state)
        self.plot_coms_checkbox.stateChanged.connect(self.update_run_button_state)
        self.get_params_checkbox.stateChanged.connect(self.update_run_button_state)
        self.plot_masks_checkbox.stateChanged.connect(self.update_run_button_state)
        self.make_video_checkbox.stateChanged.connect(self.update_run_button_state)
        self.sg_new_checkbox.stateChanged.connect(self.update_run_button_state)
        self.calc_frd_checkbox.stateChanged.connect(self.update_run_button_state)
        self.plot_sutherland_checkbox.stateChanged.connect(self.update_run_button_state)
        self.plot_f_ratio_circles_on_raw_checkbox.stateChanged.connect(self.update_run_button_state)

        self.calibration_folder_label = QLabel("Calibration Folder:")
        self.calibration_folder_input = QLineEdit()
        self.calibration_folder_input.textChanged.connect(self.update_run_button_state)
        self.calibration_folder_button = QPushButton("Choose Calibration Folder")
        self.calibration_folder_button.clicked.connect(self.choose_calibration_folder)

        layout.addWidget(self.get_params_checkbox)
        layout.addWidget(self.plot_sg_checkbox)
        layout.addWidget(self.calc_sg_checkbox)
        layout.addWidget(self.plot_coms_checkbox)
        layout.addWidget(self.plot_masks_checkbox)
        layout.addWidget(self.make_video_checkbox)
        layout.addWidget(self.sg_new_checkbox)
        layout.addWidget(self.calc_frd_checkbox)
        layout.addWidget(self.plot_sutherland_checkbox)
        layout.addWidget(self.plot_f_ratio_circles_on_raw_checkbox)

        layout.addWidget(self.calibration_folder_label)
        layout.addWidget(self.calibration_folder_input)
        layout.addWidget(self.calibration_folder_button)

        layout.addStretch()

        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.setDisabled(True)
        self.run_analysis_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_analysis_button)

        self.analyse_tab.setLayout(layout)
        self.update_analysis_tab()

    def update_analysis_tab(self):
        analysis_type = self.analysis_type_combo.currentText()

        # Reset all checkboxes
        for checkbox in [self.plot_sg_checkbox, self.calc_sg_checkbox, self.plot_coms_checkbox,
                         self.get_params_checkbox, self.plot_masks_checkbox, self.make_video_checkbox,
                         self.sg_new_checkbox, self.calc_frd_checkbox, self.plot_sutherland_checkbox,
                         self.plot_f_ratio_circles_on_raw_checkbox]:
            checkbox.setChecked(False)

        if analysis_type == "SG":
            self.plot_sg_checkbox.show()
            self.calc_sg_checkbox.show()
            self.plot_coms_checkbox.show()
            self.get_params_checkbox.show()
            self.plot_masks_checkbox.show()
            self.make_video_checkbox.show()
            self.sg_new_checkbox.show()
            self.calc_frd_checkbox.hide()
            self.plot_sutherland_checkbox.hide()
            self.plot_f_ratio_circles_on_raw_checkbox.hide()
            self.calibration_folder_label.hide()
            self.calibration_folder_input.hide()
            self.calibration_folder_button.hide()
        elif analysis_type == "FRD":
            # Add options for FRD analysis here
            self.plot_sg_checkbox.hide()
            self.calc_sg_checkbox.hide()
            self.plot_coms_checkbox.hide()
            self.get_params_checkbox.hide()
            self.plot_masks_checkbox.hide()
            self.make_video_checkbox.hide()
            self.sg_new_checkbox.hide()
            self.calc_frd_checkbox.show()
            self.plot_sutherland_checkbox.show()
            self.plot_f_ratio_circles_on_raw_checkbox.show()
            self.calibration_folder_label.hide()
            self.calibration_folder_input.hide()
            self.calibration_folder_button.hide()
        elif analysis_type == "Throughput":
            self.plot_sg_checkbox.hide()
            self.calc_sg_checkbox.hide()
            self.plot_coms_checkbox.hide()
            self.get_params_checkbox.hide()
            self.plot_masks_checkbox.hide()
            self.make_video_checkbox.hide()
            self.sg_new_checkbox.hide()
            self.calc_frd_checkbox.hide()
            self.plot_sutherland_checkbox.hide()
            self.plot_f_ratio_circles_on_raw_checkbox.hide()
            self.calibration_folder_label.show()
            self.calibration_folder_input.show()
            self.calibration_folder_button.show()

    def choose_calibration_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Calibration Folder",
                                                       self.base_directory + "/Calibration")
        if folder_path:
            self.calibration_folder_input.setText(folder_path)

    def check_existing_measurements(self, folder):
        measurements = []
        if os.path.exists(os.path.join(folder, "FRD")):
            measurements.append("FRD")
        if os.path.exists(os.path.join(folder, "SG")):
            measurements.append("SG")
        if os.path.exists(os.path.join(folder, "Throughput")):
            measurements.append("Throughput")

        if measurements:
            self.existing_measurements_label.setText(f"Measurements already done: {', '.join(measurements)}")
        else:
            self.existing_measurements_label.setText("No measurements done yet.")

    def update_checklist(self):
        measurement_type = self.measurement_type_combo.currentText()

        self.check1.setChecked(False)
        self.check2.setChecked(False)
        self.check3.setChecked(False)
        self.check4.setChecked(False)
        self.check5.setChecked(False)

        self.check4.show()
        self.check5.show()

        if measurement_type == "SG":
            self.check1.setText("Fiber in place: Output at small camera")
            self.check2.setText("Input spot in center and in focus and oriented horizontally. Exit camera fiber also in focus")
            self.check3.setText("ThorCam software closed")
            self.check4.setText("Motor controller plugged in")
            self.check5.setText("Lights Out")

            self.exposure_time_label_mt.hide()
            self.exposure_time_input_mt.hide()

        elif measurement_type == "FRD":
            self.check1.setText("Fiber in place: Output at large camera")
            self.check2.setText("Spot on Fiber")
            self.check3.setText("Camera enabled and max counts in range")
            self.check4.setText("ThorCam/N.I.N.A closed")
            self.check5.setText("Lights Out")

            self.exposure_time_label_mt.show()
            self.exposure_time_input_mt.show()

        elif measurement_type == "Throughput":
            self.check1.setText("Fiber in place: Output on Photodetector")
            self.check2.setText("Spot on fiber")
            self.check3.setText("Lights Out")
            self.check4.hide()
            self.check5.hide()
            self.check4.setChecked(True)
            self.check5.setChecked(True)

            self.exposure_time_label_mt.hide()
            self.exposure_time_input_mt.hide()

        self.update_measurement_button_state()

    def update_measurement_button_state(self):
        if (self.folder_name and self.fiber_dimension and self.fiber_shape != ""
                and self.check1.isChecked()
                and self.check2.isChecked() and self.check3.isChecked()
                and self.check4.isChecked() and self.check5.isChecked()
        ):
            self.run_measurement_button.setDisabled(False)
        else:
            self.run_measurement_button.setDisabled(True)

    def run_measurement(self):
        if self.folder_name and self.fiber_shape and self.fiber_dimension == "":
            self.show_message("Please enter fiber data before running the measurement.")

        if not (self.check1.isChecked() and self.check2.isChecked() and self.check3.isChecked()
                and self.check4.isChecked() and self.check5.isChecked()
        ):
            self.show_message("Please complete all checklist items before running the measurement.")
            return

        fiber_name = self.folder_name
        fiber_shape = self.fiber_shape

        if fiber_shape == "rectangular":
            fiber_diameter = (int(self.fiber_dimension[0]), int(self.fiber_dimension[1]))
        else:
            fiber_diameter = int(self.fiber_dimension)

        measurement_type = self.measurement_type_combo.currentText()
        working_dir = self.working_dir_display.text()

        self.experiment_running = True
        self.update_ui_state()

        threading.Thread(target=self.run_measurement_thread, args=(measurement_type, working_dir, fiber_diameter,
                                                                   fiber_shape)).start()

    def run_measurement_thread(self, measurement_type, working_dir, fiber_diameter, fiber_shape):
        self.progress_signal.emit("Starting measurement...")
        # Call the appropriate measurement function
        if measurement_type == "SG":
            working_dir = os.path.join(working_dir, "SG")
            self.measure_sg(working_dir, fiber_diameter, fiber_shape)
        elif measurement_type == "FRD":
            working_dir = os.path.join(working_dir, "FRD")
            self.measure_frd(working_dir, fiber_diameter, fiber_shape)
        elif measurement_type == "Throughput":
            working_dir = os.path.join(working_dir, "Throughput")
            self.measure_throughput(working_dir, fiber_diameter, fiber_shape)


        self.progress_signal.emit("Measurement complete.")
        self.experiment_running = False
        self.update_ui_state()

    def run_analysis(self):
        if not self.folder_name and self.fiber_dimension and self.fiber_shape != "":
            self.show_message("Please lock the inputs before running the analysis.")
            return

        analysis_type = self.analysis_type_combo.currentText()
        working_dir = self.working_dir_display.text()
        fiber_shape = self.fiber_shape

        if fiber_shape == "rectangular":
            fiber_diameter = (int(self.fiber_dimension[0]), int(self.fiber_dimension[1]))
        else:
            fiber_diameter = int(self.fiber_dimension)

        calibration_folder = self.calibration_folder_input.text() if analysis_type == "Throughput" else None

        self.experiment_running = True
        self.update_ui_state()

        threading.Thread(target=self.run_analysis_thread,
                         args=(analysis_type, working_dir, fiber_diameter, fiber_shape, calibration_folder)).start()

    def run_analysis_thread(self, analysis_type, working_dir, fiber_diameter, fiber_shape, calibration_folder):
        self.progress_signal.emit("Starting analysis...")
        if analysis_type == "SG":
            directory = os.path.join(working_dir, "SG")
            import sg_pipeline
            if self.get_params_checkbox.isChecked():
                print("Getting SG parameters with fiber diameter:", fiber_diameter, "and fiber shape:", fiber_shape)
                sg_pipeline.get_sg_params(directory, fiber_diameter, fiber_shape, progress_signal=self.progress_signal)

            if self.plot_sg_checkbox.isChecked():
                sg_pipeline.plot_sg_cool_like(directory, fiber_diameter, progress_signal=self.progress_signal)

            if self.calc_sg_checkbox.isChecked():
                sg_pipeline.calc_sg(directory, progress_signal=self.progress_signal)

            if self.plot_coms_checkbox.isChecked():
                sg_pipeline.plot_coms(directory, progress_signal=self.progress_signal)

            if self.plot_masks_checkbox.isChecked():
                sg_pipeline.plot_masks(directory, fiber_diameter, progress_signal=self.progress_signal)

            if self.make_video_checkbox.isChecked():
                sg_pipeline.make_comparison_video(directory, fiber_diameter)

            if self.sg_new_checkbox.isChecked():
                sg_pipeline.sg_new(directory, progress_signal=self.progress_signal)

        elif analysis_type == "FRD":
            directory = os.path.join(working_dir, "FRD")
            import fiber_frd_measurements as frd
            if self.calc_frd_checkbox.isChecked():
                frd.main_analyse_all_filters(directory, progress_signal=self.progress_signal)
            if self.plot_sutherland_checkbox.isChecked():
                frd.sutherland_plot(directory)
            if self.plot_f_ratio_circles_on_raw_checkbox.isChecked():
                frd.plot_f_ratio_circles_on_raw(directory)

        elif analysis_type == "Throughput":
            directory = os.path.join(working_dir, "Throughput")
            import throughput_analysis
            print(calibration_folder)
            throughput_analysis.main(directory, calibration_folder)

        self.progress_signal.emit("Analysis complete.")
        self.experiment_running = False
        self.update_ui_state()

    def update_ui_state(self):
        state = self.experiment_running
        if state:
            self.run_analysis_button.setDisabled(True)
            self.run_measurement_button.setDisabled(True)
            self.choose_folder_button.setDisabled(True)

    def measure_sg(self, working_dir, fiber_diameter, fiber_shape):
        self.show_message(f"Running SG measurement with working dir: {working_dir}, fiber diameter: {fiber_diameter}, and fiber shape: {fiber_shape}")
        import sg_pipeline
        sg_pipeline.capture_images_and_reduce(working_dir, fiber_diameter, progress_signal=self.progress_signal)

    def measure_frd(self, working_dir, fiber_diameter, fiber_shape):
        #import fiber_frd_measurements
        self.show_message(f"Running FRD measurement with working dir: {working_dir}")
        import analyse_main as am
        import qhy_ccd_take_image as qhy
        exposure_time = qhy.convert_to_us(self.exposure_time_input_mt.text())
        am.main_measure_new(working_dir, progress_signal=self.progress_signal, exp_time=exposure_time)
        #fiber_frd_measurements.main_measure_all_filters(working_dir, progress_signal=self.progress_signal, base_directory=self.base_directory)

    def measure_throughput(self, working_dir, fiber_diameter, fiber_shape):
        self.show_message(f"Running Throughput measurement with working dir: {working_dir}")
        import throughput_analysis
        throughput_analysis.measure_all_filters(working_dir, progress_signal=self.progress_signal, base_directory=self.base_directory)

class FiberDataWindow(QDialog):
    fiberDataChanged = pyqtSignal(str, object, str)  # Define a signal

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fiber Data")
        self.setGeometry(150, 150, 400, 300)
        self.layout = QVBoxLayout(self)

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: red; font-weight: bold;")
        self.layout.addWidget(self.message_label)

        self.fiber_name_label = QLabel("Fiber Name:")
        self.fiber_name_input = QLineEdit()
        self.fiber_name_input.textChanged.connect(lambda: self.check_folder_exists())
        self.fiber_name_input.setFixedWidth(300)

        self.fiber_diameter_label = QLabel("Fiber Diameter (µm):")
        self.fiber_diameter_input = QLineEdit()
        self.fiber_diameter_input.setFixedWidth(300)

        self.fiber_width_label = QLabel("Fiber Width (µm):")
        self.fiber_width_input = QLineEdit()
        self.fiber_width_input.setFixedWidth(300)
        self.fiber_width_label.hide()
        self.fiber_width_input.hide()

        self.fiber_height_label = QLabel("Fiber Height (µm):")
        self.fiber_height_input = QLineEdit()
        self.fiber_height_input.setFixedWidth(300)
        self.fiber_height_label.hide()
        self.fiber_height_input.hide()

        self.fiber_length_label = QLabel("Fiber Length (m):")
        self.fiber_length_input = QLineEdit()
        self.fiber_length_input.setFixedWidth(300)

        self.fiber_shape_label = QLabel("Fiber Shape:")
        self.fiber_shape_combo = QComboBox()
        self.fiber_shape_combo.addItems(["None", "circular", "octagonal", "rectangular"])
        self.fiber_shape_combo.setFixedWidth(300)
        self.fiber_shape_combo.currentIndexChanged.connect(self.update_fiber_shape_inputs)

        self.numerical_aperature_label = QLabel("Numerical Aperature:")
        self.numerical_aperature_input = QLineEdit()
        self.numerical_aperature_input.setFixedWidth(300)

        self.coating_type_label = QLabel("Coating Type:")
        self.coating_type_combo = QComboBox()
        self.coating_type_combo.setEditable(True)
        self.coating_type_combo.addItems(["Polyimide", "Acrylate"])  # Add your predefined options here
        self.coating_type_combo.setFixedWidth(300)

        self.manufacturer_label = QLabel("Manufacturer:")
        self.manufacturer_combo = QComboBox()
        self.manufacturer_combo.setEditable(True)
        self.manufacturer_combo.addItems(["Thorlabs", "Option 2", "Option 3", "Option 4"])  # Add your predefined options here
        self.manufacturer_combo.setFixedWidth(300)

        self.save_button = QPushButton("Save and close")
        self.save_button.clicked.connect(self.check_inputs_and_save)

        self.layout.addWidget(self.fiber_name_label)
        self.layout.addWidget(self.fiber_name_input)
        self.layout.addWidget(self.fiber_shape_label)
        self.layout.addWidget(self.fiber_shape_combo)
        self.layout.addWidget(self.fiber_diameter_label)
        self.layout.addWidget(self.fiber_diameter_input)
        self.layout.addWidget(self.fiber_width_label)
        self.layout.addWidget(self.fiber_width_input)
        self.layout.addWidget(self.fiber_height_label)
        self.layout.addWidget(self.fiber_height_input)
        self.layout.addWidget(self.fiber_length_label)
        self.layout.addWidget(self.fiber_length_input)
        self.layout.addWidget(self.numerical_aperature_label)
        self.layout.addWidget(self.numerical_aperature_input)
        self.layout.addWidget(self.coating_type_label)
        self.layout.addWidget(self.coating_type_combo)
        self.layout.addWidget(self.manufacturer_label)
        self.layout.addWidget(self.manufacturer_combo)

        self.layout.addWidget(self.save_button)

        self.fiber_dimension = ""
        self.fiber_shape = ""
        self.fiber_length = ""
        self.fiber_name = ""
        self.numerical_aperature = ""
        self.coating_type = ""
        self.manufacturer = ""

        self.update_from_load_token = False

    def update_window(self):
        self.show_message("")
        self.fiber_name_input.setText(self.fiber_name)
        self.fiber_length_input.setText(self.fiber_length)
        self.fiber_shape_combo.setCurrentText(self.fiber_shape)
        self.numerical_aperature_input.setText(self.numerical_aperature)
        self.coating_type_combo.setCurrentText(self.coating_type)
        self.manufacturer_combo.setCurrentText(self.manufacturer)
        self.update_fiber_shape_inputs()
        if self.fiber_shape == "rectangular":
            self.fiber_width_input.setText(str(self.fiber_dimension[0]))
            self.fiber_height_input.setText(str(self.fiber_dimension[1]))
        else:
            self.fiber_diameter_input.setText(str(self.fiber_dimension))
        self.check_folder_exists()

    def save_fiber_data(self, folder):
        file_path = os.path.join(folder, "fiber_data.json")
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file_path, "w") as file:
            # noinspection PyTypeChecker
            json.dump({"fiber_name": os.path.basename(folder), "fiber_dimension": self.fiber_dimension, "fiber_shape": self.fiber_shape,
                       "fiber_length": self.fiber_length, "numerical_aperature": self.numerical_aperature, "coating_type": self.coating_type,
                          "manufacturer": self.manufacturer}, file)

    def load_fiber_data(self, folder):
        file_path = os.path.join(folder, "fiber_data.json")
        self.update_from_load_token = True
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
                fiber_dimension = data.get("fiber_dimension", "")
                fiber_shape = data.get("fiber_shape", "")
                fiber_length = data.get("fiber_length", "")
                numerical_aperature = data.get("numerical_aperature", "")
                coating_type = data.get("coating_type", "")
                manufacturer = data.get("manufacturer", "")

                self.fiber_shape = fiber_shape if fiber_shape else ""
                self.fiber_dimension = fiber_dimension if fiber_dimension else ""
                self.fiber_length = fiber_length if fiber_length else ""
                self.numerical_aperature = numerical_aperature if numerical_aperature else ""
                self.coating_type = coating_type if coating_type else ""
                self.manufacturer = manufacturer if manufacturer else ""
        else:
            self.show_message("No fiber data file found.")

        self.fiber_name_input.setText(os.path.basename(folder))
        self.fiber_name = os.path.basename(folder)
        self.fiber_shape_combo.setCurrentText(self.fiber_shape)
        self.update_fiber_shape_inputs()
        if self.fiber_shape == "rectangular":
            if self.fiber_dimension == "":
                self.fiber_width_input.setText("")
                self.fiber_height_input.setText("")
            else:
                self.fiber_width_input.setText(str(self.fiber_dimension[0]))
                self.fiber_height_input.setText(str(self.fiber_dimension[1]))
        else:
            self.fiber_diameter_input.setText(str(self.fiber_dimension))

        self.fiber_length_input.setText(self.fiber_length)
        self.numerical_aperature_input.setText(self.numerical_aperature)
        self.coating_type_combo.setCurrentText(self.coating_type)
        self.manufacturer_combo.setCurrentText(self.manufacturer)

    def check_folder_exists(self, folder=None):
        if folder is None:
            folder = os.path.join(self.parent().base_directory, self.fiber_name_input.text())
        if os.path.exists(folder) and not self.update_from_load_token:
            self.show_message("Folder already exists, contents may be overwritten.")
            return True
        else:
            self.show_message("")
            return False

    def check_inputs_and_save(self):
        if self.fiber_diameter_input.text() != "":
            self.fiber_dimension = self.fiber_diameter_input.text()
        elif self.fiber_width_input.text() != "" and self.fiber_height_input.text() != "":
            self.fiber_dimension = (self.fiber_width_input.text(), self.fiber_height_input.text())
        else:
            self.show_message("Please enter fiber diameter or height and width.")
            return

        if (self.fiber_name_input.text() != ""
                and self.fiber_shape_combo.currentText() != "None"):
            folder = os.path.join(self.parent().base_directory, self.fiber_name_input.text())

            if self.check_folder_exists(folder): #and not self.update_from_load_token:
                reply = QMessageBox.question(self, 'Folder Exists',
                                             'Folder already exists. Do you want to overwrite the contents?',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return
            self.update_from_load_token = False
            self.fiber_shape = self.fiber_shape_combo.currentText()
            self.fiber_length = self.fiber_length_input.text()
            self.fiber_name = self.fiber_name_input.text()
            self.numerical_aperature = self.numerical_aperature_input.text()
            self.coating_type = self.coating_type_combo.currentText()
            self.manufacturer = self.manufacturer_combo.currentText()

            self.emit_fiber_data_changed()

            self.save_fiber_data(folder)
            self.close()
        else:
            self.show_message("Please enter fiber name and shape.")

    def show_message(self, message):
        self.message_label.setText(message)

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

    @pyqtSlot()
    def emit_fiber_data_changed(self):
        self.fiberDataChanged.emit(
            self.fiber_name_input.text(),
            self.fiber_dimension,
            self.fiber_shape_combo.currentText()
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())