import sys
import os
import json
import threading
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import pyqtSignal
import sg_pipeline


def save_recent_folders(recent_folders, file_path='D:/Vincent/recent_folders.json'):
    with open(file_path, 'w') as file:
        json.dump(recent_folders, file)

def load_recent_folders(file_path='D:/Vincent/recent_folders.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def update_recent_folders(folder, recent_folders, max_recent=2):
    if folder in recent_folders:
        recent_folders.remove(folder)
    recent_folders.insert(0, folder)
    if len(recent_folders) > max_recent:
        recent_folders.pop()
    save_recent_folders(recent_folders)

# Todo: Add feature to view plots in GUI
# Todo: This would be cool: For more complex fiber shapes add a custom feature where the user can trace the fiber shape around the fiber image
# the mask can then be scaled and used for the calculations

class MainWindow(QMainWindow):
    progress_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.base_directory = "D:/Vincent"
        self.inputs_locked = False
        self.experiment_running = False

        self.setWindowTitle("Fiber Measurement and Analysis")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.init_ui()

    def init_ui(self):
        self.folder_name_label = QLabel("Fiber Name:")
        self.folder_name_input = QLineEdit()
        self.folder_name_input.setFixedWidth(700)
        self.folder_name_input.textChanged.connect(self.update_working_dir)

        self.fiber_diameter_label = QLabel("Fiber Diameter (µm):")
        self.fiber_diameter_input = QLineEdit()
        self.fiber_diameter_input.setFixedWidth(700)

        self.fiber_width_label = QLabel("Fiber Width (µm):")
        self.fiber_width_input = QLineEdit()
        self.fiber_width_input.setFixedWidth(290)
        self.fiber_width_label.hide()
        self.fiber_width_input.hide()

        self.fiber_height_label = QLabel("Fiber Height (µm):")
        self.fiber_height_input = QLineEdit()
        self.fiber_height_input.setFixedWidth(290)
        self.fiber_height_label.hide()
        self.fiber_height_input.hide()

        self.fiber_length_label = QLabel("Fiber Length (m):")
        self.fiber_length_input = QLineEdit()
        self.fiber_length_input.setFixedWidth(700)

        self.fiber_shape_label = QLabel("Fiber Shape:")
        self.fiber_shape_combo = QComboBox()
        self.fiber_shape_combo.addItems(["None", "circular", "octagonal", "rectangular"])
        self.fiber_shape_combo.setFixedWidth(700)
        self.fiber_shape_combo.currentIndexChanged.connect(self.update_fiber_shape_inputs)

        self.working_dir_label = QLabel("Working Directory:")
        self.working_dir_display = QLabel("")

        self.choose_folder_button = QPushButton("Choose Existing Folder")
        self.choose_folder_button.clicked.connect(self.choose_folder)

        self.metadata_button = QPushButton("Add Metadata")
        self.metadata_button.clicked.connect(self.access_metadata)
        self.metadata_button.setDisabled(True)

        self.lock_button = QPushButton("Lock In")
        self.lock_button.clicked.connect(self.lock_inputs)

        self.unlock_button = QPushButton("Unlock")
        self.unlock_button.clicked.connect(self.unlock_inputs)
        self.unlock_button.setDisabled(True)

        self.recent_folders = load_recent_folders()
        self.recent_folders_combo = QComboBox()
        self.update_recent_folders_combo()
        self.recent_folders_combo.currentIndexChanged.connect(self.select_recent_folder)

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: red; font-weight: bold;")

        self.progress_label = QLabel("")
        self.progress_text_edit = QTextEdit()
        self.progress_text_edit.setReadOnly(True)
        self.progress_text_edit.hide()  # Initially hidden

        self.layout.addLayout(self.create_hbox_layout(self.folder_name_label, self.folder_name_input))
        self.layout.addLayout(self.create_hbox_layout(self.fiber_diameter_label, self.fiber_diameter_input))

        self.width_height_layout = QHBoxLayout()
        self.width_height_layout.addWidget(self.fiber_width_label)
        self.width_height_layout.addWidget(self.fiber_width_input)
        self.width_height_layout.addWidget(self.fiber_height_label)
        self.width_height_layout.addWidget(self.fiber_height_input)
        self.layout.addLayout(self.width_height_layout)

        self.layout.addLayout(self.create_hbox_layout(self.fiber_length_label, self.fiber_length_input))
        self.layout.addLayout(self.create_hbox_layout(self.fiber_shape_label, self.fiber_shape_combo))
        self.layout.addLayout(self.create_hbox_layout(self.working_dir_label, self.working_dir_display))

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_folder_button)
        button_layout.addWidget(self.metadata_button)
        button_layout.addWidget(self.lock_button)
        button_layout.addWidget(self.unlock_button)
        self.layout.addLayout(button_layout)

        self.layout.addWidget(QLabel("Recent Folders:"))
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

        self.tabs.addTab(self.measure_tab, "Measure")
        self.tabs.addTab(self.analyse_tab, "Analyse")
        self.tabs.addTab(self.general_tab, "General")

        self.init_measure_tab()
        self.init_analyse_tab()
        self.init_general_tab()

        self.progress_signal.connect(self.update_progress)

    def update_input_visibility(self):
        if self.tabs.currentWidget() == self.general_tab:
            self.folder_name_label.setText("Folder Name:")
            self.fiber_diameter_label.hide()
            self.fiber_diameter_input.hide()
            self.fiber_width_label.hide()
            self.fiber_width_input.hide()
            self.fiber_height_label.hide()
            self.fiber_height_input.hide()
            self.fiber_length_label.hide()
            self.fiber_length_input.hide()
            self.fiber_shape_label.hide()
            self.fiber_shape_combo.hide()
            if not hasattr(self, 'placeholder_spacer'):
                self.placeholder_spacer = QSpacerItem(20, 86)
                self.layout.insertItem(self.layout.count() - 1, self.placeholder_spacer)
        else:
            if hasattr(self, 'placeholder_spacer'):
                self.layout.removeItem(self.placeholder_spacer)
                del self.placeholder_spacer
                self.layout.update()

            self.folder_name_label.setText("Fiber Name:")
            self.fiber_diameter_label.show()
            self.fiber_diameter_input.show()
            self.fiber_length_label.show()
            self.fiber_length_input.show()
            self.fiber_shape_label.show()
            self.fiber_shape_combo.show()
            self.update_fiber_shape_inputs()

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
            self.load_fiber_data(folder)
            self.update_metadata_button()

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

    def access_metadata(self):
        working_dir = self.working_dir_display.text()
        if not working_dir:
            self.show_message("Please select a working directory first.")
            return

        metadata_file_path = os.path.join(working_dir, "metadata.txt")
        if not os.path.exists(metadata_file_path):
            with open(metadata_file_path, "w") as file:
                file.write("Metadata for the fiber measurements.\n")
            self.metadata_button.setText("Access Metadata")
        else:
            os.startfile(metadata_file_path)

    def update_metadata_button(self):
        working_dir = self.working_dir_display.text()
        metadata_file_path = os.path.join(working_dir, "metadata.txt")
        if os.path.exists(metadata_file_path) and os.path.getsize(metadata_file_path) > 0:
            self.metadata_button.setText("Access Metadata")
        else:
            self.metadata_button.setText("Add Metadata")

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
            self.load_fiber_data(folder)
            if self.inputs_locked:
                self.check_existing_measurements(folder)

            # Check if all required inputs are filled before locking
            if self.folder_name_input.text() and (self.fiber_diameter_input.text() or (self.fiber_length_input.text() and self.fiber_width_input.text())) and self.fiber_length_input.text() and self.fiber_shape_combo.currentText() != "None":
                self.lock_inputs()
            else:
                self.show_message("Please enter fiber name, diameter, length, and shape before locking inputs.")
            self.update_metadata_button()
            update_recent_folders(folder, self.recent_folders)
            self.update_recent_folders_combo()

    def load_fiber_data(self, folder):
        file_path = os.path.join(folder, "fiber_data.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
                fiber_diameter = data.get("fiber_diameter", "")
                fiber_shape = data.get("fiber_shape", "")
                fiber_length = data.get("fiber_length", "")

                if fiber_shape:
                    self.fiber_shape_combo.setCurrentText(fiber_shape)
                else:
                    self.fiber_shape_combo.setCurrentText("None")
                if fiber_shape == "rectangular" and isinstance(fiber_diameter, list) and len(fiber_diameter) == 2:
                    self.fiber_width_input.setText(str(fiber_diameter[0]))
                    self.fiber_height_input.setText(str(fiber_diameter[1]))
                    self.fiber_diameter_input.setText("")
                else:
                    self.fiber_diameter_input.setText(str(fiber_diameter))
                    self.fiber_width_input.setText("")
                    self.fiber_height_input.setText("")
                if fiber_length:
                    self.fiber_length_input.setText(str(fiber_length))
                else:
                    self.fiber_length_input.setText("")
        else:
            self.fiber_diameter_input.setText("")
            self.fiber_shape_combo.setCurrentText("None")
            self.fiber_length_input.setText("")
            self.fiber_width_input.setText("")
            self.fiber_height_input.setText("")

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
            fiber_name = self.folder_name_input.text()
            fiber_shape = self.fiber_shape_combo.currentText()
            fiber_length = self.fiber_length_input.text()
            width = self.fiber_width_input.text()
            height = self.fiber_height_input.text()
            diameter = self.fiber_diameter_input.text()

            if fiber_shape == "None":
                self.show_message("Please enter fiber name, diameter, length and shape.")
                return

            if ((width == "" or height == "") and diameter == ""
                    or any(value == "" for value in [fiber_name, fiber_length])):
                self.show_message("Please enter fiber name, diameter, length and shape.")
                return

            if fiber_shape == "rectangular":
                fiber_diameter = (int(self.fiber_width_input.text()), int(self.fiber_height_input.text()))
            else:
                fiber_diameter = int(self.fiber_diameter_input.text())

            working_dir = self.working_dir_display.text()
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)

        self.folder_name_input.setDisabled(True)
        self.fiber_diameter_input.setDisabled(True)
        self.fiber_length_input.setDisabled(True)
        self.fiber_shape_combo.setDisabled(True)
        self.fiber_width_input.setDisabled(True)
        self.fiber_height_input.setDisabled(True)
        self.inputs_locked = True
        self.show_message("Inputs locked.")
        self.run_measurement_button.setDisabled(True)
        self.run_analysis_button.setDisabled(True)
        self.check_existing_measurements(working_dir)
        self.update_checklist()

        # Save fiber data to JSON
        if self.tabs.currentWidget() != self.general_tab:
            self.save_fiber_data(working_dir, fiber_diameter, fiber_shape, fiber_length)

        # Update all run buttons
        self.update_measurement_button_state()
        self.update_throughput_analysis_button_state()
        self.update_general_analysis_button_state()

        self.metadata_button.setDisabled(False)
        self.update_metadata_button()

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
        self.metadata_button.setDisabled(True)
        self.lock_button.setDisabled(False)
        self.unlock_button.setDisabled(True)
        self.update_run_button_state()

    def update_run_button_state(self):
        self.run_button.setDisabled(not self.inputs_locked)

    def init_general_tab(self):
        layout = QVBoxLayout()

        self.general_function_label = QLabel("Select Function:")
        self.general_function_combo = QComboBox()
        self.general_function_combo.addItems(["Measure System F-ratio", "Make Throughput Calibration", "Adjust Tip/Tilt", "Motor Controller: Reference", "Measure Eccentricity"])

        # Create a widget for the function chooser and set its position
        function_widget = QWidget()
        function_layout = QHBoxLayout(function_widget)
        function_layout.addWidget(self.general_function_label)
        function_layout.addWidget(self.general_function_combo)
        function_layout.addStretch()

        layout.addWidget(function_widget)

        # Add a spacer item to push the button to the bottom
        layout.addStretch()

        # Add the Run button to the General tab
        self.run_button = QPushButton("Run")
        self.run_button.setDisabled(True)  # Initially disabled
        self.run_button.clicked.connect(self.run_general_function)
        layout.addWidget(self.run_button)

        self.general_tab.setLayout(layout)

    def run_general_function(self):
        if not self.inputs_locked:
            self.show_message("Please lock the inputs before running the function.")
            return

        working_dir = self.working_dir_display.text()
        if not working_dir:
            self.show_message("Please select a working directory first.")
            return

        selected_function = self.general_function_combo.currentText()
        self.experiment_running = True
        self.update_ui_state()

        threading.Thread(target=self.run_general_function_thread, args=(selected_function, working_dir)).start()

    def run_general_function_thread(self, selected_function, working_dir):
        self.progress_signal.emit(f"Running {selected_function} with working dir: {working_dir}")
        if selected_function == "Measure System F-ratio":
            import fiber_frd_measurements as frd
            frd.main_measure_all_filters(working_dir, progress_signal=self.progress_signal)
            frd.main_analyse_all_filters(working_dir, progress_signal=self.progress_signal)
        elif selected_function == "Make Throughput Calibration":
            import throughput_analysis as ta
            calibration_file_name = os.path.basename(working_dir)
            ta.measure_all_filters(working_dir, progress_signal=self.progress_signal, calibration=calibration_file_name)
        elif selected_function == "Adjust Tip/Tilt":
            import qhyccd_cam_control
            qhyccd_cam_control.use_camera("tiptilt")
        elif selected_function == "Motor Controller: Reference":
            import step_motor_control as smc
            smc.make_reference_move()
        elif selected_function == "Measure Eccentricity":
            import qhyccd_cam_control
            qhyccd_cam_control.use_camera("eccentricity")

        self.progress_signal.emit(f"{selected_function} complete.")
        self.experiment_running = False
        self.update_ui_state()

    def init_measure_tab(self):
        layout = QVBoxLayout()

        self.measurement_type_label = QLabel("Measurement Type:")
        self.measurement_type_combo = QComboBox()
        self.measurement_type_combo.addItems(["SG", "FRD", "Throughput"])
        self.measurement_type_combo.currentIndexChanged.connect(self.update_checklist)

        # Create a widget for the measurement type chooser and set its position
        measurement_type_widget = QWidget()
        measurement_type_layout = QHBoxLayout(measurement_type_widget)
        measurement_type_layout.addWidget(self.measurement_type_label)
        measurement_type_layout.addWidget(self.measurement_type_combo)
        measurement_type_layout.addStretch()

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

        # Create a widget for the analysis type chooser and set its position
        analysis_type_widget = QWidget()
        analysis_type_layout = QHBoxLayout(analysis_type_widget)
        analysis_type_layout.addWidget(self.analysis_type_label)
        analysis_type_layout.addWidget(self.analysis_type_combo)
        analysis_type_layout.addStretch()

        layout.addWidget(analysis_type_widget)

        self.plot_sg_checkbox = QCheckBox("Plot SG")
        self.calc_sg_checkbox = QCheckBox("Calc SG")
        self.plot_coms_checkbox = QCheckBox("Plot COMs")
        self.get_params_checkbox = QCheckBox("Get Parameters")
        self.plot_masks_checkbox = QCheckBox("Plot Masks")
        self.make_video_checkbox = QCheckBox("Make Video")
        self.sg_new_checkbox = QCheckBox("SG New")
        self.calc_frd_checkbox = QCheckBox("Calculate FRD")
        self.plot_sutherland_checkbox = QCheckBox("Make Sutherland Plot")

        self.plot_sg_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.calc_sg_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.plot_coms_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.get_params_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.plot_masks_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.make_video_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.sg_new_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.calc_frd_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.plot_sutherland_checkbox.stateChanged.connect(self.update_general_analysis_button_state)

        self.calibration_file_label = QLabel("Calibration File:")
        self.calibration_file_input = QLineEdit()
        self.calibration_file_input.textChanged.connect(self.update_throughput_analysis_button_state)
        self.calibration_file_button = QPushButton("Choose Calibration File")
        self.calibration_file_button.clicked.connect(self.choose_calibration_file)

        layout.addWidget(self.get_params_checkbox)
        layout.addWidget(self.plot_sg_checkbox)
        layout.addWidget(self.calc_sg_checkbox)
        layout.addWidget(self.plot_coms_checkbox)
        layout.addWidget(self.plot_masks_checkbox)
        layout.addWidget(self.make_video_checkbox)
        layout.addWidget(self.sg_new_checkbox)
        layout.addWidget(self.calc_frd_checkbox)
        layout.addWidget(self.plot_sutherland_checkbox)

        layout.addWidget(self.calibration_file_label)
        layout.addWidget(self.calibration_file_input)
        layout.addWidget(self.calibration_file_button)

        # Add a spacer item to push the run_analysis_button to the bottom
        layout.addStretch()

        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.setDisabled(True)
        self.run_analysis_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_analysis_button)

        self.analyse_tab.setLayout(layout)
        self.update_analysis_tab()

    def update_throughput_analysis_button_state(self):
        if self.inputs_locked and self.analysis_type_combo.currentText() == "Throughput" and self.calibration_file_input.text():
            self.run_analysis_button.setDisabled(False)
        else:
            self.update_general_analysis_button_state()

    def update_analysis_tab(self):
        analysis_type = self.analysis_type_combo.currentText()
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
            self.calibration_file_label.hide()
            self.calibration_file_input.hide()
            self.calibration_file_button.hide()
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
            self.calibration_file_label.hide()
            self.calibration_file_input.hide()
            self.calibration_file_button.hide()
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
            self.calibration_file_label.show()
            self.calibration_file_input.show()
            self.calibration_file_button.show()

    def choose_calibration_file(self):
        file_path = \
        QFileDialog.getOpenFileName(self, "Select Calibration File", self.base_directory + "/Calibration", "JSON Files (*.json)")[0]
        if file_path:
            self.calibration_file_input.setText(file_path)

    def update_general_analysis_button_state(self):
        if self.inputs_locked and (self.plot_sg_checkbox.isChecked() or self.calc_sg_checkbox.isChecked()
                                    or self.plot_coms_checkbox.isChecked() or self.get_params_checkbox.isChecked()
                                    or self.plot_masks_checkbox.isChecked() or self.make_video_checkbox.isChecked()
                                    or self.sg_new_checkbox.isChecked() or self.calc_frd_checkbox.isChecked()
                                    or self.plot_sutherland_checkbox.isChecked()
        ):
            self.run_analysis_button.setDisabled(False)
        else:
            self.run_analysis_button.setDisabled(True)

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
        if measurement_type == "SG":
            self.check1.setText("Fiber in place: Output at small camera")
            self.check2.setText("Input spot in center and in focus. Exit camera fiber also in focus")
            self.check3.setText("ThorCam software closed")
            self.check4.setText("Motor controller plugged in")
            self.check5.setText("Lights Out")

        elif measurement_type == "FRD":
            self.check1.setText("Fiber in place: Output at large camera")
            self.check2.setText("Spot on Fiber")
            self.check3.setText("Camera enabled and max counts in range")
            self.check4.setText("ThorCam/N.I.N.A closed")
            self.check5.setText("Lights Out")

        elif measurement_type == "Throughput":
            self.check1.setText("Throughput Check 1")
            self.check2.setText("Throughput Check 2")
            self.check3.setText("Throughput Check 3")
            self.check4.hide()
            self.check5.hide()

        self.check1.setChecked(False)
        self.check2.setChecked(False)
        self.check3.setChecked(False)
        self.check4.setChecked(False)
        self.check5.setChecked(False)
        self.update_measurement_button_state()

    def update_measurement_button_state(self):
        if (self.inputs_locked and self.check1.isChecked() and self.check2.isChecked() and self.check3.isChecked()
                and self.check4.isChecked() and self.check5.isChecked()
        ):
            self.run_measurement_button.setDisabled(False)
        else:
            self.run_measurement_button.setDisabled(True)

    def run_measurement(self):
        if not self.inputs_locked:
            self.show_message("Please lock the inputs before running the measurement.")
            return

        if not (self.check1.isChecked() and self.check2.isChecked() and self.check3.isChecked()
                and self.check4.isChecked() and self.check5.isChecked()
        ):
            self.show_message("Please complete all checklist items before running the measurement.")
            return

        fiber_name = self.folder_name_input.text()
        fiber_shape = self.fiber_shape_combo.currentText()

        if fiber_shape == "rectangular":
            fiber_diameter = (int(self.fiber_width_input.text()), int(self.fiber_height_input.text()))
        else:
            fiber_diameter = int(self.fiber_diameter_input.text())

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
        if not self.inputs_locked:
            self.show_message("Please lock the inputs before running the analysis.")
            return

        analysis_type = self.analysis_type_combo.currentText()
        working_dir = self.working_dir_display.text()
        fiber_shape = self.fiber_shape_combo.currentText()

        if fiber_shape == "rectangular":
            fiber_diameter = (int(self.fiber_width_input.text()), int(self.fiber_height_input.text()))
        else:
            fiber_diameter = int(self.fiber_diameter_input.text())

        calibration_file = self.calibration_file_input.text() if analysis_type == "Throughput" else None

        self.experiment_running = True
        self.update_ui_state()

        threading.Thread(target=self.run_analysis_thread,
                         args=(analysis_type, working_dir, fiber_diameter, fiber_shape, calibration_file)).start()

    def run_analysis_thread(self, analysis_type, working_dir, fiber_diameter, fiber_shape, calibration_file):
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

        elif analysis_type == "Throughput":
            directory = os.path.join(working_dir, "Throughput")
            import throughput_analysis
            throughput_analysis.main(directory, calibration_file)

        self.progress_signal.emit("Analysis complete.")
        self.experiment_running = False
        self.update_ui_state()

    def update_ui_state(self):
        state = not self.experiment_running
        self.choose_folder_button.setDisabled(not state)
        self.lock_button.setDisabled(not state)
        self.unlock_button.setDisabled(not state)
        self.run_measurement_button.setDisabled(not state or not (self.check1.isChecked() and self.check2.isChecked()
                                                and self.check3.isChecked() and self.check4.isChecked()
                                                and self.check5.isChecked()
        ))
        self.run_analysis_button.setDisabled(not state)

    def measure_sg(self, working_dir, fiber_diameter, fiber_shape):
        self.show_message(f"Running SG measurement with working dir: {working_dir}, fiber diameter: {fiber_diameter}, and fiber shape: {fiber_shape}")
        import sg_pipeline
        sg_pipeline.capture_images_and_reduce(working_dir, fiber_diameter, progress_signal=self.progress_signal)  # Todo: Add number of positions as input

    def measure_frd(self, working_dir, fiber_diameter, fiber_shape):
        import fiber_frd_measurements
        self.show_message(f"Running FRD measurement with working dir: {working_dir}, fiber diameter: {fiber_diameter}, and fiber shape: {fiber_shape}")
        fiber_frd_measurements.main_measure_all_filters(working_dir, progress_signal=self.progress_signal)

    def measure_throughput(self, working_dir, fiber_diameter, fiber_shape):
        self.show_message(f"Running Throughput measurement with working dir: {working_dir}, fiber diameter: {fiber_diameter}, and fiber shape: {fiber_shape}")
        import throughput_analysis
        throughput_analysis.measure_all_filters(working_dir, progress_signal=self.progress_signal)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())