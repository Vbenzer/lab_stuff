import sys
import os
import json
import threading
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit)
from PyQt6.QtCore import pyqtSignal
import sg_pipeline


# Todo: Add feature to view plots in GUI
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

        self.fiber_name_label = QLabel("Fiber Name:")
        self.fiber_name_input = QLineEdit()
        self.fiber_name_input.textChanged.connect(self.update_working_dir)

        self.fiber_diameter_label = QLabel("Fiber Diameter (µm):")
        self.fiber_diameter_input = QLineEdit()

        self.fiber_length_label = QLabel("Fiber Length (m):")
        self.fiber_length_input = QLineEdit()

        self.fiber_shape_label = QLabel("Fiber Shape:")
        self.fiber_shape_combo = QComboBox()
        self.fiber_shape_combo.addItems(["None", "circular", "octagon"])

        self.working_dir_label = QLabel("Working Directory:")
        self.working_dir_display = QLabel("")

        self.choose_folder_button = QPushButton("Choose Existing Folder")
        self.choose_folder_button.clicked.connect(self.choose_folder)

        self.lock_button = QPushButton("Lock In")
        self.lock_button.clicked.connect(self.lock_inputs)

        self.unlock_button = QPushButton("Unlock")
        self.unlock_button.clicked.connect(self.unlock_inputs)

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: red; font-weight: bold;")

        self.layout.addWidget(self.fiber_name_label)
        self.layout.addWidget(self.fiber_name_input)
        self.layout.addWidget(self.fiber_diameter_label)
        self.layout.addWidget(self.fiber_diameter_input)
        self.layout.addWidget(self.fiber_length_label)
        self.layout.addWidget(self.fiber_length_input)
        self.layout.addWidget(self.fiber_shape_label)
        self.layout.addWidget(self.fiber_shape_combo)
        self.layout.addWidget(self.working_dir_label)
        self.layout.addWidget(self.working_dir_display)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_folder_button)
        button_layout.addWidget(self.lock_button)
        button_layout.addWidget(self.unlock_button)
        self.layout.addLayout(button_layout)

        self.layout.addWidget(self.message_label)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.measure_tab = QWidget()
        self.analyse_tab = QWidget()

        self.tabs.addTab(self.measure_tab, "Measure")
        self.tabs.addTab(self.analyse_tab, "Analyse")

        self.init_measure_tab()
        self.init_analyse_tab()

        self.progress_signal.connect(self.update_progress)

        self.progress_label = QLabel("")
        self.layout.addWidget(self.progress_label)

        self.progress_text_edit = QTextEdit()
        self.progress_text_edit.setReadOnly(True)
        self.progress_text_edit.hide()  # Initially hidden

        self.layout.addWidget(self.progress_text_edit)

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
        fiber_name = self.fiber_name_input.text()
        if fiber_name:
            working_dir = os.path.join(self.base_directory, fiber_name)
            self.working_dir_display.setText(working_dir)
        else:
            self.working_dir_display.setText("")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.base_directory)
        if folder:
            self.working_dir_display.setText(folder)
            self.fiber_name_input.setText(os.path.basename(folder))
            self.load_fiber_data(folder)
            if self.inputs_locked:
                self.check_existing_measurements(folder)

        self.lock_inputs()

    def load_fiber_data(self, folder):
        file_path = os.path.join(folder, "fiber_data.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
                fiber_diameter = data.get("fiber_diameter", "")
                fiber_shape = data.get("fiber_shape", "")
                fiber_length = data.get("fiber_length", "")
                if fiber_diameter:
                    self.fiber_diameter_input.setText(str(int(fiber_diameter)))
                else:
                    self.fiber_diameter_input.setText("")
                if fiber_shape:
                    self.fiber_shape_combo.setCurrentText(fiber_shape)
                else:
                    self.fiber_shape_combo.setCurrentText("None")
                if fiber_length:
                    self.fiber_length_input.setText(str(fiber_length))
                else:
                    self.fiber_length_input.setText("")

        else:
            self.fiber_diameter_input.setText("")
            self.fiber_shape_combo.setCurrentText("None")
            self.fiber_length_input.setText("")

    def save_fiber_data(self, folder, fiber_diameter, fiber_shape, fiber_length):
        file_path = os.path.join(folder, "fiber_data.json")
        with open(file_path, "w") as file:
            # noinspection PyTypeChecker
            json.dump({"fiber_diameter": int(fiber_diameter), "fiber_shape": fiber_shape,
                       "fiber_length": fiber_length}, file)

    def show_message(self, message):
        self.message_label.setText(message)

    def lock_inputs(self):
        fiber_name = self.fiber_name_input.text()
        fiber_diameter = self.fiber_diameter_input.text()
        fiber_length = self.fiber_length_input.text()
        fiber_shape = self.fiber_shape_combo.currentText()

        if not fiber_name or not fiber_diameter or not fiber_shape or not fiber_length:
            self.show_message("Please enter fiber name, diameter, length and shape.")
            return

        working_dir = self.working_dir_display.text()
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.fiber_name_input.setDisabled(True)
        self.fiber_diameter_input.setDisabled(True)
        self.fiber_length_input.setDisabled(True)
        self.fiber_shape_combo.setDisabled(True)
        self.inputs_locked = True
        self.show_message("Inputs locked.")
        self.run_measurement_button.setDisabled(True)
        self.run_analysis_button.setDisabled(True)
        self.check_existing_measurements(working_dir)
        self.update_checklist()

        # Save fiber data to JSON
        self.save_fiber_data(working_dir, fiber_diameter, fiber_shape, fiber_length)

        # Update all run buttons
        self.update_measurement_button_state()
        self.update_throughput_analysis_button_state()
        self.update_general_analysis_button_state()

        # Update the UI state
        self.update_ui_state()

    def unlock_inputs(self):
        self.fiber_name_input.setDisabled(False)
        self.fiber_diameter_input.setDisabled(False)
        self.fiber_length_input.setDisabled(False)
        self.fiber_shape_combo.setDisabled(False)
        self.inputs_locked = False
        self.show_message("Inputs unlocked.")
        self.run_measurement_button.setDisabled(True)
        self.run_analysis_button.setDisabled(True)
        self.existing_measurements_label.setText("")

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
        self.check1 = QCheckBox("Check 1")
        self.check2 = QCheckBox("Check 2")
        self.check3 = QCheckBox("Check 3")

        self.check1.stateChanged.connect(self.update_measurement_button_state)
        self.check2.stateChanged.connect(self.update_measurement_button_state)
        self.check3.stateChanged.connect(self.update_measurement_button_state)

        layout.addWidget(self.checklist_label)
        layout.addWidget(self.check1)
        layout.addWidget(self.check2)
        layout.addWidget(self.check3)

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

        self.plot_sg_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.calc_sg_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.plot_coms_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.get_params_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.plot_masks_checkbox.stateChanged.connect(self.update_general_analysis_button_state)
        self.make_video_checkbox.stateChanged.connect(self.update_general_analysis_button_state)

        self.calibration_file_label = QLabel("Calibration File:")
        self.calibration_file_input = QLineEdit()
        self.calibration_file_input.textChanged.connect(self.update_throughput_analysis_button_state)
        self.calibration_file_button = QPushButton("Choose Calibration File")
        self.calibration_file_button.clicked.connect(self.choose_calibration_file)

        layout.addWidget(self.plot_sg_checkbox)
        layout.addWidget(self.calc_sg_checkbox)
        layout.addWidget(self.plot_coms_checkbox)
        layout.addWidget(self.get_params_checkbox)
        layout.addWidget(self.plot_masks_checkbox)
        layout.addWidget(self.make_video_checkbox)
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
            self.check1.setText("Fiber in place")
            self.check2.setText("Spot in center")
            self.check3.setText("Spot in focus")
        elif measurement_type == "FRD":
            self.check1.setText("Fiber in place: Input and Output")
            self.check2.setText("Spot on Fiber")
            self.check3.setText("Camera Enabled and max counts in range")
        elif measurement_type == "Throughput":
            self.check1.setText("Throughput Check 1")
            self.check2.setText("Throughput Check 2")
            self.check3.setText("Throughput Check 3")

        self.check1.setChecked(False)
        self.check2.setChecked(False)
        self.check3.setChecked(False)
        self.update_measurement_button_state()

    def update_measurement_button_state(self):
        if self.inputs_locked and self.check1.isChecked() and self.check2.isChecked() and self.check3.isChecked():
            self.run_measurement_button.setDisabled(False)
        else:
            self.run_measurement_button.setDisabled(True)

    def run_measurement(self):
        if not self.inputs_locked:
            self.show_message("Please lock the inputs before running the measurement.")
            return

        if not (self.check1.isChecked() and self.check2.isChecked() and self.check3.isChecked()):
            self.show_message("Please complete all checklist items before running the measurement.")
            return

        fiber_name = self.fiber_name_input.text()
        fiber_diameter = int(self.fiber_diameter_input.text())
        fiber_shape = self.fiber_shape_combo.currentText()
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
        fiber_diameter = int(self.fiber_diameter_input.text())
        fiber_shape = self.fiber_shape_combo.currentText()
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
            if self.plot_sg_checkbox.isChecked():
                sg_pipeline.plot_sg_cool_like(directory, fiber_diameter, progress_signal=self.progress_signal)

            if self.calc_sg_checkbox.isChecked():
                sg_pipeline.calc_sg(directory, progress_signal=self.progress_signal)

            if self.plot_coms_checkbox.isChecked():
                sg_pipeline.plot_coms(directory, progress_signal=self.progress_signal)

            if self.get_params_checkbox.isChecked():
                sg_pipeline.get_sg_params(directory, fiber_diameter, fiber_shape, progress_signal=self.progress_signal)

            if self.plot_masks_checkbox.isChecked():
                sg_pipeline.plot_masks(directory, fiber_diameter, progress_signal=self.progress_signal)

            if self.make_video_checkbox.isChecked():
                sg_pipeline.make_comparison_video(directory, fiber_diameter)

        elif analysis_type == "FRD":
            directory = os.path.join(working_dir, "FRD")
            import fiber_frd_measurements as frd
            frd.main_analyse_all_filters(directory, progress_signal=self.progress_signal)

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
        self.run_measurement_button.setDisabled(not state or not (self.check1.isChecked() and self.check2.isChecked() and self.check3.isChecked()))
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