import sys
import os
import json
import threading
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox

# Todo: Add feature to view plots in GUI
class MainWindow(QMainWindow):
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

        self.fiber_diameter_label = QLabel("Fiber Diameter (microns):")
        self.fiber_diameter_input = QLineEdit()

        self.fiber_shape_label = QLabel("Fiber Shape:")
        self.fiber_shape_combo = QComboBox()
        self.fiber_shape_combo.addItems(["circular", "octagon"])

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

    def load_fiber_data(self, folder):
        file_path = os.path.join(folder, "fiber_data.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
                fiber_diameter = data.get("fiber_diameter", "")
                fiber_shape = data.get("fiber_shape", "")
                if fiber_diameter:
                    self.fiber_diameter_input.setText(str(int(fiber_diameter)))
                else:
                    self.fiber_diameter_input.setText("")
                if fiber_shape:
                    self.fiber_shape_combo.setCurrentText(fiber_shape)
                else:
                    self.fiber_shape_combo.setCurrentIndex(-1)

    def save_fiber_data(self, folder, fiber_diameter, fiber_shape):
        file_path = os.path.join(folder, "fiber_data.json")
        with open(file_path, "w") as file:
            json.dump({"fiber_diameter": int(fiber_diameter), "fiber_shape": fiber_shape}, file)

    def show_message(self, message):
        self.message_label.setText(message)

    def lock_inputs(self):
        fiber_name = self.fiber_name_input.text()
        fiber_diameter = self.fiber_diameter_input.text()
        fiber_shape = self.fiber_shape_combo.currentText()

        if not fiber_name or not fiber_diameter or not fiber_shape:
            self.show_message("Please enter fiber name, diameter, and shape.")
            return

        working_dir = self.working_dir_display.text()
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.fiber_name_input.setDisabled(True)
        self.fiber_diameter_input.setDisabled(True)
        self.fiber_shape_combo.setDisabled(True)
        self.inputs_locked = True
        self.show_message("Inputs locked.")
        self.run_measurement_button.setDisabled(True)
        self.run_analysis_button.setDisabled(True)
        self.check_existing_measurements(working_dir)
        self.update_checklist()

    def unlock_inputs(self):
        self.fiber_name_input.setDisabled(False)
        self.fiber_diameter_input.setDisabled(False)
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

        self.checklist_label = QLabel("Checklist:")
        self.checklist_label.setStyleSheet("font-weight: bold;")
        self.check1 = QCheckBox("Check 1")
        self.check2 = QCheckBox("Check 2")
        self.check3 = QCheckBox("Check 3")

        self.check1.stateChanged.connect(self.update_run_button_state)
        self.check2.stateChanged.connect(self.update_run_button_state)
        self.check3.stateChanged.connect(self.update_run_button_state)

        self.run_measurement_button = QPushButton("Run Measurement")
        self.run_measurement_button.setDisabled(True)
        self.run_measurement_button.clicked.connect(self.run_measurement)

        self.existing_measurements_label = QLabel("")
        self.existing_measurements_label.setStyleSheet("color: green; font-weight: bold;")

        layout.addWidget(self.measurement_type_label)
        layout.addWidget(self.measurement_type_combo)
        layout.addWidget(self.checklist_label)
        layout.addWidget(self.check1)
        layout.addWidget(self.check2)
        layout.addWidget(self.check3)
        layout.addWidget(self.run_measurement_button)
        layout.addWidget(self.existing_measurements_label)

        self.measure_tab.setLayout(layout)

    def init_analyse_tab(self):
        layout = QVBoxLayout()

        self.analysis_type_label = QLabel("Analysis Type:")
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["SG", "FRD", "Throughput"])

        self.plot_sg_checkbox = QCheckBox("Plot SG")
        self.calc_sg_checkbox = QCheckBox("Calc SG")
        self.plot_coms_checkbox = QCheckBox("Plot COMs")
        self.get_params_checkbox = QCheckBox("Get Parameters")
        self.plot_masks_checkbox = QCheckBox("Plot Masks")

        self.plot_sg_checkbox.stateChanged.connect(self.update_analysis_button_state)
        self.calc_sg_checkbox.stateChanged.connect(self.update_analysis_button_state)
        self.plot_coms_checkbox.stateChanged.connect(self.update_analysis_button_state)
        self.get_params_checkbox.stateChanged.connect(self.update_analysis_button_state)
        self.plot_masks_checkbox.stateChanged.connect(self.update_analysis_button_state)

        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.setDisabled(True)
        self.run_analysis_button.clicked.connect(self.run_analysis)

        layout.addWidget(self.analysis_type_label)
        layout.addWidget(self.analysis_type_combo)
        layout.addWidget(self.plot_sg_checkbox)
        layout.addWidget(self.calc_sg_checkbox)
        layout.addWidget(self.plot_coms_checkbox)
        layout.addWidget(self.get_params_checkbox)
        layout.addWidget(self.plot_masks_checkbox)
        layout.addWidget(self.run_analysis_button)

        self.analyse_tab.setLayout(layout)

    def update_analysis_button_state(self):
        if (self.plot_sg_checkbox.isChecked() or self.calc_sg_checkbox.isChecked() or self.plot_coms_checkbox.isChecked()
                or self.get_params_checkbox.isChecked() or self.plot_masks_checkbox.isChecked()):
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
            self.check1.setText("FRD Check 1")
            self.check2.setText("FRD Check 2")
            self.check3.setText("FRD Check 3")
        elif measurement_type == "Throughput":
            self.check1.setText("Throughput Check 1")
            self.check2.setText("Throughput Check 2")
            self.check3.setText("Throughput Check 3")

        self.check1.setChecked(False)
        self.check2.setChecked(False)
        self.check3.setChecked(False)
        self.update_run_button_state()

    def update_run_button_state(self):
        if self.check1.isChecked() and self.check2.isChecked() and self.check3.isChecked():
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

        self.save_fiber_data(working_dir, fiber_diameter, fiber_shape)

        self.experiment_running = True
        self.update_ui_state()

        threading.Thread(target=self.run_measurement_thread, args=(measurement_type, working_dir, fiber_diameter, fiber_shape)).start()

    def run_measurement_thread(self, measurement_type, working_dir, fiber_diameter, fiber_shape):
        # Call the appropriate measurement function
        if measurement_type == "SG":
            self.measure_sg(working_dir, fiber_diameter, fiber_shape)
        elif measurement_type == "FRD":
            self.measure_frd(working_dir, fiber_diameter, fiber_shape)
        elif measurement_type == "Throughput":
            self.measure_throughput(working_dir, fiber_diameter, fiber_shape)

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

        self.experiment_running = True
        self.update_ui_state()

        threading.Thread(target=self.run_analysis_thread, args=(analysis_type, working_dir, fiber_diameter, fiber_shape)).start()

    def run_analysis_thread(self, analysis_type, working_dir, fiber_diameter, fiber_shape):
        if analysis_type == "SG":
            directory = os.path.join(working_dir, "SG")
            import sg_pipeline
            if self.plot_sg_checkbox.isChecked():
                sg_pipeline.plot_sg_cool_like(directory, fiber_diameter)

            if self.calc_sg_checkbox.isChecked():
                sg_pipeline.calc_sg(directory)

            if self.plot_coms_checkbox.isChecked():
                sg_pipeline.plot_coms(directory)

            if self.get_params_checkbox.isChecked():
                sg_pipeline.get_sg_params(directory, fiber_diameter, fiber_shape=fiber_shape)

            if self.plot_masks_checkbox.isChecked():
                sg_pipeline.plot_masks(directory, fiber_diameter)

        elif analysis_type == "FRD":
            # Add corresponding functions for FRD analysis
            pass
        elif analysis_type == "Throughput":
            # Add corresponding functions for Throughput analysis
            pass

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
        sg_pipeline.capture_images_and_reduce(working_dir, fiber_diameter)  # Todo: Add number of positions as input

    def measure_frd(self, working_dir, fiber_diameter, fiber_shape):
        self.show_message(f"Running FRD measurement with working dir: {working_dir}, fiber diameter: {fiber_diameter}, and fiber shape: {fiber_shape}")

    def measure_throughput(self, working_dir, fiber_diameter, fiber_shape):
        self.show_message(f"Running Throughput measurement with working dir: {working_dir}, fiber diameter: {fiber_diameter}, and fiber shape: {fiber_shape}")

    def analyse_sg(self, working_dir):
        self.show_message(f"Running SG analysis with working dir: {working_dir}")

    def analyse_frd(self, working_dir):
        self.show_message(f"Running FRD analysis with working dir: {working_dir}")

    def analyse_throughput(self, working_dir):
        self.show_message(f"Running Throughput analysis with working dir: {working_dir}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())