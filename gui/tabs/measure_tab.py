from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QCheckBox, QVBoxLayout
                             )
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

import threading

import analysis.frd_analysis
import analysis.sg_analysis
import os

class MeasureTab:
    def __init__(self, main_ctrl, main_init_ctrl):
        self.main = main_ctrl
        self.main_init = main_init_ctrl
        self.main_init.log_data("MeasureTab initialized.")  # Log initialization

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

        self.exposure_time_label_mt_exit = QLabel("Exit Cam Exposure Time:")  # exit cam exp time control
        self.exposure_time_input_mt_exit = QLineEdit()
        self.exposure_time_input_mt_exit.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+(\.\d+)?(ms|s|us)$")))
        self.exposure_time_input_mt_exit.setFixedWidth(100)
        self.exposure_time_input_mt_exit.setText("10ms")

        self.exposure_time_label_mt_entrance = QLabel("Entrance Cam Exposure Time:")  # entrance cam exp time control
        self.exposure_time_input_mt_entrance = QLineEdit()
        self.exposure_time_input_mt_entrance.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+(\.\d+)?(ms|s|us)$")))
        self.exposure_time_input_mt_entrance.setFixedWidth(100)
        self.exposure_time_input_mt_entrance.setText("10ms")
        # Initially hidden
        self.exposure_time_label_mt_exit.hide()
        self.exposure_time_input_mt_exit.hide()
        self.exposure_time_label_mt_entrance.hide()
        self.exposure_time_input_mt_entrance.hide()


        # Create a widget for the measurement type chooser and set its position
        measurement_type_widget = QWidget()
        measurement_type_layout = QHBoxLayout(measurement_type_widget)
        measurement_type_layout.addWidget(self.measurement_type_label)
        measurement_type_layout.addWidget(self.measurement_type_combo)
        measurement_type_layout.addStretch()

        measurement_type_layout.addWidget(self.exposure_time_label_mt)
        measurement_type_layout.addWidget(self.exposure_time_input_mt)

        measurement_type_layout.addWidget(self.exposure_time_label_mt_entrance)
        measurement_type_layout.addWidget(self.exposure_time_input_mt_entrance)
        measurement_type_layout.addWidget(self.exposure_time_label_mt_exit)
        measurement_type_layout.addWidget(self.exposure_time_input_mt_exit)


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

        self.main_init.measure_tab.setLayout(layout)

        # Update the checklist based on the default selected measurement type
        self.update_checklist()

    def run_measurement(self):
        self.main_init.log_data("Run measurement started.")  # Log measurement start
        if self.main.folder_name and self.main.fiber_shape and self.main.fiber_dimension == "":
            self.main_init.log_data("Run measurement failed: Fiber data not entered.")  # Log failure
            self.show_message("Please enter fiber data before running the measurement.")
            return

        if not (self.check1.isChecked() and self.check2.isChecked() and self.check3.isChecked()
                and self.check4.isChecked() and self.check5.isChecked()):
            self.main_init.log_data("Run measurement failed: Checklist not completed.")  # Log failure
            self.show_message("Please complete all checklist items before running the measurement.")
            return

        self.main.experiment_running = True
        self.main_init.update_ui_state()
        self.main_init.log_data("Experiment running state set to True.")  # Log state change

        fiber_name = self.main.folder_name
        fiber_shape = self.main.fiber_shape
        fiber_diameter = (int(self.main.fiber_dimension[0]), int(self.main.fiber_dimension[1])) if fiber_shape == "rectangular" else int(self.main.fiber_dimension)
        measurement_type = self.measurement_type_combo.currentText()
        working_dir = self.main_init.working_dir_display.text()

        threading.Thread(target=self.run_measurement_thread, args=(measurement_type, working_dir, fiber_diameter, fiber_shape)).start()

    def run_measurement_thread(self, measurement_type, working_dir, fiber_diameter, fiber_shape):
        self.main.progress_signal.emit("Starting measurement...")
        self.main_init.log_data(f"Measurement thread started for type: {measurement_type}.")  # Log thread start
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

        self.main.progress_signal.emit("Measurement complete.")
        self.main.experiment_running = False
        self.main_init.update_ui_state()
        self.main_init.log_data(f"Measurement completed for type: {measurement_type}.")  # Log completion

    def measure_sg(self, working_dir, fiber_diameter, fiber_shape):
        self.main_init.log_data(f"SG measurement started with working dir: {working_dir}, fiber diameter: {fiber_diameter}, and fiber shape: {fiber_shape}.")  # Log SG start
        exposure_time_exit = self.exposure_time_input_mt_exit.text()
        exposure_time_entrance = self.exposure_time_input_mt_entrance.text()
        exp_times = {"exit": exposure_time_exit, "entrance": exposure_time_entrance}

        analysis.sg_analysis.capture_images_and_reduce(working_dir, fiber_diameter, progress_signal=self.main.progress_signal, exposure_times=exp_times)

    def measure_frd(self, working_dir, fiber_diameter, fiber_shape):
        self.main_init.log_data(f"FRD measurement started with working dir: {working_dir}.")  # Log FRD start
        from core.hardware.cameras import qhyccd_control as qhy

        self.main_init.show_message(f"Running FRD measurement with working dir: {working_dir}")

        exposure_time = qhy.convert_to_us(self.exposure_time_input_mt.text())
        analysis.frd_analysis.main_measure_frd(working_dir, progress_signal=self.main.progress_signal, exp_time=exposure_time)

    def measure_throughput(self, working_dir, fiber_diameter, fiber_shape):
        self.main_init.log_data(f"Throughput measurement started with working dir: {working_dir}.")  # Log throughput start
        self.show_message(f"Running Throughput measurement with working dir: {working_dir}")
        from analysis import throughput_analysis
        throughput_analysis.measure_all_filters(working_dir, progress_signal=self.main.progress_signal, base_directory=self.main.base_directory)

    def update_measurement_button_state(self):
        self.main_init.log_data("Measurement button state updated.")  # Log button state update
        if (self.main.folder_name and self.main.fiber_dimension and self.main.fiber_shape != ""
                and self.check1.isChecked()
                and self.check2.isChecked() and self.check3.isChecked()
                and self.check4.isChecked() and self.check5.isChecked()
        ):
            self.run_measurement_button.setDisabled(False)
        else:
            self.run_measurement_button.setDisabled(True)

    def update_checklist(self):
        measurement_type = self.measurement_type_combo.currentText()
        self.main_init.log_data(f"Checklist updated for measurement type: {measurement_type}.")  # Log checklist update

        self.check1.setChecked(False)
        self.check2.setChecked(False)
        self.check3.setChecked(False)
        self.check4.setChecked(False)
        self.check5.setChecked(False)

        self.check4.show()
        self.check5.show()

        if measurement_type == "SG":
            self.check1.setText("Fiber in place: Output at small camera")
            self.check2.setText("Motor controller plugged in and referenced")
            self.check3.setText("Input spot in center, in focus and oriented horizontally. Exit camera fiber also in focus")
            self.check4.setText("Exposure times set. ThorCam software closed")
            self.check5.setText("Lights Out")

            self.exposure_time_label_mt.hide()
            self.exposure_time_input_mt.hide()

            self.exposure_time_label_mt_exit.show()
            self.exposure_time_input_mt_exit.show()
            self.exposure_time_label_mt_entrance.show()
            self.exposure_time_input_mt_entrance.show()

        elif measurement_type == "FRD":
            self.check1.setText("Fiber in place: Output at large camera")
            self.check2.setText("Spot on Fiber")
            self.check3.setText("Camera enabled and max counts in range")
            self.check4.setText("ThorCam/N.I.N.A closed")
            self.check5.setText("Lights Out")

            self.exposure_time_label_mt.show()
            self.exposure_time_input_mt.show()

            self.exposure_time_label_mt_exit.hide()
            self.exposure_time_input_mt_exit.hide()
            self.exposure_time_label_mt_entrance.hide()
            self.exposure_time_input_mt_entrance.hide()

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

            self.exposure_time_label_mt_exit.hide()
            self.exposure_time_input_mt_exit.hide()
            self.exposure_time_label_mt_entrance.hide()
            self.exposure_time_input_mt_entrance.hide()

        self.update_measurement_button_state()
