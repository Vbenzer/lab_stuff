"""Module general_tab.py.

Auto-generated docstring for better readability.
"""
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QVBoxLayout
                             )
from PyQt6.QtCore import QRegularExpression, Qt
from PyQt6.QtGui import QRegularExpressionValidator

import analysis.frd_analysis

from gui.tabs.helpers import HelperFunctions

import threading
import os

class GeneralTab(HelperFunctions):
    def __init__(self, main, main_init):
        self.main = main
        self.main_init = main_init
        self.main_init.log_data("GeneralTab initialized.")  # Log initialization
        layout = QVBoxLayout()

        self.general_function_label = QLabel("Select Function:")
        self.general_function_combo = QComboBox()
        self.general_function_combo.addItems(
            ["Measure System F-ratio", "Make Throughput Calibration", "Adjust Tip/Tilt",
             "Motor Controller: Reference", "Motor Controller: Move to Position",
             "Measure Eccentricity", "FF with each Filter", "Change Color Filter", "Change System F-ratio",
             "Measure Fiber Size", "Near-Field, Far-Field Comparison"])

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

        self.exposure_time_label_gt_2 = QLabel("Exposure Time (Qhyccd):")  # gt = General Tab
        self.exposure_time_input_gt_2 = QLineEdit()
        self.exposure_time_input_gt_2.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+(\.\d+)?(ms|s|us)$")))
        self.exposure_time_input_gt_2.setFixedWidth(100)
        self.exposure_time_input_gt_2.setText("70ms")
        layout.addLayout(self.create_hbox_layout(self.exposure_time_label_gt_2, self.exposure_time_input_gt_2))
        # Initially hidden
        self.exposure_time_label_gt_2.hide()
        self.exposure_time_input_gt_2.hide()

        # F-ratio input for the system
        self.fratio_input_label = QLabel("F-ratio:")
        self.fratio_input_combo = QComboBox()
        self.fratio_input_combo.addItems(["2.5", "3.5", "4.0", "4.5", "5.0", "6.0"])
        layout.addLayout(self.create_hbox_layout(self.fratio_input_label, self.fratio_input_combo))
        # Initially hidden
        self.fratio_input_label.hide()
        self.fratio_input_combo.hide()

        # Input field for choosing the driving width
        self.driving_width_label = QLabel("Driving Width in µm:")
        self.driving_width_input = QLineEdit()
        self.driving_width_input.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+(\.\d+)?$")))
        self.driving_width_input.setFixedWidth(100)
        self.driving_width_input.setText("0")
        layout.addLayout(self.create_hbox_layout(self.driving_width_label, self.driving_width_input))
        # Initially hidden
        self.driving_width_label.hide()
        self.driving_width_input.hide()

        # Input field for number of positions
        self.number_pos_input_label = QLabel("Number of Positions:")
        self.number_pos_input = QLineEdit()
        self.number_pos_input.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+$")))
        self.number_pos_input.setFixedWidth(100)
        self.number_pos_input.setText("11")
        layout.addLayout(self.create_hbox_layout(self.number_pos_input_label, self.number_pos_input))
        # Initially hidden
        self.number_pos_input_label.hide()
        self.number_pos_input.hide()

        # Scale chooser
        self.scale_label = QLabel("Scale:")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Logarithmic"])
        layout.addLayout(self.create_hbox_layout(self.scale_label, self.scale_combo))
        # Initially hidden
        self.scale_label.hide()
        self.scale_combo.hide()

        # Motor drive buttons as three separate buttons
        self.motor_drive_label = QLabel("Drive to Position:")
        self.motor_drive_button_0 = QPushButton("0")
        self.motor_drive_button_0.setFixedWidth(30)
        self.motor_drive_button_5 = QPushButton("5")
        self.motor_drive_button_5.setFixedWidth(30)
        self.motor_drive_button_9_9 = QPushButton("9.9")
        self.motor_drive_button_9_9.setFixedWidth(30)

        motor_drive_buttons_layout = QHBoxLayout()
        motor_drive_buttons_layout.addWidget(self.motor_drive_label)
        motor_drive_buttons_layout.addWidget(self.motor_drive_button_0)
        motor_drive_buttons_layout.addWidget(self.motor_drive_button_5)
        motor_drive_buttons_layout.addWidget(self.motor_drive_button_9_9)
        layout.addLayout(motor_drive_buttons_layout)

        # Connect buttons to handler
        self.motor_drive_button_0.clicked.connect(lambda: self.handle_motor_drive_button(0))
        self.motor_drive_button_5.clicked.connect(lambda: self.handle_motor_drive_button(5))
        self.motor_drive_button_9_9.clicked.connect(lambda: self.handle_motor_drive_button(9.9))

        # Initially hidden
        self.motor_drive_label.hide()
        self.motor_drive_button_0.hide()
        self.motor_drive_button_5.hide()
        self.motor_drive_button_9_9.hide()

        # Add a spacer item to push the button to the bottom
        layout.addStretch()

        self.stop_button = QPushButton("Stop")
        self.stop_button.setDisabled(True)
        self.stop_button.clicked.connect(self.main_init.stop_general_function)
        layout.addWidget(self.stop_button)

        # Add the Run button to the General tab
        self.run_button_gt = QPushButton("Run")
        self.run_button_gt.setDisabled(True)  # Initially disabled
        self.run_button_gt.clicked.connect(self.run_general_function)
        layout.addWidget(self.run_button_gt)

        self.main_init.general_tab.setLayout(layout)

        # Connect the signal and update the stop button visibility
        self.general_function_combo.currentIndexChanged.connect(self.update_general_tab_buttons)
        #self.main_init.folder_name_input.textChanged.connect(self.update_general_tab_buttons)
        self.update_general_tab_buttons()

    def handle_motor_drive_combo(self):
        def run_motor_move():
            position = float(self.motor_drive_combo.currentText())
            from core.hardware import motor_control as smc
            smc.open_connection()
            smc.move_motor_to_position(position, progress_signal=self.main.progress_signal)
            smc.close_connection()

        threading.Thread(target=run_motor_move).start()

    def run_general_function(self):
        selected_function = self.general_function_combo.currentText()
        self.main_init.log_data(f"Run general function started: {selected_function}.")  # Log function start

        if (selected_function in ["Measure Eccentricity", "Adjust Tip/Tilt", "FF with each Filter", "Measure Fiber Size"]
                and self.main.folder_name != ""):
            self.main_init.log_data(f"Run general function failed: Folder name not entered for {selected_function}.")  # Log failure
            self.show_message("Please enter folder name before running the function.")
            return

        self.main.experiment_running = True
        self.main_init.update_ui_state()
        self.main_init.log_data("Experiment running state set to True.")  # Log state change

        working_dir = self.main_init.working_dir_display.text()
        threading.Thread(target=self.run_general_function_thread, args=(selected_function, working_dir)).start()

    def run_general_function_thread(self, selected_function, working_dir):
        import analysis.general_analysis
        self.main.progress_signal.emit(f"Running {selected_function}...")
        self.main_init.log_data(f"General function thread started: {selected_function}.")  # Log thread start
        self.stop_event = threading.Event()
        if selected_function == "Measure System F-ratio":
            """import fiber_frd_measurements as frd
            frd.main_measure_all_filters(working_dir, progress_signal=self.progress_signal,
                                         base_directory=self.base_directory)"""
            from core.hardware.cameras import qhyccd_control as qhy
            exposure_time = qhy.convert_to_us(self.exposure_time_input_gt.text())
            analysis.frd_analysis.main_measure_frd(working_dir, progress_signal=self.main.progress_signal, exp_time=exposure_time)

            analysis.frd_analysis.main_analyse_all_filters(working_dir, progress_signal=self.main.progress_signal)

        elif selected_function == "Make Throughput Calibration":
            from analysis import throughput_analysis as ta
            calibration_folder_name = os.path.basename(working_dir)
            ta.measure_all_filters(working_dir, progress_signal=self.main.progress_signal, calibration=calibration_folder_name,
                                   base_directory=self.main.base_directory)
        elif selected_function == "Adjust Tip/Tilt":
            qhyccd_cam_control.use_camera("tiptilt", self.stop_event)
        elif selected_function == "Motor Controller: Reference":
            from core.hardware import motor_control as smc
            smc.open_connection()
            smc.make_reference_move(progress_signal=self.main.progress_signal)
            smc.close_connection()
        elif selected_function == "Motor Controller: Move to Position":
            from core.hardware import motor_control as smc
            smc.open_connection()
            position = float(self.number_input.text())
            smc.move_motor_to_position(position, progress_signal=self.main.progress_signal)
            smc.close_connection()
        elif selected_function == "Measure Eccentricity":
            qhyccd_cam_control.use_camera("eccentricity", self.stop_event)
        elif selected_function == "FF with each Filter":
            analysis.general_analysis.get_ff_with_all_filters(working_dir, progress_signal=self.main.progress_signal)
        elif selected_function == "Change Color Filter":
            from core.hardware import filter_wheel_color as fwc
            filter_name = self.filter_input_combo.currentText()
            fwc.move(filter_name, progress_signal=self.main.progress_signal)
        elif selected_function == "Change System F-ratio":
            # Check if the filter wheel is available
            if self.main_init.filter_wheel:
                f_ratio = self.fratio_input_combo.currentText()
                self.main_init.filter_wheel.move_to_filter(f_ratio, progress_signal=self.main.progress_signal)
        elif selected_function == "Measure Fiber Size":
            exposure_times = {
                "exit_cam": self.exposure_time_input_gt.text()
            }
            analysis.general_analysis.measure_fiber_size(working_dir, exposure_times=exposure_times,
                                                         progress_signal=self.main.progress_signal)
        elif selected_function == "Near-Field, Far-Field Comparison":
            from core.hardware.cameras.qhyccd_control import convert_to_us
            exposure_times = {
                "exit_cam": convert_to_us(self.exposure_time_input_gt_2.text()),
                "entrance_cam": self.exposure_time_input_gt.text()
            }

            if type(self.main.fiber_dimension) == "str":
                fiber_dimension =  float(self.main.fiber_dimension)
            else:
                fiber_dimension = float(self.main.fiber_dimension)

            # Define wd
            wdir = os.path.join(working_dir, "NF_FF_Comparison")

            # Define driving width
            driving_width = float(self.driving_width_input.text())

            # Define number of positions
            number_of_positions = int(self.number_pos_input.text())

            if number_of_positions == 0:
                self.main_init.show_message("Number of positions must be greater than 0...")
                return

            # Define scale
            scale_dict = {"Linear": "lin", "Logarithmic": "log"}
            scale = scale_dict[self.scale_combo.currentText()]

            # First capture the images
            analysis.general_analysis.nf_ff_capture(wdir, fiber_diameter=fiber_dimension, exposure_times=exposure_times,
                                                    progress_signal=self.main.progress_signal,
                                                    driving_width=driving_width, number_of_positions=number_of_positions
                                                    )
            self.main.progress_signal.emit(f"Capture done, now processing...")
            # Then analyze the images
            analysis.general_analysis.nf_ff_process(wdir, fiber_diameter=fiber_dimension,
                                                    progress_signal=self.main.progress_signal, output_scale=scale)

        self.main.progress_signal.emit(f"{selected_function} complete.")
        self.main.experiment_running = False
        self.main_init.update_ui_state()
        self.main_init.log_data(f"General function completed: {selected_function}.")  # Log function completion

    def update_general_tab_buttons(self):
        if self.main_init.tabs.currentWidget() != self.main_init.general_tab:
            return

        selected_function = self.general_function_combo.currentText()
        self.main_init.log_data(
            f"General tab buttons updated for function: {selected_function}."
        )

        spacer_height = self._calculate_spacer_height(selected_function)
        if hasattr(self.main, "placeholder_spacer"):
            self.main.layout.removeItem(self.main.placeholder_spacer)
            del self.main.placeholder_spacer
        self.main.insert_spacer(spacer_height)

        self._hide_all_general_elements()
        self._show_elements_for_function(selected_function)
        self._update_run_button_state(selected_function)

    def _calculate_spacer_height(self, selected_function: str) -> int:
        if selected_function in [
            "Adjust Tip/Tilt",
            "Motor Controller: Reference",
            "Motor Controller: Move to Position",
            "Measure Eccentricity",
            "Change Color Filter",
            "Change System F-ratio",
        ]:
            return 162
        if selected_function in [
            "Make Throughput Calibration",
            "FF with each Filter",
            "Measure Fiber Size",
        ]:
            return 82
        if selected_function in [
            "Measure System F-ratio",
            "Near-Field, Far-Field Comparison",
        ]:
            return 0
        return 200

    def _hide_all_general_elements(self) -> None:
        self.main_init.create_datasheet_button.hide()
        self.main_init.open_fiber_data_button.hide()
        self.main_init.recent_folders_combo.hide()
        self.main_init.recent_folders_label.hide()
        self.main_init.choose_folder_button.hide()
        self.exposure_time_label_gt.hide()
        self.exposure_time_input_gt.hide()
        self.exposure_time_label_gt_2.hide()
        self.exposure_time_input_gt_2.hide()
        self.number_input_label.hide()
        self.number_input.hide()
        self.filter_input_label.hide()
        self.filter_input_combo.hide()
        self.fratio_input_label.hide()
        self.fratio_input_combo.hide()
        self.stop_button.hide()
        self.main_init.folder_name_input.hide()
        self.main_init.folder_name_label.hide()
        self.main_init.working_dir_label.hide()
        self.main_init.working_dir_display.hide()
        self.main_init.comments_button.hide()
        self.driving_width_input.hide()
        self.driving_width_label.hide()
        self.number_pos_input.hide()
        self.number_pos_input_label.hide()
        self.scale_label.hide()
        self.scale_combo.hide()
        self.motor_drive_label.hide()
        self.motor_drive_button_0.hide()
        self.motor_drive_button_5.hide()
        self.motor_drive_button_9_9.hide()
        self.run_button_gt.setDisabled(True)
        self.main_init.folder_name_label.setText("Folder Name:")
        self.main_init.show_message("")

    def _show_elements_for_function(self, selected_function: str) -> None:
        if selected_function == "Measure System F-ratio":
            print("Setting focus policy to StrongFocus for folder name input.")
            self.main_init.folder_name_input.show()
            self.main_init.folder_name_label.show()
            self.main_init.working_dir_label.show()
            self.main_init.working_dir_display.show()
            self.main_init.comments_button.show()
            self.exposure_time_label_gt.show()
            self.exposure_time_input_gt.show()
            self.main_init.open_fiber_data_button.show()
            self.main_init.recent_folders_combo.show()
            self.main_init.recent_folders_label.show()
            self.main_init.choose_folder_button.show()
        elif selected_function in ["Adjust Tip/Tilt", "Measure Eccentricity"]:
            self.stop_button.show()
        elif selected_function == "Motor Controller: Move to Position":
            self.number_input_label.show()
            self.number_input.show()
            self.motor_drive_label.show()
            self.motor_drive_button_0.show()
            self.motor_drive_button_5.show()
            self.motor_drive_button_9_9.show()
        elif selected_function == "FF with each Filter":
            self.main_init.folder_name_label.show()
            self.main_init.folder_name_input.show()
            self.main_init.working_dir_label.show()
            self.main_init.working_dir_display.show()
            self.main_init.comments_button.show()
        elif selected_function == "Change Color Filter":
            self.filter_input_label.show()
            self.filter_input_combo.show()
        elif selected_function == "Change System F-ratio":
            if not self.main.filter_wheel_initiated:
                threading.Thread(target=self.main.initialize_filter_wheel).start()
            self.fratio_input_label.show()
            self.fratio_input_combo.show()
            self.fratio_input_combo.setDisabled(not self.main.filter_wheel_ready)
        elif selected_function == "Measure Fiber Size":
            self.main_init.folder_name_input.setText("Fiber_Size_Measurement")
            self.main_init.folder_name_label.show()
            self.main_init.folder_name_input.show()
            self.main_init.working_dir_label.show()
            self.main_init.working_dir_display.show()
            self.main_init.comments_button.show()
            self.exposure_time_label_gt.show()
            self.exposure_time_input_gt.show()
        elif selected_function == "Near-Field, Far-Field Comparison":
            self.main_init.folder_name_input.setText(self.main.folder_name)
            self.main_init.folder_name_input.show()
            self.main_init.folder_name_label.show()
            self.main_init.working_dir_label.show()
            self.main_init.working_dir_display.show()
            self.main_init.comments_button.show()
            self.main_init.open_fiber_data_button.show()
            self.main_init.recent_folders_combo.show()
            self.main_init.recent_folders_label.show()
            self.main_init.choose_folder_button.show()
            self.exposure_time_label_gt.show()
            self.exposure_time_input_gt.show()
            self.exposure_time_label_gt_2.show()
            self.exposure_time_input_gt_2.show()
            self.driving_width_input.show()
            self.driving_width_label.show()
            self.number_pos_input.show()
            self.number_pos_input_label.show()
            self.scale_label.show()
            self.scale_combo.show()
            if not self.main.fiber_dimension:
                self.main_init.show_message(
                    "Please enter fiber dimension before running the function."
                )
                return

            fiber_dimension = (
                float(self.main.fiber_dimension)
                if isinstance(self.main.fiber_dimension, str)
                else float(self.main.fiber_dimension)
            )

            if isinstance(fiber_dimension, (list, tuple)):
                fiber_radius = max(fiber_dimension[0], fiber_dimension[1])
            else:
                fiber_radius = fiber_dimension

            self.driving_width_input.setText(str(fiber_radius))
        elif selected_function == "Make Throughput Calibration":
            self.main_init.folder_name_input.show()
            self.main_init.folder_name_label.show()
            self.main_init.working_dir_label.show()
            self.main_init.working_dir_display.show()
            self.main_init.comments_button.show()
            self.main_init.folder_name_label.setText("Calibration Name:")

    def _update_run_button_state(self, selected_function: str) -> None:
        if selected_function in [
            "Measure System F-ratio",
            "Near-Field, Far-Field Comparison",
            "FF with each Filter",
            "Make Throughput Calibration",
            "Measure Fiber Size",
        ]:
            self.run_button_gt.setDisabled(
                self.main_init.folder_name_input.text() == ""
            )
        elif selected_function in ["Change System F-ratio"] and not self.main.filter_wheel_ready:
            self.run_button_gt.setDisabled(True)
        else:
            self.run_button_gt.setDisabled(False)
