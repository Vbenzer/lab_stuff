from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QUrl, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator
from gui.tabs.helpers import HelperFunctions

import threading

class GeneralTab(HelperFunctions):
    def __init__(self, main, main_init):
        self.main = main
        self.main_init = main_init
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
        self.main_init.folder_name_input.textChanged.connect(self.update_general_tab_buttons)
        self.update_general_tab_buttons()

    def run_general_function(self):
        selected_function = self.general_function_combo.currentText()

        if (selected_function in ["Measure Eccentricity", "Adjust Tip/Tilt", "FF with each Filter", "Measure Fiber Size"]
                and self.main.folder_name != ""):
            self.show_message("Please enter folder name before running the function.")
            return

        self.main.experiment_running = True
        self.main_init.update_ui_state()

        working_dir = self.main_init.working_dir_display.text()
        threading.Thread(target=self.run_general_function_thread, args=(selected_function, working_dir)).start()

    def run_general_function_thread(self, selected_function, working_dir):
        self.main.progress_signal.emit(f"Running {selected_function}...")
        self.stop_event = threading.Event()
        if selected_function == "Measure System F-ratio":
            """import fiber_frd_measurements as frd
            frd.main_measure_all_filters(working_dir, progress_signal=self.progress_signal,
                                         base_directory=self.base_directory)"""
            import qhy_ccd_take_image as qhy
            exposure_time = qhy.convert_to_us(self.exposure_time_input_gt.text())
            import analyse_main as am
            am.main_measure_new(working_dir, progress_signal=self.main.progress_signal, exp_time=exposure_time)

            import fiber_frd_measurements as frd
            frd.main_analyse_all_filters(working_dir, progress_signal=self.main.progress_signal)

        elif selected_function == "Make Throughput Calibration":
            import throughput_analysis as ta
            calibration_folder_name = os.path.basename(working_dir)
            ta.measure_all_filters(working_dir, progress_signal=self.main.progress_signal, calibration=calibration_folder_name,
                                   base_directory=self.main.base_directory)
        elif selected_function == "Adjust Tip/Tilt":
            import qhyccd_cam_control
            qhyccd_cam_control.use_camera("tiptilt", self.stop_event)
        elif selected_function == "Motor Controller: Reference":
            import step_motor_control as smc
            smc.open_connection()
            smc.make_reference_move()
            smc.close_connection()
        elif selected_function == "Motor Controller: Move to Position":
            import step_motor_control as smc
            smc.open_connection()
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
            move_to_filter.move(filter_name)
        elif selected_function == "Change System F-ratio":
            # Check if the filter wheel is available
            if self.filter_wheel:
                f_ratio = self.fratio_input_combo.currentText()
                self.filter_wheel.move_to_filter(f_ratio)
        elif selected_function == "Measure Fiber Size":
            import fiber_frd_measurements as frd
            exposure_times = {
                "exit_cam": self.exposure_time_input_gt.text()
            }
            frd.measure_fiber_size(self.working_dir, exposure_times=exposure_times)
        elif selected_function == "Near-Field, Far-Field Comparison":
            import fiber_frd_measurements as frd
            exposure_times = {
                "exit_cam": self.exposure_time_input_gt_2.text(),
                "entrance_cam": self.exposure_time_input_gt.text()
            }
            if type(self.fiber_dimension) == "str":
                fiber_dimension =  float(self.fiber_dimension)
            else:
                fiber_dimension = self.fiber_dimension
            # First capture the images
            frd.nf_ff_capture(self.working_dir, fiber_diameter=fiber_dimension, exposure_times=exposure_times)
            self.main.progress_signal.emit(f"Capture done, now processing...")
            # Then analyze the images
            frd.nf_ff_process(self.working_dir, fiber_diameter=fiber_dimension)

        self.main.progress_signal.emit(f"{selected_function} complete.")
        self.main.experiment_running = False
        self.main_init.update_ui_state()

    def update_general_tab_buttons(self):
        if self.main_init.tabs.currentWidget() != self.main_init.general_tab:
            return

        selected_function = self.general_function_combo.currentText()

        # Spacer management
        spacer_height = (
            162 if selected_function in ["Adjust Tip/Tilt", "Motor Controller: Reference",
                                         "Motor Controller: Move to Position", "Measure Eccentricity",
                                         "Change Color Filter",
                                         "Change System F-ratio"
                                         ]
            else 82 if selected_function in ["Measure System F-ratio", "Make Throughput Calibration",
                                             "FF with each Filter", "Measure Fiber Size",
                                             ]
            else 0 if selected_function in ["Near-Field, Far-Field Comparison"]
            else 200
        )

        if hasattr(self.main, 'placeholder_spacer'):
            self.main.layout.removeItem(self.main.placeholder_spacer)
            del self.main.placeholder_spacer
        self.main.insert_spacer(spacer_height)

        # Hide all elements by default
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
        self.main_init.folder_name_label.setText("Folder Name:")
        self.main_init.show_message("")
        self.run_button_gt.setDisabled(True)

        # Handle specific functions
        if selected_function == "Measure System F-ratio":
            self.main_init.folder_name_input.show()
            self.main_init.folder_name_label.show()
            self.main_init.working_dir_label.show()
            self.main_init.working_dir_display.show()
            self.main_init.comments_button.show()
            self.exposure_time_label_gt.show()
            self.exposure_time_input_gt.show()

        elif selected_function in ["Adjust Tip/Tilt", "Measure Eccentricity"]:
            self.stop_button.show()

        elif selected_function == "Motor Controller: Move to Position":
            self.number_input_label.show()
            self.number_input.show()

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
            if not self.main.fiber_dimension:
                self.main_init.show_message("Please enter fiber dimension before running the function.")
                return

        elif selected_function == "Make Throughput Calibration":
            self.main_init.folder_name_input.show()
            self.main_init.folder_name_label.show()
            self.main_init.working_dir_label.show()
            self.main_init.working_dir_display.show()
            self.main_init.comments_button.show()
            self.main_init.folder_name_label.setText("Calibration Name:")

        # Run button state
        if selected_function in ["Measure System F-ratio", "Near-Field, Far-Field Comparison", "FF with each Filter",
                                 "Make Throughput Calibration", "Measure Fiber Size"]:
            self.run_button_gt.setDisabled(self.main_init.folder_name_input.text() == "")
        elif selected_function in ["Change System F-ratio"] and not self.main.filter_wheel_ready:
            self.run_button_gt.setDisabled(True)
        else:
            self.run_button_gt.setDisabled(False)
