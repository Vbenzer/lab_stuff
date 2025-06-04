"""Module analyse_tab.py.

Auto-generated docstring for better readability.
"""
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QCheckBox, QVBoxLayout, QFileDialog
                             )

import threading, os

import analysis.frd_analysis
import analysis.sg_analysis
import analysis.visualization
from gui.tabs.helpers import load_recent_folders, update_recent_folders


class AnalyseTab:
    def __init__(self, main, main_init):
        self.main = main
        self.main_init = main_init
        self.main_init.log_data("AnalyseTab initialized.")  # Log initialization
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
        self.plot_nf_horizontal_cut_checkbox = QCheckBox("Plot NF Horizontal Cut")
        self.plot_ff_horizontal_cut_checkbox = QCheckBox("Plot FF Horizontal Cut")
        self.plot_com_comk_on_image_cut_checkbox = QCheckBox("Plot COM and COMK on Cut Image")

        self.plot_sg_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.calc_sg_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.plot_coms_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.get_params_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.plot_masks_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.make_video_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.sg_new_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.calc_frd_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.plot_sutherland_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.plot_f_ratio_circles_on_raw_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.plot_nf_horizontal_cut_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.plot_ff_horizontal_cut_checkbox.stateChanged.connect(self.main_init.update_run_button_state)
        self.plot_com_comk_on_image_cut_checkbox.stateChanged.connect(self.main_init.update_run_button_state)

        self.calibration_folder_label = QLabel("Calibration Folder:")
        self.calibration_folder_input = QLineEdit()
        self.calibration_folder_input.textChanged.connect(self.main_init.update_run_button_state)
        self.calibration_folder_button = QPushButton("Choose Calibration Folder")
        self.calibration_folder_button.clicked.connect(lambda: self.choose_calibration_folder(mode="FRD"))


        layout.addWidget(self.get_params_checkbox)
        layout.addWidget(self.plot_sg_checkbox)
        layout.addWidget(self.calc_sg_checkbox)
        layout.addWidget(self.plot_coms_checkbox)
        layout.addWidget(self.plot_masks_checkbox)
        layout.addWidget(self.make_video_checkbox)
        layout.addWidget(self.plot_com_comk_on_image_cut_checkbox)
        layout.addWidget(self.sg_new_checkbox)
        layout.addWidget(self.calc_frd_checkbox)
        layout.addWidget(self.plot_sutherland_checkbox)
        layout.addWidget(self.plot_f_ratio_circles_on_raw_checkbox)
        layout.addWidget(self.plot_nf_horizontal_cut_checkbox)
        layout.addWidget(self.plot_ff_horizontal_cut_checkbox)

        layout.addWidget(self.calibration_folder_label)
        layout.addWidget(self.calibration_folder_input)
        layout.addWidget(self.calibration_folder_button)

        layout.addStretch()

        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.setDisabled(True)
        self.run_analysis_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_analysis_button)

        self.main_init.analyse_tab.setLayout(layout)
        self.update_analysis_tab()

    def choose_calibration_folder(self, mode:str):
        if mode not in ["FRD", "Throughput"]:
            raise ValueError("Mode must be either 'FRD' or 'Throughput'.")
        if mode == "FRD":
            folder_path = QFileDialog.getExistingDirectory(self.main, "Select FRD Calibration Folder",
                                                           self.main.base_directory + "/FRD_Calibrations")
            if folder_path:
                folder_name = os.path.basename(folder_path)
                self.calibration_folder_input.setText(folder_name)
                recent_folders = load_recent_folders(self.main.base_directory + "/recent_frd_calibration_folders.json")
                update_recent_folders(folder_name, recent_folders,
                                      file_path=self.main.base_directory + "/recent_frd_calibration_folders.json")

        elif mode == "Throughput":
            folder_path = QFileDialog.getExistingDirectory(self.main, "Select SG Calibration Folder",
                                                           self.main.base_directory + "/Throughput_Calibrations")
            if folder_path:
                folder_name = os.path.basename(folder_path)
                self.calibration_folder_input.setText(folder_name)

                recent_folders = load_recent_folders(self.main.base_directory + "/recent_throughput_calibration_folders.json")
                update_recent_folders(folder_name, recent_folders,
                                      file_path=self.main.base_directory + "/recent_throughput_calibration_folders.json")

    def run_analysis(self):
        if not self.main.folder_name and self.main.fiber_dimension and self.main.fiber_shape != "":
            self.main_init.log_data("Run analysis failed: Inputs not locked.")  # Log failure
            self.show_message("Please lock the inputs before running the analysis.")
            return

        analysis_type = self.analysis_type_combo.currentText()
        self.main_init.log_data(f"Run analysis started for type: {analysis_type}")  # Log analysis type
        working_dir = self.main_init.working_dir_display.text()
        fiber_shape = self.main.fiber_shape

        if fiber_shape == "rectangular":
            fiber_diameter = (int(self.main.fiber_dimension[0]), int(self.main.fiber_dimension[1]))
        else:
            fiber_diameter = int(self.main.fiber_dimension)

        calibration_folder = self.calibration_folder_input.text() if analysis_type == "Throughput" else None

        self.main.experiment_running = True
        self.main_init.update_ui_state()
        self.main_init.log_data("Experiment running state set to True.")  # Log state change

        threading.Thread(target=self.run_analysis_thread,
                         args=(analysis_type, working_dir, fiber_diameter, fiber_shape, calibration_folder)).start()

    def get_latest_calibration_folder(self, mode:str):
        """
        Get the latest calibration folder from the main instance.
        Returns:
            str: Path to the latest calibration folder.
            mode: The type of calibration folder to look for, either "FRD" or "Throughput".
        """

        if mode == "FRD":
            # Find the latest folder containing 'FRD_Calibration' in its name
            calibration_folder = self.main.base_directory + "/FRD_Calibrations"
            folders = [f for f in os.listdir(calibration_folder) if
                       "FRD_Calibration" in f and os.path.isdir(os.path.join(calibration_folder, f))]
            if not folders:
                return None
            latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(calibration_folder, f)))
            self.main_init.log_data(f"Latest FRD calibration folder: {latest_folder}")  # Log latest folder
            self.main.progress_signal.emit(f"Using latest FRD calibration folder: {latest_folder}")
            return latest_folder

        elif mode == "Throughput":
            # Find the latest folder containing 'Throughput_Calibration' in its name
            calibration_folder = self.main.base_directory + "/Throughput_Calibrations"
            folders = [f for f in os.listdir(calibration_folder) if
                       "Throughput_Calibration" in f and os.path.isdir(os.path.join(calibration_folder, f))]
            if not folders:
                return None
            latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(calibration_folder, f)))
            self.main_init.log_data(f"Latest Throughput calibration folder: {latest_folder}")
            self.main.progress_signal.emit(f"Using latest Throughput calibration folder: {latest_folder}")
            return latest_folder

        else:
            return None


    def run_analysis_thread(self, analysis_type, working_dir, fiber_diameter, fiber_shape, calibration_folder):
        self.main.progress_signal.emit("Starting analysis...")
        self.main_init.log_data(f"Analysis thread started for type: {analysis_type}")  # Log thread start
        if analysis_type == "SG":
            directory = os.path.join(working_dir, "SG")
            if self.get_params_checkbox.isChecked():
                self.main_init.log_data("Getting SG parameters.")  # Log SG parameters
                self.main.progress_signal.emit("Getting SG parameters.")
                analysis.sg_analysis.get_sg_params(directory, fiber_diameter, fiber_shape, progress_signal=self.main.progress_signal)

            if self.plot_sg_checkbox.isChecked():
                self.main_init.log_data("Plotting SG.")  # Log SG plotting
                self.main.progress_signal.emit("Plotting SG.")
                from unused_functions.unused_sg_functions import plot_sg_cool_like
                plot_sg_cool_like(directory, fiber_diameter, progress_signal=self.main.progress_signal)

            if self.calc_sg_checkbox.isChecked():
                self.main_init.log_data("Calculating SG.")  # Log SG calculation
                self.main.progress_signal.emit("Calculating SG.")
                from unused_functions.unused_sg_functions import calc_sg
                calc_sg(directory, progress_signal=self.main.progress_signal)

            if self.plot_coms_checkbox.isChecked():
                self.main_init.log_data("Plotting COMs.")  # Log COMs plotting
                self.main.progress_signal.emit("Plotting COMs.")
                analysis.visualization.plot_coms(directory, progress_signal=self.main.progress_signal)

            if self.plot_masks_checkbox.isChecked():
                self.main_init.log_data("Plotting masks.")  # Log mask plotting
                self.main.progress_signal.emit("Plotting masks.")
                analysis.visualization.plot_masks(directory, fiber_diameter, progress_signal=self.main.progress_signal)

            if self.make_video_checkbox.isChecked():
                self.main_init.log_data("Making comparison video.")  # Log video creation
                self.main.progress_signal.emit("Making comparison video.")
                analysis.visualization.make_comparison_video(directory, fiber_diameter,
                                                             progress_signal=self.main.progress_signal)

            if self.plot_com_comk_on_image_cut_checkbox.isChecked():
                self.main_init.log_data("Plotting COM and COMK on image cut.")  # Log COM/COMK plotting
                self.main.progress_signal.emit("Plotting COM and COMK on image cut.")
                analysis.visualization.plot_com_comk_on_image_cut(directory, progress_signal=self.main.progress_signal)

            if self.sg_new_checkbox.isChecked():
                self.main_init.log_data("Running SG new analysis.")  # Log SG new analysis
                self.main.progress_signal.emit("Running SG new analysis.")
                analysis.sg_analysis.sg_new(directory, progress_signal=self.main.progress_signal)

            if self.plot_nf_horizontal_cut_checkbox.isChecked():
                self.main_init.log_data("Plotting NF horizontal cut.")  # Log NF horizontal cut
                self.main.progress_signal.emit("Plotting NF horizontal cut.")
                analysis.visualization.plot_horizontal_cut_nf(directory)

        elif analysis_type == "FRD":
            directory = os.path.join(working_dir, "FRD")
            calibration_folder = self.main.base_directory + "/FRD_Calibrations/" + self.calibration_folder_input.text()
            if self.calc_frd_checkbox.isChecked():
                self.main_init.log_data("Calculating FRD.")  # Log FRD calculation
                self.main.progress_signal.emit("Calculating FRD.")
                analysis.frd_analysis.main_analyse_all_filters(directory, calibration_folder, progress_signal=self.main.progress_signal)
            if self.plot_sutherland_checkbox.isChecked():
                self.main_init.log_data("Plotting Sutherland plot.")  # Log Sutherland plot
                self.main.progress_signal.emit("Plotting Sutherland plot.")
                analysis.visualization.sutherland_plot(directory)
            if self.plot_f_ratio_circles_on_raw_checkbox.isChecked():
                self.main_init.log_data("Plotting F-ratio circles on raw image.")  # Log F-ratio circles
                self.main.progress_signal.emit("Plotting F-ratio circles on raw image.")
                analysis.visualization.plot_f_ratio_circles_on_raw(directory)
            if self.plot_nf_horizontal_cut_checkbox.isChecked():
                self.main_init.log_data("Plotting NF horizontal cut for FRD.")  # Log NF horizontal cut for FRD
                self.main.progress_signal.emit("Plotting NF horizontal cut for FRD.")
                analysis.visualization.plot_horizontal_cut_ff(directory)

        elif analysis_type == "Throughput":
            directory = os.path.join(working_dir, "Throughput")
            self.main_init.log_data("Running throughput analysis.")  # Log throughput analysis
            from analysis import throughput_analysis
            throughput_analysis.main(directory, calibration_folder)

        self.main.progress_signal.emit("Analysis complete.")
        self.main_init.log_data("Analysis complete.")  # Log analysis completion
        self.main.experiment_running = False
        self.main_init.update_ui_state()
        self.main_init.log_data("Experiment running state set to False.")  # Log state change

    def update_analysis_tab(self):
        analysis_type = self.analysis_type_combo.currentText()
        self.main_init.log_data(f"Analysis tab updated for type: {analysis_type}")  # Log tab update

        # Reset all checkboxes
        for checkbox in [self.plot_sg_checkbox, self.calc_sg_checkbox, self.plot_coms_checkbox,
                         self.get_params_checkbox, self.plot_masks_checkbox, self.make_video_checkbox,
                         self.sg_new_checkbox, self.calc_frd_checkbox, self.plot_sutherland_checkbox,
                         self.plot_f_ratio_circles_on_raw_checkbox, self.plot_nf_horizontal_cut_checkbox,
                         self.plot_ff_horizontal_cut_checkbox, self.plot_com_comk_on_image_cut_checkbox,
                         ]:
            checkbox.setChecked(False)

        if analysis_type == "SG":
            self.plot_sg_checkbox.show()
            self.calc_sg_checkbox.show()
            self.plot_coms_checkbox.show()
            self.get_params_checkbox.show()
            self.plot_masks_checkbox.show()
            self.make_video_checkbox.show()
            self.sg_new_checkbox.show()
            self.plot_nf_horizontal_cut_checkbox.show()
            self.plot_com_comk_on_image_cut_checkbox.show()
            self.calc_frd_checkbox.hide()
            self.plot_sutherland_checkbox.hide()
            self.plot_f_ratio_circles_on_raw_checkbox.hide()
            self.calibration_folder_label.hide()
            self.calibration_folder_input.hide()
            self.calibration_folder_button.hide()
            self.plot_ff_horizontal_cut_checkbox.hide()
        elif analysis_type == "FRD":
            # Add options for FRD analysis here
            self.plot_sg_checkbox.hide()
            self.calc_sg_checkbox.hide()
            self.plot_coms_checkbox.hide()
            self.get_params_checkbox.hide()
            self.plot_masks_checkbox.hide()
            self.make_video_checkbox.hide()
            self.sg_new_checkbox.hide()
            self.plot_nf_horizontal_cut_checkbox.hide()
            self.plot_com_comk_on_image_cut_checkbox.hide()
            self.calc_frd_checkbox.show()
            self.plot_sutherland_checkbox.show()
            self.plot_f_ratio_circles_on_raw_checkbox.show()
            self.calibration_folder_label.show()
            self.calibration_folder_input.show()
            self.calibration_folder_button.show()
            self.plot_ff_horizontal_cut_checkbox.show()
            self.calibration_folder_button.clicked.disconnect()
            self.calibration_folder_button.clicked.connect(lambda: self.choose_calibration_folder(mode="FRD"))
            recent_folders = load_recent_folders(self.main.base_directory + "/recent_frd_calibration_folders.json")
            self.calibration_folder_input.setText(recent_folders[0] if recent_folders else "")

        elif analysis_type == "Throughput":
            self.plot_sg_checkbox.hide()
            self.calc_sg_checkbox.hide()
            self.plot_coms_checkbox.hide()
            self.get_params_checkbox.hide()
            self.plot_masks_checkbox.hide()
            self.make_video_checkbox.hide()
            self.sg_new_checkbox.hide()
            self.plot_nf_horizontal_cut_checkbox.hide()
            self.plot_com_comk_on_image_cut_checkbox.hide()
            self.calc_frd_checkbox.hide()
            self.plot_sutherland_checkbox.hide()
            self.plot_f_ratio_circles_on_raw_checkbox.hide()
            self.calibration_folder_label.show()
            self.calibration_folder_input.show()
            self.calibration_folder_button.show()
            self.plot_ff_horizontal_cut_checkbox.hide()
            self.calibration_folder_button.clicked.disconnect()
            self.calibration_folder_button.clicked.connect(lambda: self.choose_calibration_folder(mode="Throughput"))
            recent_folders = load_recent_folders(self.main.base_directory + "/recent_throughput_calibration_folders.json")
            self.calibration_folder_input.setText(recent_folders[0] if recent_folders else "")
