from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QUrl, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

import threading, os, json

class AnalyseTab:
    def __init__(self, main, main_init):
        self.main = main
        self.main_init = main_init
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
        self.calibration_folder_button.clicked.connect(self.main.choose_calibration_folder)

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

    def run_analysis(self):
        if not self.main.folder_name and self.main.fiber_dimension and self.main.fiber_shape != "":
            self.show_message("Please lock the inputs before running the analysis.")
            return

        analysis_type = self.analysis_type_combo.currentText()
        working_dir = self.main_init.working_dir_display.text()
        fiber_shape = self.main.fiber_shape

        if fiber_shape == "rectangular":
            fiber_diameter = (int(self.main.fiber_dimension[0]), int(self.main.fiber_dimension[1]))
        else:
            fiber_diameter = int(self.main.fiber_dimension)

        calibration_folder = self.calibration_folder_input.text() if analysis_type == "Throughput" else None

        self.main.experiment_running = True
        self.main_init.update_ui_state()

        threading.Thread(target=self.run_analysis_thread,
                         args=(analysis_type, working_dir, fiber_diameter, fiber_shape, calibration_folder)).start()

    def run_analysis_thread(self, analysis_type, working_dir, fiber_diameter, fiber_shape, calibration_folder):
        self.main.progress_signal.emit("Starting analysis...")
        if analysis_type == "SG":
            directory = os.path.join(working_dir, "SG")
            import sg_pipeline
            if self.get_params_checkbox.isChecked():
                print("Getting SG parameters with fiber diameter:", fiber_diameter, "and fiber shape:", fiber_shape)
                sg_pipeline.get_sg_params(directory, fiber_diameter, fiber_shape, progress_signal=self.main.progress_signal)

            if self.plot_sg_checkbox.isChecked():
                sg_pipeline.plot_sg_cool_like(directory, fiber_diameter, progress_signal=self.main.progress_signal)

            if self.calc_sg_checkbox.isChecked():
                sg_pipeline.calc_sg(directory, progress_signal=self.main.progress_signal)

            if self.plot_coms_checkbox.isChecked():
                sg_pipeline.plot_coms(directory, progress_signal=self.main.progress_signal)

            if self.plot_masks_checkbox.isChecked():
                sg_pipeline.plot_masks(directory, fiber_diameter, progress_signal=self.main.progress_signal)

            if self.make_video_checkbox.isChecked():
                sg_pipeline.make_comparison_video(directory, fiber_diameter)

            if self.plot_com_comk_on_image_cut_checkbox.isChecked():
                self.main.progress_signal.emit("Running plot_com_comk_on_image_cut...")
                sg_pipeline.plot_com_comk_on_image_cut(directory)

            if self.sg_new_checkbox.isChecked():
                sg_pipeline.sg_new(directory, progress_signal=self.main.progress_signal)

            if self.plot_nf_horizontal_cut_checkbox.isChecked():
                sg_pipeline.plot_horizontal_cut_nf(directory)

        elif analysis_type == "FRD":
            directory = os.path.join(working_dir, "FRD")
            import fiber_frd_measurements as frd
            if self.calc_frd_checkbox.isChecked():
                frd.main_analyse_all_filters(directory, progress_signal=self.main.progress_signal)
            if self.plot_sutherland_checkbox.isChecked():
                frd.sutherland_plot(directory)
            if self.plot_f_ratio_circles_on_raw_checkbox.isChecked():
                frd.plot_f_ratio_circles_on_raw(directory)
            if self.plot_nf_horizontal_cut_checkbox.isChecked():
                frd.plot_horizontal_cut_ff(directory)

        elif analysis_type == "Throughput":
            directory = os.path.join(working_dir, "Throughput")
            import throughput_analysis
            throughput_analysis.main(directory, calibration_folder)

        self.main.progress_signal.emit("Analysis complete.")
        self.main.experiment_running = False
        self.main_init.update_ui_state()

    def update_analysis_tab(self):
        analysis_type = self.analysis_type_combo.currentText()

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
            self.calibration_folder_label.hide()
            self.calibration_folder_input.hide()
            self.calibration_folder_button.hide()
            self.plot_ff_horizontal_cut_checkbox.show()
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

