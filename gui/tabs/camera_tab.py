from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QUrl, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

from gui.tabs.helpers import HelperFunctions

import threading

class CameraTab(HelperFunctions):
    def __init__(self, main, main_init):
        self.main = main
        self.main_init = main_init
        self.main_init.update_input_visibility()

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
        self.run_button_ct = QPushButton("Run")
        self.run_button_ct.setDisabled(True)  # Initially disabled
        self.run_button_ct.clicked.connect(self.run_camera_function)
        layout.addWidget(self.run_button_ct)

        self.main_init.camera_tab.setLayout(layout)

    def run_camera_function(self):
        selected_function = self.camera_function_combo.currentText()
        folder_name = self.main.folder_name

        if selected_function in ["Thorlabs Camera", "Qhyccd Camera"] and folder_name != "":
            self.show_message("Please enter folder name before running the function.")
            return

        if not self.exposure_time_input.hasAcceptableInput():
            return

        self.main.experiment_running = True
        self.main_init.update_ui_state()

        working_dir = self.main_init.working_dir_display.text()

        os.makedirs(working_dir, exist_ok=True)

        threading.Thread(target=self.run_camera_function_thread, args=(selected_function, working_dir)).start()

    def run_camera_function_thread(self, selected_function, working_dir):
        self.main.progress_signal.emit(f"Running {selected_function}...")
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

        self.main.progress_signal.emit(f"{selected_function} complete.")
        self.main.experiment_running = False
        self.main_init.update_ui_state()

    def update_camera_tab_buttons(self):
        if self.main_init.tabs.currentWidget() != self.main_init.camera_tab:
            return

        self.main_init.update_run_button_state()

        selected_function = self.camera_function_combo.currentText()

        if selected_function == "Thorlabs Camera Live":
            self.main_init.folder_name_input.hide()
            self.main_init.folder_name_label.hide()
            self.main_init.choose_folder_button.hide()
            self.main_init.recent_folders_combo.hide()
            self.main_init.recent_folders_label.hide()
            self.main_init.working_dir_label.hide()
            self.main_init.working_dir_display.hide()
            self.main_init.comments_button.hide()
            self.exposure_time_label.hide()
            self.exposure_time_input.hide()
            self.run_button_ct.setDisabled(False)

            if hasattr(self.main, 'placeholder_spacer'):
                self.main.layout.removeItem(self.main.placeholder_spacer)
                del self.main.placeholder_spacer
                self.main.insert_spacer(162)
            else:
                self.main.insert_spacer(162)

        else:
            if hasattr(self.main, 'placeholder_spacer'):
                self.main.layout.removeItem(self.main.placeholder_spacer)
                del self.main.placeholder_spacer
                self.main.insert_spacer(30)
            else:
                self.main.insert_spacer(30)

            self.main_init.folder_name_input.show()
            self.main_init.folder_name_label.show()
            self.main_init.choose_folder_button.show()
            self.main_init.recent_folders_combo.show()
            self.main_init.recent_folders_label.show()
            self.main_init.working_dir_label.show()
            self.main_init.working_dir_display.show()
            self.main_init.comments_button.show()
            self.exposure_time_label.show()
            self.exposure_time_input.show()
            self.run_button_ct.setDisabled(self.main_init.folder_name_input.text() == "")

        if selected_function == "Thorlabs Camera Single":
            self.camera_chooser_label.show()
            self.camera_chooser_combo.show()
        else:
            self.camera_chooser_label.hide()
            self.camera_chooser_combo.hide()

