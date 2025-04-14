from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QUrl, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

class CameraTab:
    def __init__(self, main_ctrl):
        self.main = main_ctrl
        self.main.update_input_visibility()

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

        self.main.camera_tab.setLayout(layout)

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

        # Create folder if it doesnt exits
        os.makedirs(working_dir, exist_ok=True)

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

    def update_camera_tab_buttons(self):
        if self.tabs.currentWidget() != self.camera_tab:
            return

        self.update_run_button_state()

        selected_function = self.camera_function_combo.currentText()

        if selected_function == "Thorlabs Camera Live":
            self.folder_name_input.hide()
            self.folder_name_label.hide()
            self.choose_folder_button.hide()
            self.recent_folders_combo.hide()
            self.recent_folders_label.hide()
            self.working_dir_label.hide()
            self.working_dir_display.hide()
            self.comments_button.hide()
            self.exposure_time_label.hide()
            self.exposure_time_input.hide()
            self.run_button_ct.setDisabled(False)

            if hasattr(self, 'placeholder_spacer'):
                self.layout.removeItem(self.placeholder_spacer)
                del self.placeholder_spacer
                self.insert_spacer(162)
            else:
                self.insert_spacer(162)

        else:
            if hasattr(self, 'placeholder_spacer'):
                self.layout.removeItem(self.placeholder_spacer)
                del self.placeholder_spacer
                self.insert_spacer(30)
            else:
                self.insert_spacer(30)

            self.folder_name_input.show()
            self.folder_name_label.show()
            self.choose_folder_button.show()
            self.recent_folders_combo.show()
            self.recent_folders_label.show()
            self.working_dir_label.show()
            self.working_dir_display.show()
            self.comments_button.show()
            self.exposure_time_label.show()
            self.exposure_time_input.show()
            self.run_button_ct.setDisabled(self.folder_name_input.text() == "")

        if selected_function == "Thorlabs Camera Single":
            self.camera_chooser_label.show()
            self.camera_chooser_combo.show()
        else:
            self.camera_chooser_label.hide()
            self.camera_chooser_combo.hide()
