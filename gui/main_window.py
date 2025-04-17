from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QUrl, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

import os

from gui.tabs.helpers import HelperFunctions
from gui.tabs.analyse_tab import AnalyseTab
from gui.tabs.camera_tab import CameraTab
from gui.tabs.measure_tab import MeasureTab
from gui.tabs.general_tab import GeneralTab
from widgets import *

def load_recent_folders(file_path:str):
    import os, json
    """
    Load the recent folders from a JSON file.
    Args:
        file_path: Path of the JSON file to load from.

    Returns: List of recent folders.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def update_recent_folders(folder:str, recent_folders:list[str], max_recent=2, base_directory:str=None):
    """
    Update the list of recent folders.
    Args:
        folder: New folder to add.
        recent_folders: List of recent folders.
        max_recent: Maximum number of recent folders to keep.
        base_directory: Base directory to save the recent folders file.
    """
    if folder in recent_folders:
        recent_folders.remove(folder)
    recent_folders.insert(0, folder)
    if len(recent_folders) > max_recent:
        recent_folders.pop()
    save_recent_folders(recent_folders, file_path=base_directory + r'\recent_folders.json')

class MainWindowInit(HelperFunctions, Widgets):
    def __init__(self, main_ctrl):
        self.main = main_ctrl
        self.main_init = self  # Define reference to self as main_init

        self.folder_name_label = QLabel("Fiber Name:")
        self.folder_name_input = QLineEdit()
        self.folder_name_input.setFixedWidth(700)
        self.folder_name_input.setReadOnly(True)
        self.folder_name_input.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.folder_name_input.textChanged.connect(self.update_working_dir)
        self.folder_name_input.textChanged.connect(self.update_run_button_state)

        self.open_fiber_data_button = QPushButton("Open Fiber Data")
        self.open_fiber_data_button.clicked.connect(self.main.open_fiber_data_window)

        self.create_datasheet_button = QPushButton("Create Datasheet")
        self.create_datasheet_button.clicked.connect(self.create_datasheet)
        # Initially deactivated
        self.create_datasheet_button.setDisabled(True)

        self.working_dir_label = QLabel("Working Directory:")
        self.working_dir_display = QLabel("")

        self.choose_folder_button = QPushButton("Choose Existing Folder")
        self.choose_folder_button.clicked.connect(self.choose_folder)

        self.comments_button = QPushButton("Add Comments")
        self.comments_button.clicked.connect(self.access_comments_file)
        self.comments_button.setDisabled(True)
        self.folder_name_input.textChanged.connect(
            lambda: self.comments_button.setDisabled(self.folder_name_input.text() == ""))

        self.recent_folders = load_recent_folders(file_path=self.main.base_directory + r'\recent_folders.json')
        self.recent_folders_combo = QComboBox()
        self.update_recent_folders_combo()
        self.recent_folders_combo.currentIndexChanged.connect(self.select_recent_folder)

        self.progress_label = QLabel("")
        self.progress_text_edit = QTextEdit()
        self.progress_text_edit.setReadOnly(True)
        self.progress_text_edit.hide()  # Initially hidden

        self.main.layout.addLayout(self.create_hbox_layout(self.folder_name_label, self.folder_name_input))
        self.main.layout.addLayout(self.create_hbox_layout(self.working_dir_label, self.working_dir_display))

        self.main.layout.addLayout(self.create_hbox_layout(self.open_fiber_data_button, self.create_datasheet_button))

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_folder_button)
        button_layout.addWidget(self.comments_button)
        self.main.layout.addLayout(button_layout)

        self.recent_folders_label = QLabel("Recent Folders:")
        self.main.layout.addWidget(self.recent_folders_label)

        self.main.layout.addWidget(self.recent_folders_combo)
        self.main.layout.addWidget(self.main.message_label)
        self.main.layout.addWidget(self.progress_label)
        self.main.layout.addWidget(self.progress_text_edit)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.update_input_visibility)
        self.main.layout.addWidget(self.tabs)

        self.measure_tab = QWidget()
        self.analyse_tab = QWidget()
        self.general_tab = QWidget()
        self.camera_tab = QWidget()

        self.tabs.addTab(self.measure_tab, "Measure")
        self.tabs.addTab(self.analyse_tab, "Analyse")
        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.camera_tab, "Cameras")

        self.measure_tab_init = MeasureTab(self.main, self.main_init)
        self.analyse_tab_init = AnalyseTab(self.main, self.main_init)
        self.general_tab_init = GeneralTab(self.main, self.main_init)
        self.camera_tab_init = CameraTab(self.main, self.main_init)

        self.tabs.currentChanged.connect(self.update_ui_state)
        self.tabs.currentChanged.connect(self.update_run_button_state)

        self.main.progress_signal.connect(self.update_progress)

    def update_ui_state(self):
        state = self.main.experiment_running
        if state:
            self.analyse_tab_init.run_analysis_button.setDisabled(True)
            self.measure_tab_init.run_measurement_button.setDisabled(True)
            self.choose_folder_button.setDisabled(True)
        else:
            self.analyse_tab_init.run_analysis_button.setDisabled(False)
            self.measure_tab_init.run_measurement_button.setDisabled(False)
            self.choose_folder_button.setDisabled(False)

    def update_input_visibility(self):
        """
        Update the visibility of the input fields based on the selected tab.
        """
        if self.main.folder_name != "":
            self.comments_button.setDisabled(False)
            self.create_datasheet_button.setDisabled(False)
        else:
            self.comments_button.setDisabled(True)
            self.create_datasheet_button.setDisabled(True)

        if self.tabs.currentWidget() == self.general_tab:
            self.create_datasheet_button.hide()
            self.open_fiber_data_button.hide()
            self.folder_name_input.setReadOnly(False)
            self.folder_name_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.recent_folders_label.hide()
            self.recent_folders_combo.hide()
            self.choose_folder_button.hide()
            self.folder_name_label.setText("Folder Name:")
            self.folder_name_input.setText("")
            if hasattr(self.main, 'placeholder_spacer'):
                self.main.layout.removeItem(self.main.placeholder_spacer)
                del self.main.placeholder_spacer
                self.main.insert_spacer(82)
            else:
                self.main.insert_spacer(82)
            self.general_tab_init.update_general_tab_buttons()  # Ensure buttons are correctly updated

        elif self.tabs.currentWidget() == self.camera_tab:
            self.create_datasheet_button.hide()
            self.open_fiber_data_button.hide()
            self.folder_name_input.setReadOnly(False)
            self.folder_name_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.folder_name_input.setText("Manual_Images")
            self.folder_name_label.setText("Folder Name:")

            self.camera_tab_init.update_camera_tab_buttons()  # Ensure buttons are correctly updated

        else:
            if hasattr(self.main, 'placeholder_spacer'):
                self.main.layout.removeItem(self.main.placeholder_spacer)
                del self.main.placeholder_spacer
                self.main.layout.update()

            self.create_datasheet_button.show()
            self.folder_name_label.setText("Fiber Name:")
            self.folder_name_input.setReadOnly(True)
            self.folder_name_input.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.folder_name_input.setText(self.main.folder_name)

            # Show the buttons that were hidden for some functions in the general tab
            self.folder_name_label.show()
            self.folder_name_input.show()
            self.choose_folder_button.show()
            self.open_fiber_data_button.show()
            self.comments_button.show()
            self.recent_folders_combo.show()
            self.working_dir_label.show()
            self.recent_folders_label.show()

    def update_run_button_state(self):
        if self.tabs.currentWidget() == self.general_tab:
            return

        if self.tabs.currentWidget() == self.camera_tab:
            if self.folder_name_input.text() == "":
                self.camera_tab_init.run_button_ct.setDisabled(True)
            else:
                self.camera_tab_init.run_button_ct.setDisabled(False)
            return

        if self.main.folder_name and self.main.fiber_shape and self.main.fiber_dimension != "":
            if self.analyse_tab_init.analysis_type_combo.currentText() == "Throughput":
                self.analyse_tab_init.run_analysis_button.setDisabled(False)

            elif self.analyse_tab_init.analysis_type_combo.currentText() == "SG" or self.analyse_tab_init.analysis_type_combo.currentText() == "FRD":
                if (self.analyse_tab_init.plot_sg_checkbox.isChecked() or self.analyse_tab_init.calc_sg_checkbox.isChecked()
                    or self.analyse_tab_init.plot_coms_checkbox.isChecked() or self.analyse_tab_init.get_params_checkbox.isChecked()
                    or self.analyse_tab_init.plot_masks_checkbox.isChecked() or self.analyse_tab_init.make_video_checkbox.isChecked()
                    or self.analyse_tab_init.sg_new_checkbox.isChecked() or self.analyse_tab_init.calc_frd_checkbox.isChecked()
                    or self.analyse_tab_init.plot_sutherland_checkbox.isChecked() or self.analyse_tab_init.plot_f_ratio_circles_on_raw_checkbox.isChecked()
                    or self.analyse_tab_init.plot_com_comk_on_image_cut_checkbox.isChecked() or self.analyse_tab_init.plot_nf_horizontal_cut_checkbox.isChecked()


                ):
                    self.analyse_tab_init.run_analysis_button.setDisabled(False)
                else:
                    self.analyse_tab_init.run_analysis_button.setDisabled(True)
            else:
                self.analyse_tab_init.run_analysis_button.setDisabled(True)

        else:
            self.analyse_tab_init.run_analysis_button.setDisabled(True)

    def choose_folder(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Folder")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setDirectory(self.main.base_directory)
        if dialog.exec():
            folder = dialog.selectedFiles()[0]
        else:
            return

        if folder:
            self.working_dir_display.setText(folder)
            self.folder_name_input.setText(os.path.basename(folder))

            self.main.open_fiber_data_window()
            self.main.fiber_data_window.load_fiber_data(folder)

            self.update_comments_button()
            update_recent_folders(folder, self.recent_folders, base_directory=self.main.base_directory)
            self.update_recent_folders_combo()

    def select_recent_folder(self, index):
        if 0 < index <= len(self.recent_folders):
            folder = self.recent_folders[index - 1]
            self.working_dir_display.setText(folder)
            self.folder_name_input.setText(os.path.basename(folder))
            self.main.open_fiber_data_window()
            self.main.fiber_data_window.load_fiber_data(folder)
            self.update_comments_button()

    def update_recent_folders_combo(self):
        self.recent_folders_combo.clear()
        self.recent_folders_combo.addItem("None Selected")
        self.recent_folders_combo.addItems(self.recent_folders)
        self.recent_folders_combo.setCurrentIndex(0)

    def show_message(self, message):
        self.main.message_label.setText(message)

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
            working_dir = os.path.join(self.main.base_directory, fiber_name)
            self.working_dir_display.setText(working_dir)
        else:
            self.working_dir_display.setText("")

    def check_existing_measurements(self, folder):
        measurements = []
        if os.path.exists(os.path.join(folder, "FRD")):
            measurements.append("FRD")
        if os.path.exists(os.path.join(folder, "SG")):
            measurements.append("SG")
        if os.path.exists(os.path.join(folder, "Throughput")):
            measurements.append("Throughput")

        if measurements:
            self.measure_tab_init.existing_measurements_label.setText(f"Measurements already done: {', '.join(measurements)}")
        else:
            self.measure_tab_init.existing_measurements_label.setText("No measurements done yet.")

    def create_datasheet(self):
        working_dir = self.working_dir_display.text()
        fiber_folder = self.folder_name_input.text()
        if working_dir == "":
            self.show_message("Please select a working directory first.")
            return
        if not os.path.exists(working_dir):
            self.show_message("Working directory does not exist.")
            return


        # Use anaconda?
        anaconda = False
        if anaconda:
            command = [
                r"C:\Users\User\anaconda3\Scripts\activate",
                "&&",
                "quarto",
                "render",
                r"fiber_datasheet\fiber_datasheet_template.ipynb",
                "--execute",
                f"-P fiber_folder:'{fiber_folder}'"
            ]
        else:
            command = [
                r""
                "quarto",
                "render",
                r"fiber_datasheet/fiber_datasheet_template.ipynb",
                "--execute",
                f"-P fiber_folder:{fiber_folder}"
            ]

        env = os.environ.copy()
        env["QUARTO_PYTHON"] = "/usr/bin/python3"
        try:
            self.show_message("Creating datasheet...")
            result = subprocess.run(" ".join(command), env=env, shell=True, check=True, capture_output=True, text=True)
            print("Command output:", result.stdout)
            self.show_message("Datasheet created successfully.")
            # Copy the datasheet to the working directory
            datasheet_path = os.path.join("fiber_datasheet", "fiber_datasheet_template.pdf")
            shutil.copyfile(datasheet_path, working_dir + "/Datasheet.pdf")
            self.show_message("Datasheet copied to working directory.")

        except subprocess.CalledProcessError as e:
            print("Error:", e.stderr)
            self.show_message("Error creating datasheet.")

    def stop_general_function(self):
        self.stop_event.set()
        self.progress_signal.emit("Stopping function...")
