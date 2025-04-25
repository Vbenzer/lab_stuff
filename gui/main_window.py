import socket
import threading

import serial.tools
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QUrl, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

import os, sys, json, subprocess, time

from core.hardware.filter_wheel_fratio import FilterWheel

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

def save_recent_folders(recent_folders:str, file_path:str):
    """
    Save the recent folders to a JSON file.
    Args:
        recent_folders: Folder names to save.
        file_path: Path of the JSON file to save to.
    """
    with open(file_path, 'w') as file:
        json.dump(recent_folders, file)

class MainWindowInit(HelperFunctions, Widgets):
    def __init__(self, main_ctrl):
        self.main = main_ctrl
        self.main_init = self  # Define reference to self as main_init

        # Create log file
        self.create_log_file()
        self.log_data("MainWindowInit initialized.")  # Log initialization

        self.log_data(f"Base directory: {self.main.base_directory}")  # Log base directory
        self.log_data(f"OS name: {self.main.os_name}")  # Log OS name

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
        dialog = QFileDialog(self.main)
        dialog.setWindowTitle("Select Folder")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setDirectory(self.main.base_directory)
        if dialog.exec():
            folder = dialog.selectedFiles()[0]
        else:
            return

        if folder:
            self.log_data(f"Folder chosen: {folder}")  # Log folder selection
            self.working_dir_display.setText(folder)
            self.folder_name_input.setText(os.path.basename(folder))

            self.main.open_fiber_data_window()
            self.main.fiber_data_window.load_fiber_data(folder)

            self.update_comments_button()
            update_recent_folders(folder, self.recent_folders, base_directory=self.main.base_directory)
            self.update_recent_folders_combo()
            self.log_data(f"Recent folders updated with: {folder}")  # Log recent folder update

    def select_recent_folder(self, index):
        if 0 < index <= len(self.recent_folders):
            folder = self.recent_folders[index - 1]
            self.log_data(f"Recent folder selected: {folder}")  # Log recent folder selection
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

    def create_log_file(self):
        # Create log file in log folder to log all actions
        log_folder = os.path.join(self.main.base_directory, "logs")
        os.makedirs(log_folder, exist_ok=True)
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_name = f"log_{current_time}.txt"
        self.log_name = log_name
        log_file_path = os.path.join(log_folder, log_name)
        # Create the log file
        with open(log_file_path, "w") as log_file:
            log_file.write(f"{current_time}: Log file created.\n")

    def log_data(self, message):
        # Append the message to the log file
        log_folder = os.path.join(self.main.base_directory, "logs")
        current_time = time.strftime("%H-%M-%S")
        log_file_path = os.path.join(log_folder, self.log_name)
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{current_time}: {message}\n")

    def create_datasheet(self):
        working_dir = self.working_dir_display.text()
        fiber_folder = self.folder_name_input.text()
        if working_dir == "":
            self.log_data("Datasheet creation failed: No working directory selected.")  # Log failure
            self.show_message("Please select a working directory first.")
            return
        if not os.path.exists(working_dir):
            self.log_data("Datasheet creation failed: Working directory does not exist.")  # Log failure
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
            self.log_data("Datasheet creation started.")  # Log start
            self.show_message("Creating datasheet...")
            result = subprocess.run(" ".join(command), env=env, shell=True, check=True, capture_output=True, text=True)
            print("Command output:", result.stdout)
            self.log_data("Datasheet created successfully.")  # Log success
            self.show_message("Datasheet created successfully.")
            # Copy the datasheet to the working directory
            datasheet_path = os.path.join("fiber_datasheet", "fiber_datasheet_template.pdf")
            shutil.copyfile(datasheet_path, working_dir + "/Datasheet.pdf")
            self.show_message("Datasheet copied to working directory.")

        except subprocess.CalledProcessError as e:
            self.log_data(f"Datasheet creation failed: {e.stderr}")  # Log error
            print("Error:", e.stderr)
            self.show_message("Error creating datasheet.")

    def update_progress(self, message):
        if not self.progress_text_edit.isVisible():
            self.progress_text_edit.show()
        self.progress_text_edit.append(message)

        self.log_data(f"Progress updated: {message}")  # Log progress updates

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
            self.log_data(f"Working directory updated: {working_dir}")  # Log working directory update
            self.working_dir_display.setText(working_dir)
        else:
            self.log_data("Working directory cleared.")  # Log clearing of working directory
            self.working_dir_display.setText("")

    def check_existing_measurements(self, folder):
        measurements = []
        if os.path.exists(os.path.join(folder, "FRD")):
            measurements.append("FRD")
        if os.path.exists(os.path.join(folder, "SG")):
            measurements.append("SG")
        if os.path.exists(os.path.join(folder, "Throughput")):
            measurements.append("Throughput")

        self.log_data(f"Existing measurements checked: {', '.join(measurements) if measurements else 'None'}")  # Log measurements

        if measurements:
            self.measure_tab_init.existing_measurements_label.setText(f"Measurements already done: {', '.join(measurements)}")
        else:
            self.measure_tab_init.existing_measurements_label.setText("No measurements done yet.")

    def stop_general_function(self):
        self.stop_event.set()
        self.progress_signal.emit("Stopping function...")



class MainWindow(QMainWindow, HelperFunctions, Widgets):
    progress_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.stop_event = threading.Event()
        self.choose_base_path()
        #print(self.base_directory)
        self.inputs_locked = False
        self.experiment_running = False

        self.setWindowTitle("Fiber Measurement and Analysis")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.fiber_diameter = ""
        self.fiber_width = ""
        self.fiber_height = ""
        self.fiber_shape = ""
        self.folder_name = ""
        self.fiber_dimension = ""

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: red; font-weight: bold;")

        self.filter_wheel_ready = False
        self.filter_wheel_initiated = False

        self.main_window_init = MainWindowInit(self)

        self.fiber_data_window = FiberDataWindow(self)

    def choose_base_path(self):
        if sys.platform.startswith("linux"):
            os_name = "Linux"
            base_path = r"/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements"
        elif sys.platform.startswith("win"):
            os_name = "Windows"
            hostname = socket.gethostname()
            if hostname == "DESKTOP-HEBN59N":
                base_path = r"D:\Vincent"
            else:
                base_path = r"\\srv4\labshare\raw_data\fibers\Measurements"
        else:
            raise OSError("Unsupported OS")
        self.base_directory = base_path
        self.os_name = os_name

    def initialize_filter_wheel(self):
        import serial.tools.list_ports
        available_ports = [port.device for port in serial.tools.list_ports.comports()]
        if 'COM5' in available_ports:
            self.main_window_init.log_data("Filter wheel initialization started.")  # Log initialization start
            self.filter_wheel_initiated = True
            self.filter_wheel = FilterWheel('COM5')
            self.filter_wheel_ready = True
            self.main_window_init.log_data("Filter wheel initialized successfully.")  # Log success
            self.main_window_init.general_tab_init.update_general_tab_buttons()
        else:
            self.log_data("Filter wheel initialization failed: COM5 not available.")  # Log failure
            self.main_window_init.show_message("COM5 is not available.")

    def update_fiber_shape_inputs(self):
        fiber_shape = self.fiber_shape_combo.currentText()
        if fiber_shape == "rectangular":
            self.fiber_diameter_label.hide()
            self.fiber_diameter_input.hide()
            self.fiber_width_label.show()
            self.fiber_width_input.show()
            self.fiber_height_label.show()
            self.fiber_height_input.show()
        else:
            self.fiber_diameter_label.show()
            self.fiber_diameter_input.show()
            self.fiber_width_label.hide()
            self.fiber_width_input.hide()
            self.fiber_height_label.hide()
            self.fiber_height_input.hide()

    def choose_calibration_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Calibration Folder",
                                                       self.base_directory + "/Calibration")
        if folder_path:
            self.calibration_folder_input.setText(folder_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

