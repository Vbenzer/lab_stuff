from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QUrl, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

import os, json

class FiberDataWindow(QDialog):
    fiberDataChanged = pyqtSignal(str, object, str)  # Define a signal

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fiber Data")
        self.setGeometry(150, 150, 400, 300)
        self.layout = QVBoxLayout(self)

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: red; font-weight: bold;")
        self.layout.addWidget(self.message_label)

        self.fiber_name_label = QLabel("Fiber Name:")
        self.fiber_name_input = QLineEdit()
        self.fiber_name_input.textChanged.connect(lambda: self.check_folder_exists())
        self.fiber_name_input.setFixedWidth(300)

        self.fiber_diameter_label = QLabel("Fiber Diameter (µm):")
        self.fiber_diameter_input = QLineEdit()
        self.fiber_diameter_input.setFixedWidth(300)

        self.fiber_width_label = QLabel("Fiber Width (µm):")
        self.fiber_width_input = QLineEdit()
        self.fiber_width_input.setFixedWidth(300)
        self.fiber_width_label.hide()
        self.fiber_width_input.hide()

        self.fiber_height_label = QLabel("Fiber Height (µm):")
        self.fiber_height_input = QLineEdit()
        self.fiber_height_input.setFixedWidth(300)
        self.fiber_height_label.hide()
        self.fiber_height_input.hide()

        self.fiber_length_label = QLabel("Fiber Length (m):")
        self.fiber_length_input = QLineEdit()
        self.fiber_length_input.setFixedWidth(300)

        self.fiber_shape_label = QLabel("Fiber Shape:")
        self.fiber_shape_combo = QComboBox()
        self.fiber_shape_combo.addItems(["None", "circular", "octagonal", "rectangular"])
        self.fiber_shape_combo.setFixedWidth(300)
        self.fiber_shape_combo.currentIndexChanged.connect(self.update_fiber_shape_inputs)

        self.numerical_aperature_label = QLabel("Numerical Aperature:")
        self.numerical_aperature_input = QLineEdit()
        self.numerical_aperature_input.setFixedWidth(300)

        self.coating_type_label = QLabel("Coating Type:")
        self.coating_type_combo = QComboBox()
        self.coating_type_combo.setEditable(True)
        self.coating_type_combo.addItems(["Polyimide", "Acrylate"])  # Add your predefined options here
        self.coating_type_combo.setFixedWidth(300)

        self.manufacturer_label = QLabel("Manufacturer:")
        self.manufacturer_combo = QComboBox()
        self.manufacturer_combo.setEditable(True)
        self.manufacturer_combo.addItems(["Thorlabs", "Option 2", "Option 3", "Option 4"])  # Add your predefined options here
        self.manufacturer_combo.setFixedWidth(300)

        self.save_button = QPushButton("Save and close")
        self.save_button.clicked.connect(self.check_inputs_and_save)

        self.layout.addWidget(self.fiber_name_label)
        self.layout.addWidget(self.fiber_name_input)
        self.layout.addWidget(self.fiber_shape_label)
        self.layout.addWidget(self.fiber_shape_combo)
        self.layout.addWidget(self.fiber_diameter_label)
        self.layout.addWidget(self.fiber_diameter_input)
        self.layout.addWidget(self.fiber_width_label)
        self.layout.addWidget(self.fiber_width_input)
        self.layout.addWidget(self.fiber_height_label)
        self.layout.addWidget(self.fiber_height_input)
        self.layout.addWidget(self.fiber_length_label)
        self.layout.addWidget(self.fiber_length_input)
        self.layout.addWidget(self.numerical_aperature_label)
        self.layout.addWidget(self.numerical_aperature_input)
        self.layout.addWidget(self.coating_type_label)
        self.layout.addWidget(self.coating_type_combo)
        self.layout.addWidget(self.manufacturer_label)
        self.layout.addWidget(self.manufacturer_combo)

        self.layout.addWidget(self.save_button)

        self.fiber_dimension = ""
        self.fiber_shape = ""
        self.fiber_length = ""
        self.fiber_name = ""
        self.numerical_aperature = ""
        self.coating_type = ""
        self.manufacturer = ""

        self.update_from_load_token = False

    def update_window(self):
        self.show_message("")
        self.fiber_name_input.setText(self.fiber_name)
        self.fiber_length_input.setText(self.fiber_length)
        self.fiber_shape_combo.setCurrentText(self.fiber_shape)
        self.numerical_aperature_input.setText(self.numerical_aperature)
        self.coating_type_combo.setCurrentText(self.coating_type)
        self.manufacturer_combo.setCurrentText(self.manufacturer)
        self.update_fiber_shape_inputs()
        if self.fiber_shape == "rectangular":
            self.fiber_width_input.setText(str(self.fiber_dimension[0]))
            self.fiber_height_input.setText(str(self.fiber_dimension[1]))
        else:
            self.fiber_diameter_input.setText(str(self.fiber_dimension))
        self.check_folder_exists()

    def save_fiber_data(self, folder):
        file_path = os.path.join(folder, "fiber_data.json")
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file_path, "w") as file:
            # noinspection PyTypeChecker
            json.dump({"fiber_name": os.path.basename(folder), "fiber_dimension": self.fiber_dimension, "fiber_shape": self.fiber_shape,
                       "fiber_length": self.fiber_length, "numerical_aperature": self.numerical_aperature, "coating_type": self.coating_type,
                          "manufacturer": self.manufacturer}, file)

    def load_fiber_data(self, folder):
        file_path = os.path.join(folder, "fiber_data.json")
        self.update_from_load_token = True
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
                fiber_dimension = data.get("fiber_dimension", "")
                fiber_shape = data.get("fiber_shape", "")
                fiber_length = data.get("fiber_length", "")
                numerical_aperature = data.get("numerical_aperature", "")
                coating_type = data.get("coating_type", "")
                manufacturer = data.get("manufacturer", "")

                self.fiber_shape = fiber_shape if fiber_shape else ""
                self.fiber_dimension = fiber_dimension if fiber_dimension else ""
                self.fiber_length = fiber_length if fiber_length else ""
                self.numerical_aperature = numerical_aperature if numerical_aperature else ""
                self.coating_type = coating_type if coating_type else ""
                self.manufacturer = manufacturer if manufacturer else ""
        else:
            self.show_message("No fiber data file found.")

        self.fiber_name_input.setText(os.path.basename(folder))
        self.fiber_name = os.path.basename(folder)
        self.fiber_shape_combo.setCurrentText(self.fiber_shape)
        self.update_fiber_shape_inputs()
        if self.fiber_shape == "rectangular":
            if self.fiber_dimension == "":
                self.fiber_width_input.setText("")
                self.fiber_height_input.setText("")
            else:
                self.fiber_width_input.setText(str(self.fiber_dimension[0]))
                self.fiber_height_input.setText(str(self.fiber_dimension[1]))
        else:
            self.fiber_diameter_input.setText(str(self.fiber_dimension))

        self.fiber_length_input.setText(self.fiber_length)
        self.numerical_aperature_input.setText(self.numerical_aperature)
        self.coating_type_combo.setCurrentText(self.coating_type)
        self.manufacturer_combo.setCurrentText(self.manufacturer)

    def check_folder_exists(self, folder=None):
        if folder is None:
            folder = os.path.join(self.parent().base_directory, self.fiber_name_input.text())
        if os.path.exists(folder) and not self.update_from_load_token:
            self.show_message("Folder already exists, contents may be overwritten.")
            return True
        else:
            self.show_message("")
            return False

    def check_inputs_and_save(self):
        if self.fiber_diameter_input.text() != "":
            self.fiber_dimension = self.fiber_diameter_input.text()
        elif self.fiber_width_input.text() != "" and self.fiber_height_input.text() != "":
            self.fiber_dimension = (self.fiber_width_input.text(), self.fiber_height_input.text())
        else:
            self.show_message("Please enter fiber diameter or height and width.")
            return

        if (self.fiber_name_input.text() != ""
                and self.fiber_shape_combo.currentText() != "None"):
            folder = os.path.join(self.parent().base_directory, self.fiber_name_input.text())

            if self.check_folder_exists(folder): #and not self.update_from_load_token:
                reply = QMessageBox.question(self, 'Folder Exists',
                                             'Folder already exists. Do you want to overwrite the contents?',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return
            self.update_from_load_token = False
            self.fiber_shape = self.fiber_shape_combo.currentText()
            self.fiber_length = self.fiber_length_input.text()
            self.fiber_name = self.fiber_name_input.text()
            self.numerical_aperature = self.numerical_aperature_input.text()
            self.coating_type = self.coating_type_combo.currentText()
            self.manufacturer = self.manufacturer_combo.currentText()

            self.emit_fiber_data_changed()

            self.save_fiber_data(folder)
            self.close()
        else:
            self.show_message("Please enter fiber name and shape.")

    def show_message(self, message):
        self.message_label.setText(message)

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

    @pyqtSlot(str, object, str)
    def emit_fiber_data_changed(self):
        self.fiberDataChanged.emit(
            self.fiber_name_input.text(),
            self.fiber_dimension,
            self.fiber_shape_combo.currentText()
        )

class Widgets:
    def open_fiber_data_window(self):
        self.fiber_data_window.fiberDataChanged.connect(self.update_fiber_data)
        self.fiber_data_window.update_window()
        self.fiber_data_window.show()

    @pyqtSlot(str, object, str)
    def update_fiber_data(self, name, dimension, shape):
        self.folder_name = name
        self.fiber_dimension = dimension
        self.fiber_shape = shape

        self.main_window_init.folder_name_input.setText(name)
        self.main_window_init.update_input_visibility()
        self.main_window_init.measure_tab_init.update_measurement_button_state()
        folder = os.path.join(self.base_directory, name)
        self.main_window_init.check_existing_measurements(folder)

    def save_fiber_data(self, folder, fiber_diameter, fiber_shape, fiber_length):
        file_path = os.path.join(folder, "fiber_data.json")
        with open(file_path, "w") as file:
            # noinspection PyTypeChecker
            json.dump({"fiber_diameter": fiber_diameter, "fiber_shape": fiber_shape,
                       "fiber_length": fiber_length}, file)
