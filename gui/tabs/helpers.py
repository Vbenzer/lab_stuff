from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
import os, json, sys

class HelperFunctions:
    def create_hbox_layout(self, label, widget):
        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(widget)
        return hbox

    def insert_spacer(self, height):
        self.placeholder_spacer = QSpacerItem(20, height)
        self.layout.insertItem(self.layout.count() - 1, self.placeholder_spacer)
        self.layout.update()

    def check_valid_exp_input(self):
        if not self.exposure_time_input.hasAcceptableInput():
            self.main_init.show_message("Invalid exposure time input. Please enter a valid exposure time. E.g.: 1ms, 1s, 1us")
        else:
            self.main_init.show_message("")

    def access_comments_file(self):
        working_dir = self.working_dir_display.text()
        if working_dir == "":
            self.show_message("Please select a working directory first.")
            return

        comments_file_path = os.path.join(working_dir, "comments.txt")
        if not os.path.exists(comments_file_path):
            os.makedirs(working_dir, exist_ok=True)
            with open(comments_file_path, "w") as file:
                file.write("Comments:\n")
            self.comments_button.setText("Access Comments")
        else:
            if sys.platform.startswith("win"):
                os.startfile(comments_file_path)
            elif sys.platform.startswith("linux"):
                subprocess.call(["xdg-open", comments_file_path])

    def update_comments_button(self):
        working_dir = self.working_dir_display.text()
        comments_file_path = os.path.join(working_dir, "comments.txt")
        if os.path.exists(comments_file_path) and os.path.getsize(comments_file_path) > 0:
            self.comments_button.setText("Access Comments")
        else:
            self.comments_button.setText("Add Comments")
