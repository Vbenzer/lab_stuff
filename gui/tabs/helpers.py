from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTabWidget, QFileDialog, QCheckBox, QTextEdit, QSpacerItem,
                             QSizePolicy, QDialog, QVBoxLayout, QMessageBox
                             )
import os, json, sys, subprocess



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


def update_recent_folders(folder:str, recent_folders:list[str], file_path:str, max_recent=3):
    """
    Update the list of recent folders.
    Args:
        folder: New folder to add.
        recent_folders: List of recent folders.
        file_path: Path of the JSON file to save to.
        max_recent: Maximum number of recent folders to keep.
    """
    if folder in recent_folders:
        recent_folders.remove(folder)
    recent_folders.insert(0, folder)
    if len(recent_folders) > max_recent:
        recent_folders.pop()
    save_recent_folders(recent_folders, file_path=file_path)


def save_recent_folders(recent_folders:str, file_path:str):
    """
    Save the recent folders to a JSON file.
    Args:
        recent_folders: Folder names to save.
        file_path: Path of the JSON file to save to.
    """
    with open(file_path, 'w') as file:
        json.dump(recent_folders, file)
