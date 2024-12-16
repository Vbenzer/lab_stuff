import os
import shutil

def move_files_and_folders(source_folder:str, destination_folder:str):
    """
    Move all files and folders from the source folder to the destination folder.
    """
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isfile(source_path):
            # Move file
            print(f"Moving file: {item}")
            shutil.move(source_path, destination_path)
        elif os.path.isdir(source_path):
            # Move folder
            print(f"Moving folder: {item}")
            shutil.move(source_path, destination_path)

def clear_folder(folder_path):
    """
    Remove all files and subfolders in the specified folder.
    """
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            # Remove files or symbolic links
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            # Remove directories
            shutil.rmtree(item_path)
            print(f"Deleted folder: {item_path}")

if __name__ == "__main__":
    source_folder = r"D:\stepper_motor\test_images\sequence_stepper_filter_fstop_analysis"
    destination_folder = r"D:\Vincent\f_stop_analysis"

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    move_files_and_folders(source_folder, destination_folder)
