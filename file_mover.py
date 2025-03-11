import os
import shutil

def move_files_and_folders(source_folder:str, destination_folder:str):
    """
    Move all files and subfolders from the source folder to the destination folder.
    Args:
        source_folder: Path of the source folder.
        destination_folder: Path of the destination folder.

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

def copy_files_and_folders(source_folder:str, destination_folder:str):
    """
    Copy all files and subfolders from the source folder to the destination folder.
    Args:
        source_folder: Path of the source folder.
        destination_folder: Path of the destination folder.

    """
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isfile(source_path):
            # Copy file
            print(f"Copying file: {item}")
            shutil.copy(source_path, destination_path)
        elif os.path.isdir(source_path):
            # Copy folder
            print(f"Copying folder: {item}")
            shutil.copytree(source_path, destination_path)

def clear_folder(folder_path):
    """
    Clear all files and subfolders from the specified folder.
    Args:
        folder_path: Path of the folder to clear.

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

def synchronize_directories(source_folder: str, destination_folder: str):
    """
    Synchronize the files and folders between the source and destination folders.
    Args:
        source_folder: Path of the source folder.
        destination_folder: Path of the destination folder.
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Copy files and folders from source to destination
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isfile(source_path):
            if not os.path.exists(destination_path) or not os.path.samefile(source_path, destination_path):
                print(f"Copying file: {item}")
                shutil.copy2(source_path, destination_path)
        elif os.path.isdir(source_path):
            if not os.path.exists(destination_path):
                print(f"Copying folder: {item}")
                shutil.copytree(source_path, destination_path)
            else:
                synchronize_directories(source_path, destination_path)

    # Remove files and folders from destination that are not in source
    for item in os.listdir(destination_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if not os.path.exists(source_path):
            if os.path.isfile(destination_path) or os.path.islink(destination_path):
                print(f"Deleting file: {item}")
                os.remove(destination_path)
            elif os.path.isdir(destination_path):
                print(f"Deleting folder: {item}")
                shutil.rmtree(destination_path)

if __name__ == "__main__":
    source_folder = r"D:\Vincent"
    destination_folder = r"\\srv4\labshare\raw_data\fibers\Measurements"

    # Ensure the destination folder exists
    #os.makedirs(destination_folder, exist_ok=True)
    #copy_files_and_folders(source_folder, destination_folder)
    synchronize_directories(source_folder, destination_folder)
