import os
import shutil
import subprocess
import time
import h5py
import numpy as np


def write_progress(step:str):
    """
    Writes the progress of the experiment to a file to be read by the GUI.
    Args:
        step: Description of the current step.
    """
    with open("progress.txt", "a") as f:
        f.write(f"{step}\n")


def progress_file_remove():
    """
    Removes the progress file after the experiment is complete
    """
    with open("progress.txt", "a") as f:
        f.write("Experiment Complete\n")
    time.sleep(2)
    if os.path.exists("progress.txt"):
        os.remove("progress.txt")


def create_new_hdf5(file_path:str):
    # Create a new HDF5 file
    h5py.File(file_path, 'a')
    print("File created at:", file_path)


def create_hdf5_group(folder_path:str, group_name:str):
    with h5py.File(folder_path,"w") as f:
        f.create_group(group_name)
    print("Created new group:", group_name, "at:", folder_path)


def add_data_to_hdf(path:str, data, dataset_name:str):
    with h5py.File(path, "a") as f:
        dset = f.create_dataset(dataset_name, data)
    print("Wrote data to:", dataset_name)
    return dset


def add_plot_to_hdf(file_path:str, plot_path:str, plot_name:str):
    with h5py.File(file_path, "w") as f:
        with open(plot_path, "rb") as img:
            f.create_dataset(plot_name, data=np.frombuffer(img.read(), dtype="uint8"))
    print("Plot:", plot_name, "saved to:", file_path)


def save_measurement_hdf5(filename, measurement_name, f_number, f_number_err):
    with h5py.File(filename, 'a') as file:
        # Create a group for each measurement ID
        measurement_group = file.create_group(measurement_name)

        # Save the two arrays under the measurement group
        measurement_group.create_dataset('f_number', data=f_number)
        measurement_group.create_dataset('f_number_err', data=f_number_err)

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
    hdf_file = "test.hdf5"
    test_data = np.array([1, 2, 3, 4, 5])
    create_new_hdf5(hdf_file)
    create_hdf5_group(hdf_file, "test_group")
    add_data_to_hdf(hdf_file, test_data, "test_data")

    r'''source_folder = r"D:\Vincent"
    destination_folder = r"\\srv4\labshare\raw_data\fibers\Measurements"

    # Ensure the destination folder exists
    # os.makedirs(destination_folder, exist_ok=True)
    # copy_files_and_folders(source_folder, destination_folder)
    synchronize_directories(source_folder, destination_folder)'''


def run_batch_file(batch_file_path:str):
    """
    Runs a batch file using subprocess
    Args:
        batch_file_path: Path of the file.
    """
    try:
        # Use subprocess to run the batch file
        result = subprocess.run(batch_file_path, shell=True, check=True, text=True)
        print(f"Batch file executed successfully with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the batch file: {e}")
