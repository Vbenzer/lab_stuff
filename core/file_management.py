"""Utility helpers for managing measurement files and directories.

This module bundles small convenience functions used across the
analysis scripts.  They deal with creation of HDF5 containers, storing
plots, copying directory structures and running batch files.
"""
import os
import shutil
import subprocess
from typing import Callable

import h5py
import numpy as np

def create_new_hdf5(file_path: str) -> None:
    """Create an empty HDF5 file.

    Parameters
    ----------
    file_path : str
        Location where the file should be created.
    """

    h5py.File(file_path, "a")
    print("File created at:", file_path)


def create_hdf5_group(folder_path: str, group_name: str) -> None:
    """Create ``group_name`` within the HDF5 file at ``folder_path``."""

    with h5py.File(folder_path, "w") as f:
        f.create_group(group_name)
    print("Created new group:", group_name, "at:", folder_path)


def add_data_to_hdf(path: str, data, dataset_name: str):
    """Append ``data`` to ``dataset_name`` in the HDF5 file at ``path``."""

    with h5py.File(path, "a") as f:
        dset = f.create_dataset(dataset_name, data)
    print("Wrote data to:", dataset_name)
    return dset


def add_plot_to_hdf(file_path: str, plot_path: str, plot_name: str) -> None:
    """Store an image from ``plot_path`` as ``plot_name`` in ``file_path``."""

    with h5py.File(file_path, "w") as f:
        with open(plot_path, "rb") as img:
            f.create_dataset(plot_name, data=np.frombuffer(img.read(), dtype="uint8"))
    print("Plot:", plot_name, "saved to:", file_path)


def save_measurement_hdf5(
    filename: str, measurement_name: str, f_number, f_number_err
) -> None:
    """Persist measurement values in an HDF5 file."""

    with h5py.File(filename, "a") as file:
        measurement_group = file.create_group(measurement_name)
        measurement_group.create_dataset("f_number", data=f_number)
        measurement_group.create_dataset("f_number_err", data=f_number_err)


def _transfer_items(
    source_folder: str,
    destination_folder: str,
    file_func: Callable[[str, str], None],
    folder_func: Callable[[str, str], None],
    verb: str,
) -> None:
    """Helper used by ``move_files_and_folders`` and ``copy_files_and_folders``.

    Parameters
    ----------
    source_folder:
        Directory from which to read items.
    destination_folder:
        Directory where items will be placed.
    file_func:
        Function used to process files (e.g. ``shutil.move``).
    folder_func:
        Function used to process folders (e.g. ``shutil.copytree``).
    verb:
        Action description used for console output.
    """

    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isfile(source_path):
            print(f"{verb} file: {item}")
            file_func(source_path, destination_path)
        elif os.path.isdir(source_path):
            print(f"{verb} folder: {item}")
            folder_func(source_path, destination_path)

def move_files_and_folders(source_folder: str, destination_folder: str) -> None:
    """Move all items from ``source_folder`` to ``destination_folder``."""

    _transfer_items(
        source_folder,
        destination_folder,
        shutil.move,
        shutil.move,
        "Moving",
    )


def copy_files_and_folders(source_folder: str, destination_folder: str) -> None:
    """Copy all items from ``source_folder`` to ``destination_folder``."""

    _transfer_items(
        source_folder,
        destination_folder,
        shutil.copy2,
        shutil.copytree,
        "Copying",
    )


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

def _copy_items(source_folder: str, destination_folder: str) -> None:
    """Recursively copy items from ``source_folder`` to ``destination_folder``."""

    # Iterate over each file or directory in the source folder and copy it to
    # the destination.  Existing files are overwritten, while existing folders
    # are merged recursively.

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
                _copy_items(source_path, destination_path)


def _remove_extra_items(source_folder: str, destination_folder: str) -> None:
    """Remove items from ``destination_folder`` that do not exist in ``source_folder``."""

    # Walk destination folder and delete files or folders that have no
    # counterpart in the source.  This keeps ``destination_folder``
    # synchronized with ``source_folder`` when copying.

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


def synchronize_directories(source_folder: str, destination_folder: str) -> None:
    """Synchronize contents of ``destination_folder`` with ``source_folder``."""

    # Ensure destination exists before starting the synchronization. All new
    # or changed files are copied, and obsolete files are removed so that both
    # directories mirror each other.

    os.makedirs(destination_folder, exist_ok=True)
    _copy_items(source_folder, destination_folder)
    _remove_extra_items(source_folder, destination_folder)

def run_batch_file(batch_file_path: str) -> None:
    """Execute ``batch_file_path`` using ``subprocess.run``."""
    try:
        # Use subprocess to run the batch file
        result = subprocess.run(batch_file_path, shell=True, check=True, text=True)
        print(f"Batch file executed successfully with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the batch file: {e}")

def load_frd_calibration_data(folder_path: str):
    """Load FRD calibration data located in ``folder_path``.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing ``f_number.json``.

    Returns
    -------
    tuple[list | None, list | None]
        ``f_number`` and ``f_number_err`` arrays if the file was found,
        otherwise ``(None, None)``.
    """
    import json
    calibration_data = {}
    calibration_file = "f_number.json"
    calibration_path = os.path.join(folder_path, calibration_file)
    if not os.path.exists(calibration_path):
        print(f"Calibration file {calibration_file} not found in {folder_path}.")
        return None
    try:
        with open(calibration_path, 'r') as file:
            calibration_data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {calibration_file}: {e}")
        return None

    f_number = calibration_data.get("f_number")
    f_number_err = calibration_data.get("f_number_err")

    return f_number, f_number_err


if __name__ == "__main__":
    calibration_folder = r"/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/setup_F#_EE_98_Measurement_3"
    f_number, f_number_err = load_frd_calibration_data(calibration_folder)
    print("F-number:", f_number)
    print("F-number error:", f_number_err)
