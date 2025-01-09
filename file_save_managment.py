import h5py
import numpy as np
import os
import time

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

if __name__ == "__main__":
    hdf_file="test.hdf5"
    create_new_hdf5(hdf_file)
    create_hdf5_group(hdf_file, "test_group")
    test_data=np.array([1,2,3,4,5])
    add_data_to_hdf(hdf_file, test_data, "test_data")




