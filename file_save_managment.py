import h5py
import numpy as np




def create_new_hdf5(file_path:str):
    # Create a new HDF5 file
    h5py.File(file_path, 'a')
    print("File created at:", file_path)

def create_hdf5_group(folder_path:str, group_name:str):
    with h5py.File(folder_path,"w") as f:
        f.create_group(group_name)
    print("Created new group:", group_name, "at:", folder_path)

def add_data_to_hdf(path:str, data, dataset_name:str):
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data)
    # Attributes potentially
    print("Wrote data to:", dataset_name)

def add_plot_to_hdf(file_path:str, plot_path:str, plot_name:str):
    with h5py.File(file_path, "w") as f:
        with open(plot_path, "rb") as img:
            f.create_dataset(plot_name, data=np.frombuffer(img.read(), dtype="uint8"))
    print("Plot:", plot_name, "saved to:", file_path)


if __name__ == "__main__":
    hdf_file="test"
    create_new_hdf5(hdf_file)



