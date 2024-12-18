import image_analysation
import image_reduction
import find_f_number
import file_mover
from astropy.io import fits
import os
import numpy as np
import subprocess
import time
import h5py

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


def main(project_folder:str, measurement_name:str):
    """
    Main function to run the analysis pipeline
    Args:
        project_folder: Path of the project folder.

    """
    # Start N.I.N.A. with F-stop analysis sequence
    run_batch_file("D:\stepper_motor\start_nina_with_fstop.bat")

    # Waiting for N.I.N.A. to complete
    flag_file = "D:/stepper_motor/nina_flag.txt" # Flag file created by N.I.N.A. when sequence is complete
    print("Waiting for N.I.N.A. to complete...")

    while not os.path.exists(flag_file):
        time.sleep(10)  # Check every 5 seconds
    print("N.I.N.A. completed!")

    # Clean up the flag file
    os.remove(flag_file)

    # Move files to project folder, files are initially saved to the default Nina output folder
    file_mover.move_files_and_folders("D:/Vincent/nina_output", project_folder)

    time.sleep(1)
    file_mover.clear_folder("D:/Vincent/nina_output")

    # Close Nina
    run_batch_file("D:\stepper_motor\close_nina.bat")

    # Run analysis pipeline
    run_from_existing_files(project_folder, measurement_name)

def run_from_existing_files(project_folder:str, measurement_name:str):
    """
    Run the analysis pipeline using existing files in the project folder.
    Args:
        project_folder: Path of the project folder.
    """
    # Define folders
    dark_folder = project_folder + "/DARK"  # DARK is the standard Nina output folder name for darks
    light_folder = project_folder + "/LIGHT"  # LIGHT is the standard Nina output folder name for lights
    reduce_images_folder = project_folder + "/REDUCED/"  # Folder for reduced images
    os.makedirs(reduce_images_folder, exist_ok=True)
    #with h5py.File("D:/Vincent/" + "measurements.h5", "a") as f:
    #    f.create_group(measurement_name)
    measurements_folder = project_folder + "/Measurements/"  # Folder for measurements
    os.makedirs(measurements_folder, exist_ok=True)

    pos_values = [9.9, 5, 0]  # Values of the stepper motor positions (temporary(hopefully))

    # Create master dark frame
    m_dark = image_reduction.create_master_dark(dark_folder, plot=False)

    # Light frame reduction loop
    reduced_data = []
    for file_name in os.listdir(light_folder):
        if file_name.endswith(".fits"):  # Only process FITS files
            file_path = os.path.join(light_folder, file_name)
            with fits.open(file_path) as hdul:
                light_frame = hdul[0].data.astype(np.float32)  # Convert to float for precision

            output_file = os.path.join(reduce_images_folder, os.path.splitext(file_name)[0] + "_reduced.fits")
            red_file = image_reduction.reduce_image_with_dark(light_frame, m_dark, output_file, save=True)
            reduced_data.append(red_file)

    # Radius calculation loop
    radii = image_analysation.calculate_multiple_radii(reduced_data, measurements_folder)

    # Convert to numpy arrays
    radii = np.array(radii)
    pos_values = np.array(pos_values)

    print('radii:', radii, 'pos:', pos_values)

    # Calculate F-number
    f_number, f_number_err = find_f_number.calculate_f_number(radii, pos_values, plot_regression=True,
                                                              save_path=measurements_folder)
    print(f"Calculated F-number (f/#): {f_number:.3f} ± {f_number_err:.3f}")
    """
    # Save everything to HDF5 file
    with h5py.File("D:/Vincent/" + "measurements.h5", "a") as f:
        group = f[measurement_name]
        group.create_dataset("radii", data=radii)
        group.create_dataset("pos_values", data=pos_values)
        group.create_dataset("f_number", data=f_number)
        group.create_dataset("f_number_err", data=f_number_err)

        # Save plots to HDF5 file
        with open(measurements_folder + "regression_plot.png", "rb") as plot_file:
            group.create_dataset("regression_plot", data=plot_file.read())
        for file_name in os.listdir(measurements_folder + "Radius/"):
            if file_name.endswith(".png"):
                with open(measurements_folder + f"Radius/{file_name}", "rb") as plot_file:
                    group.create_dataset(f"radius_plot_{file_name}", data=plot_file.read())
    """

if __name__ == "__main__":
    project_folder = r"D:\Vincent\filter2_newcoll"
    run_from_existing_files(project_folder)