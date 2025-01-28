import image_analysation
import image_reduction
import find_f_number
import file_mover
import file_save_managment
from astropy.io import fits
import os
import numpy as np
import subprocess
import time


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

def main_measure(project_folder:str, progress_signal=None, batch_file_path:str="D:\stepper_motor\start_nina_with_fstop.bat"):
    """
    Main function to run the analysis pipeline
    Args:
        project_folder: Path of the project folder.
        measurement_name: Name of the measurement folder.
        batch_file_path: Path of the batch file to start N.I.N.A.

    """
    # Start N.I.N.A. with F-stop analysis sequence
    run_batch_file(batch_file_path)

    # Write progress to file
    file_save_managment.write_progress("Starting N.I.N.A. with F-stop analysis sequence")

    if progress_signal:
        progress_signal.emit("Starting N.I.N.A. with F-stop analysis sequence")

    # Waiting for N.I.N.A. to complete
    flag_file = "D:/stepper_motor/nina_flag.txt" # Flag file created by N.I.N.A. when sequence is complete
    print("Waiting for N.I.N.A. to complete...")

    if progress_signal:
        progress_signal.emit("Waiting for N.I.N.A. to complete...")

    while not os.path.exists(flag_file):
        time.sleep(10)  # Check every 5 seconds
    print("N.I.N.A. completed!")

    if progress_signal:
        progress_signal.emit("N.I.N.A. completed!")

    # Clean up the flag file
    os.remove(flag_file)

    # Move files to project folder, files are initially saved to the default Nina output folder
    file_mover.move_files_and_folders("D:/Vincent/nina_output", project_folder)

    time.sleep(1)
    file_mover.clear_folder("D:/Vincent/nina_output")

    # Close Nina
    run_batch_file("D:\stepper_motor\close_nina.bat")

    # Write progress to file
    file_save_managment.write_progress("N.I.N.A. closed, starting analysis pipeline")

    if progress_signal:
        progress_signal.emit("N.I.N.A. closed, starting analysis pipeline")

    """# Run analysis pipeline
    run_from_existing_files(project_folder, measurement_name)"""

def run_from_existing_files(project_folder:str, progress_signal=None):
    """
    Run the analysis pipeline using existing files in the project folder.
    Args:
        project_folder: Path of the project folder.
    """

    # Write progress to file
    file_save_managment.write_progress("Creating analysis folders")

    # Define folders
    dark_folder = project_folder + "/DARK"  # DARK is the standard Nina output folder name for darks
    light_folder = project_folder + "/LIGHT"  # LIGHT is the standard Nina output folder name for lights
    reduce_images_folder = project_folder + "/REDUCED/"  # Folder for reduced images
    os.makedirs(reduce_images_folder, exist_ok=True)
    measurements_folder = project_folder + "/Measurements/"  # Folder for measurements
    os.makedirs(measurements_folder, exist_ok=True)

    pos_values = [9.9, 5, 0]  # Values of the stepper motor positions (temporary(hopefully))

    if progress_signal:
        progress_signal.emit("Creating master dark frame")

    # Create master dark frame
    m_dark = image_reduction.create_master_dark(dark_folder, plot=False)

    if progress_signal:
        progress_signal.emit("Reducing light frames")

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

    if progress_signal:
        progress_signal.emit("Calculating radii")

    # Radius calculation loop
    radii = image_analysation.calculate_multiple_radii(reduced_data, measurements_folder)

    # Convert to numpy arrays
    radii = np.array(radii)
    pos_values = np.array(pos_values)

    print('radii:', radii, 'pos:', pos_values)

    if progress_signal:
        progress_signal.emit("Calculating F-number")

    # Calculate F-number
    f_number, f_number_err = find_f_number.calculate_f_number(radii, pos_values, plot_regression=False,
                                                              save_path=measurements_folder)
    print(f"Calculated F-number (f/#): {f_number:.3f} ± {f_number_err:.3f}")

if __name__ == "__main__":
    project_folder = r"D:\Vincent\filter2_newcoll"
    run_from_existing_files(project_folder)