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
import json
from scipy.stats import linregress
import matplotlib.pyplot as plt

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

def plot_cones(project_folder:str):
    """
    Plot the light cones (upper half) of the output light rays of each f-number.
    Args:
        project_folder: Path of the project folder.

    Returns:

    """
    pos_values = [9.9, 5, 0]  # Values of the stepper motor positions
    dist_to_chip = []
    slopes = []
    intercepts = []
    radii_all = []
    for i in range(2,7):
        # Get radii from json
        radii = []
        for n in range (0, 3):
            with open(project_folder + f"/filter_{i}/Measurements/Radius/datapoint{n}radius.json") as f:
                params = json.load(f)

            radius = params["radius"]
            radii.append(radius)

        radii = np.array(radii)
        pos_values = np.array(pos_values)

        # Sort to descending order
        radii = np.sort(radii)[::-1]
        pos_values = np.sort(pos_values)[::-1]

        radii = radii * 7.52e-3

        print('radii:', radii, 'pos:', pos_values)

        # Calculate slope of function
        slope, intercept, r_value, p_value, std_err = linregress(pos_values, radii)

        distance_to_chip = intercept / slope
        print(f"Distance to chip: {distance_to_chip:.4f}")

        dist_to_chip.append(distance_to_chip)

        slopes.append(slope)
        intercepts.append(intercept)

        radii_all.append(radii)


    def function(s, x, d, i):
        return s*(x-d) + i

    print(slopes)

    # Plot the cones
    plt.figure()
    for i in range(2,7):
        radii = radii_all[i-2]
        plt.scatter(pos_values + dist_to_chip[i-2], radii, label=f"Filter {i}")
        x = np.linspace(0, 30, 1000)
        y = function(slopes[i-2], x, dist_to_chip[i-2], intercepts[i-2])
        plt.plot(x, y, label=f"Filter {i}")
    plt.xlabel("Distance to chip [mm]")
    plt.ylabel("Spot radius [mm]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    project_folder = r"D:\Vincent\OptranWF_100_187_P_measurement_3\FRD"
    #run_from_existing_files(project_folder)

    plot_cones(project_folder)