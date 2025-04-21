import json
import os
import threading
import time

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.stats import linregress

import core.data_processing
import core.file_management
import qhy_ccd_take_image

from analysis.visualization import plot_main


def main_measure_frd(project_folder:str, progress_signal=None, exp_time:int=70000):

    from core.hardware import filter_wheel_color as mtf
    from core.hardware import motor_control as smc

    if progress_signal:
        progress_signal.emit("Starting measurement pipeline, initializing devices...")

    # Connect filter wheel and camera in thread, also reference motor
    fw_thread = threading.Thread(target=init_filter_wheel)
    cam_thread = threading.Thread(target=init_camera, args=([exp_time]))
    smc_thread = threading.Thread(target=smc.make_reference_move)

    fw_thread.start()
    cam_thread.start()
    smc_thread.start()

    # Make sure everything is ready
    fw_thread.join()
    cam_thread.join()
    smc_thread.join()

    # Define lists
    f_ratios = ["3.5", "4.0", "4.5", "5.0", "6.0"] # Skipping 2.5 bc it's too wide
    pos_values = [9.9, 5, 0]  # Values of the stepper motor positions

    # Iterate over f-ratios and positions
    for f_ratio in f_ratios:
        if progress_signal:
            progress_signal.emit(f"Measuring f-ratio {f_ratio}...")

        # Create folders for each f-ratio
        current_dark_folder = project_folder + f"/filter_{f_ratio}/DARK"
        current_light_folder = project_folder + f"/filter_{f_ratio}/LIGHT"

        os.makedirs(current_dark_folder)
        os.makedirs(current_light_folder)

        # Move filter wheel
        fw.move_to_filter(f_ratio)

        for pos in pos_values:
            # Move stepper motor
            smc.move_motor_to_position(pos) # Idea for minmax: start at 0, move to 5, move to 9.9, move to 5, move to 0 etc.

            # Close shutter
            mtf.move("Closed")
            time.sleep(0.5) # Just to make sure it's not moving

            # Take dark
            cam.take_single_frame(current_dark_folder, f"filter_{f_ratio}_pos_{pos}_dark.fits")

            # Open shutter
            mtf.move("Open")
            time.sleep(0.5)

            # Take image
            cam.take_single_frame(current_light_folder, f"filter_{f_ratio}_pos_{pos}_light.fits")

    if progress_signal:
        progress_signal.emit("All measurements completed!")


def run_from_existing_files(project_folder:str, progress_signal=None):
    """
    Run the analysis pipeline using existing files in the project folder.
    Args:
        project_folder: Path of the project folder.
    """

    # Write progress to file
    core.file_management.write_progress("Creating analysis folders")

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
    m_dark = core.data_processing.create_master_dark(dark_folder, plot=False)

    if progress_signal:
        progress_signal.emit("Reducing light frames")

    # Light frame reduction loop
    reduced_data = []
    for file_name in sorted(os.listdir(light_folder)):
        if file_name.endswith(".fits"):  # Only process FITS files
            file_path = os.path.join(light_folder, file_name)
            with fits.open(file_path) as hdul:
                light_frame = hdul[0].data.astype(np.float32)  # Convert to float for precision

            output_file = os.path.join(reduce_images_folder, os.path.splitext(file_name)[0] + "_reduced.fits")
            red_file = core.data_processing.reduce_image_with_dark(light_frame, m_dark, output_file, save=True)
            reduced_data.append(red_file)

    if progress_signal:
        progress_signal.emit("Calculating radii")

    # Radius calculation loop
    radii = core.data_processing.calculate_multiple_radii(reduced_data, measurements_folder)

    # Convert to numpy arrays
    radii = np.array(radii)
    pos_values = np.array(pos_values)

    print('radii:', radii, 'pos:', pos_values)

    if progress_signal:
        progress_signal.emit("Calculating F-number")

    # Calculate F-number
    f_number, f_number_err = calculate_f_number(radii, pos_values, plot_regression=False,
                                                save_path=measurements_folder)
    print(f"Calculated F-number (f/#): {f_number:.3f} ± {f_number_err:.3f}")


def main_analyse_all_filters(project_folder:str, progress_signal=None):
    """
    Run the frd analysis pipeline for all filters/f_number and plot the output f-numbers vs input f-numbers.
    Args:
        project_folder: Path to the project folder.
        progress_signal: Progress signal.

    Returns:

    """
    f_num = np.zeros(5)
    f_num_err = np.zeros(5)

    folder_list = [folder for folder in sorted(os.listdir(project_folder)) if "filter" in folder]
    for i, folder in enumerate(folder_list):
        # Define project subfolder for each filter
        filter_folder = os.path.join(project_folder, folder)

        if progress_signal:
            progress_signal.emit(f"Starting analysis for: {folder}")

        analysis.frd_analysis.run_from_existing_files(filter_folder, progress_signal)

        if progress_signal:
            progress_signal.emit(f"Analysis for {folder} complete!")

        # Load the f-number and its error from the JSON file
        with open(filter_folder + "/Measurements/f_number.json") as f:
            data = json.load(f)
            f_num[i] = data["f_number"]
            f_num_err[i] = data["f_number_err"]

        if progress_signal:
            progress_signal.emit(f"Result: {folder} with f-number: {f_num[i]}")

    if progress_signal:
        progress_signal.emit("All filters complete! Starting final plot.")

    # Save the output f-numbers to a json
    with open(project_folder + "/f_number.json", "w") as f:
        json.dump({"f_number": f_num.tolist(), "f_number_err": f_num_err.tolist()}, f)

    plot_main(project_folder)

    # Set measurement name to last folder name of project folder
    #measurement_name = project_folder.split("/")[-1]

    #file_management.save_measurement_hdf5("D:/Vincent/frd_measurements.h5", measurement_name, f_num, f_num_err)


def calculate_f_number(radii: np.ndarray, ccd_positions: np.ndarray, plot_regression:bool=False
                       , save_plot:bool=True, save_path:str=None):
    """
    Calculate the F-number (f/#) from the spot radii and CCD positions.

    Parameters:
        radii : Array of spot radii
        ccd_positions : Array of CCD positions
        plot_regression : If True, plot the linear regression of the data.
        save_plot : If True, save the plot to a file.
        save_path : Path to save the plot file.

    Returns:
        float: The calculated F-number with error.
    """

    # Convert spot radii to millimeters
    spot_radii = radii*7.52e-3  #mm/px value from camera
    spot_radii = np.sort(spot_radii)[::-1]  #Sort in descending order because motor is reversed when measuring fiber frd

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(ccd_positions, spot_radii)

    print("Slope: ", slope, "Std_err: ", std_err)

    # Calculate distance to chip
    distance_to_chip = intercept / slope
    distance_to_chip_err = (intercept/slope**2)*std_err

    print(f"Distance to chip: {distance_to_chip:.2f} ± {distance_to_chip_err:.2f}")

    # Calculate the F-number using the formula: f/# = 1 / (2 * tan(theta_o))
    f_number = 1 / (2 * slope)
    f_number_err = abs(-1/(2*slope**2)*std_err)

    # Plot regression if requested
    if plot_regression or save_plot:
        plt.figure()
        plt.scatter(ccd_positions, spot_radii, label="Data points")
        plt.plot(ccd_positions, slope * ccd_positions + intercept, color="green", label="Linear fit")
        plt.xlabel("CCD Position [mm]")
        plt.ylabel("Spot Radius [mm]")
        plt.title("Linear Regression of Spot Radius vs. CCD Position")
        plt.legend()
        plt.grid(True)

        # Add distance_to_chip value to the plot
        plt.text(0.05, 0.95, f"Distance to chip: {distance_to_chip:.2f} ± {distance_to_chip_err:.2f} mm",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

        if save_plot:
            if save_path is None:
                raise ValueError("'save_path' must be provided")
            plt.savefig(save_path+"regression_plot.png")

        if plot_regression:
            plt.show()
        plt.close()
    with open(save_path+"f_number.json","w") as f:
        json.dump({"f_number":f_number,"f_number_err":f_number_err, "distance_to_chip":distance_to_chip
                   ,"distance_to_chip_err":distance_to_chip_err}, f)

    return f_number,f_number_err


fw = None


def init_filter_wheel():
    global fw
    fw = qhycfw3_filter_wheel_control.FilterWheel('COM5')


def init_camera(exp_time:int):
    global cam
    cam = qhy_ccd_take_image.Camera(exp_time=exp_time)


cam = None
