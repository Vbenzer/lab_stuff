import numpy as np
import json
import matplotlib.pyplot as plt
import file_save_managment
import analyse_main
from astropy.io import fits

def main_measure_all_filters(project_folder:str, progress_signal=None):
    for i in range(2, 7):
        # Create project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        progress_signal.emit(f"Starting measurement for filter {i}")

        # Run the main measuring pipeline for each filter
        analyse_main.main_measure(filter_folder, progress_signal,
                                  batch_file_path=f"D:\stepper_motor\start_nina_with_fstop_filter{i}.bat")

        progress_signal.emit(f"Measurement for filter {i} complete!")

    print("All filters complete!")
    progress_signal.emit("All filters complete!")

def main_analyse_all_filters(project_folder:str, progress_signal=None):
    f_num = np.zeros(5)
    f_num_err = np.zeros(5)

    for i in range(2, 7):
        # Create project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        progress_signal.emit(f"Starting analysis for filter {i}")

        analyse_main.run_from_existing_files(filter_folder, progress_signal)

        progress_signal.emit(f"Analysis for filter {i} complete!")

        # Load the f-number and its error from the JSON file
        with open(filter_folder + "/Measurements/f_number.json") as f:
            data = json.load(f)
            f_num[i - 2] = data["f_number"]
            f_num_err[i - 2] = data["f_number_err"]

        progress_signal.emit(f"Filter {i} f-number: {f_num[i - 2]}")

    progress_signal.emit("All filters complete! Starting final plot.")

    # Input f-numbers
    input_f_num = np.array([6.095, 5.102, 4.571, 4.038, 3.539])
    input_f_num_err = np.array([0.036, 0.039, 0.01, 0.009, 0.023])

    # Sort the f-numbers in descending order
    f_num = np.sort(f_num)[::-1]
    f_num_err = np.sort(f_num_err)[::-1]

    # Plot the output f-numbers vs input f-numbers
    plt.errorbar(input_f_num, f_num, xerr=input_f_num_err, yerr=f_num_err, fmt="o", color="blue", label="Data points", capsize=5)
    plt.plot(input_f_num, f_num, linestyle='--', color="blue")
    plt.plot([2.5, 6], [2.5, 6], color="red", label="y=x")
    plt.xlabel("Input f/#")
    plt.ylabel("Output f/#")
    plt.title("Output f/# vs. Input f/#")
    plt.grid(True)
    plt.legend()
    plt.savefig(project_folder + "/f_number_vs_input.png")
    plt.close()
    #plt.show()

    # Set measurement name to last folder name of project folder
    measurement_name = project_folder.split("/")[-1]

    file_save_managment.save_measurement_hdf5("D:/Vincent/frd_measurements.h5", measurement_name, f_num, f_num_err)

def sutherland_plot(project_folder:str):
    # Input f-numbers
    input_f_num = np.array([6.095, 5.102, 4.571, 4.038, 3.539])
    input_f_num_err = np.array([0.036, 0.039, 0.01, 0.009, 0.023])
    #distance_to_

    for i in range(2, 7):
        # Access project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        # Load the LIGHT_0014_0.08s_reduced.fits file for each filter
        reduced_image_path = filter_folder + "/REDUCED/LIGHT_0014_0.08s_reduced.fits"
        with fits.open(reduced_image_path) as hdul:
            reduced_image = hdul[0].data.astype(np.float32)

        # Calculate the radius of a circle with input f-ratios
        # spot_radii = radii*7.52e-3, f_number = 1 / (2 * slope)
        #radius =



if __name__ == "__main__":
    exit()