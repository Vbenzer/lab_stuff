import numpy as np
import json
import matplotlib.pyplot as plt
import file_save_managment
import analyse_main
from astropy.io import fits

def main_measure_all_filters(project_folder:str, progress_signal=None):
    """
    Run the measuring pipeline for all filters.
    Args:
        project_folder: Path to the project folder.
        progress_signal: Progress signal.

    Returns:

    """
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
    """
    Run the analysis pipeline for all filters and plot the output f-numbers vs input f-numbers.
    Args:
        project_folder: Path to the project folder.
        progress_signal: Progress signal.

    Returns:

    """
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
    input_f_num = np.array([6.21, 5.103, 4.571, 4.063, 3.597]) # These are from the setup_F#_EE_98 file, 18.2.25
    input_f_num_err = np.array([0.04, 0.007, 0.01, 0.005, 0.013])

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
    """
    Plot the encircled energy vs output aperture f/#, to visualize light loss of the fiber with different f/#.
    Args:
        project_folder: Path to the project folder.

    Returns:

    """
    # Input f-numbers
    input_f_num = np.array([6.21, 5.103, 4.571, 4.063, 3.597]) # These are from the setup_F#_EE_98 file, 18.2.25
    input_f_num_err = np.array([0.04, 0.007, 0.01, 0.005, 0.013])

    # Load the distance to chip from the JSON file
    dist = np.zeros(5)
    for i in range(2, 7):
        with open(project_folder + f"/filter_{i}/Measurements/f_number.json") as f:
            data = json.load(f)
        dist[i-2] = data["distance_to_chip"]

    distance_to_chip = np.mean(dist)

    ee_list = []

    for i in range(2, 7):
        print(f"Processing filter {i}")
        # Access project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        # Load the LIGHT_0014_0.08s_reduced.fits file for each filter
        reduced_image_path = filter_folder + "/REDUCED/LIGHT_0014_0.08s_reduced.fits"
        with fits.open(reduced_image_path) as hdul:
            reduced_image = hdul[0].data.astype(np.float32)

        import image_analysation as ia
        # Trim data to area of interest (perhaps not necessary with better background reduction)
        trimmed_data = ia.cut_image(reduced_image, margin=1000)  # If margin too low mask won't fit in image

        # Locate center of mass within trimmed image (array)
        com = ia.LocateFocus(trimmed_data)

        ee_sublist = []

        for fnum in input_f_num:
            # Calculate the radius of a circle with input f-ratios
            aperture_radius = (distance_to_chip + 9.9) / (2 * fnum) # 9.9: Distance to chip at 0 position

            # Convert to pixels
            aperture_radius = aperture_radius // 7.52e-3

            # Create a circle mask
            import sg_pipeline
            mask = sg_pipeline.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius
                                                    , plot_mask=False)

            # Calculate the flux within the mask
            flux = np.sum(mask * trimmed_data)

            # Calculate the flux outside the mask
            mask_outside = np.invert(mask)
            flux_outside = np.sum(mask_outside * trimmed_data)

            # Calculate the encircled energy of the mask
            ee = flux / (flux + flux_outside)
            ee_sublist.append(ee)

        ee_list.append(ee_sublist)

    # Make spline fit
    from scipy.interpolate import PchipInterpolator

    # Change order of ee and f-numbers
    ee_list = np.array(ee_list)
    ee_list = np.flip(ee_list, axis=1)

    input_f_num = np.flip(input_f_num)
    input_f_num_err = np.flip(input_f_num_err)

    popt_list = []
    for i, ee in enumerate(ee_list):
        # Create the spline
        spline = PchipInterpolator(input_f_num, ee)
        popt_list.append([spline])

    # Plot the encircled energy vs input f-numbers
    labels = ["F/6.21", "F/5.103", "F/4.571", "F/4.063", "F/3.597"]
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    x_range = np.linspace(min(input_f_num), max(input_f_num), 1000)
    for idx, ee in enumerate(ee_list):
        # Plot the data points
        plt.errorbar(input_f_num, ee, xerr=input_f_num_err, fmt="o", color=colors[idx % len(colors)],
                     label=f"Input {labels[idx % len(labels)]}", capsize=5)

        # Plot the spline
        plt.plot(x_range, popt_list[idx][0](x_range), linestyle='-', color=colors[idx % len(colors)], linewidth=1)

        # Visualize the light loss at the output f-ratio
        plt.vlines(input_f_num[4-idx], ee[4-idx], 1, color=colors[idx % len(colors)], linestyle='--', linewidth=0.5)

        # Add text to the plot
        alignment = 'right' if idx == 0 else 'left'
        padding = -0.05 if idx == 0 else 0.05
        plt.text(input_f_num[4 - idx] + padding, ee[4 - idx], f"{ee[4 - idx]:.3f}", color=colors[idx % len(colors)],
                 fontsize=8,
                 verticalalignment='center_baseline', horizontalalignment=alignment)

    # For the legend
    plt.vlines([], [], [], color='black', linestyle='--', linewidth=0.5, label='Light loss at input = output f-ratio')

    plt.xlabel("Output Aperture f/#")
    plt.ylabel("Encircled Energy")
    plt.title("Encircled Energy vs. Output Aperture f/#")
    plt.grid(True)
    plt.legend()
    plt.savefig(project_folder + "/encircled_energy_vs_output.png")
    plt.close()
    # plt.show()



if __name__ == "__main__":
    project_folder = "D:/Vincent/40x120_300A_measurement_4/FRD"
    sutherland_plot(project_folder)