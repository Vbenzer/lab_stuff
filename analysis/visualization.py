"""Module visualization.py.

Auto-generated docstring for better readability.
"""
import json
import os

import h5py
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt, cm as cm
from scipy.stats import linregress
from skimage import io, measure

import analysis.sg_analysis
import core.data_processing
from analysis.sg_analysis import cut_image_around_comk
from core.file_management import load_frd_calibration_data

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


def plot_main(project_folder:str, calibration_folder:str):
    """
    Plot the output f-numbers vs input f-numbers.
    Args:
        project_folder: Path to the project folder.

    Returns:

    """
    # Load the output f-numbers from the JSON file
    with open(project_folder + "/f_number.json") as f:
        data = json.load(f)
        f_num = np.array(data["f_number"])
        f_num_err = np.array(data["f_number_err"])

    # Input f-numbers
    input_f_num, input_f_num_err =  load_frd_calibration_data(calibration_folder)
    # Convert to numpy arrays
    input_f_num = np.array(input_f_num)
    input_f_num_err = np.array(input_f_num_err)

    """input_f_num = np.array(
        [3.637, 4.089, 4.571, 5.103, 6.095])  # These are from the setup_F#_EE_98_ph10_0000 file, 5.5.25
    input_f_num_err = np.array([0.010, 0.009, 0.010, 0.007, 0.030])"""
    # Change order of input f-numbers
    input_f_num = np.flip(input_f_num)
    input_f_num_err = np.flip(input_f_num_err)

    # Sort the f-numbers in descending order
    f_num = np.sort(f_num)[::-1]
    f_num_err = np.sort(f_num_err)[::-1]

    # Get perimeter of the plot
    max_f_num = max(input_f_num[0], f_num[0])
    min_f_num = min(input_f_num[-1], f_num[-1])

    # Plot the output f-numbers vs input f-numbers
    plt.errorbar(input_f_num, f_num, xerr=input_f_num_err, yerr=f_num_err, fmt="o", color="blue", label="Data points",
                 capsize=5)
    plt.plot(input_f_num, f_num, linestyle='--', color="blue")
    plt.plot([input_f_num[4] - 0.2, input_f_num[0] + 0.2], [input_f_num[4] - 0.2, input_f_num[0] + 0.2], color="red",
             label="y=x")

    for i in range(5):
        plt.vlines(input_f_num[i], 0, f_num[i], color="black", linestyle='--', linewidth=0.5)
        plt.text(min_f_num - 0.04, f_num[i], f"{f_num[i]:.2f}", fontsize=8, verticalalignment='bottom',
                 horizontalalignment='right')

    for i in range(5):
        plt.hlines(f_num[i], 0, input_f_num[i], color="black", linestyle='--', linewidth=0.5)
        plt.text(input_f_num[i] + 0.05, min_f_num - 0.2, f"{input_f_num[i]:.2f}", fontsize=8,
                 verticalalignment='bottom', horizontalalignment='left')

    plt.xlim([min_f_num - 0.2, max_f_num + 0.25])
    plt.ylim([min_f_num - 0.2, max_f_num + 0.25])
    plt.xlabel("Input f/#")
    plt.ylabel("Output f/#")
    plt.title("Output f/# vs. Input f/#")
    plt.grid(True)
    plt.legend()
    plt.savefig(project_folder + "/f_number_vs_input.png")
    plt.close()
    # plt.show()


def sutherland_plot(project_folder:str):
    """
    Plot the encircled energy vs output aperture f/#, to visualize light loss of the fiber with different f/#.
    Args:
        project_folder: Path to the project folder.

    Returns:

    """
    # Input f-numbers
    input_f_num = np.array(
        [3.637, 4.089, 4.571, 5.103, 6.095])  # These are from the setup_F#_EE_98_ph10_0000 file, 5.5.25
    input_f_num_err = np.array([0.010, 0.009, 0.010, 0.007, 0.030])
    # Change order of input f-numbers
    input_f_num = np.flip(input_f_num)
    input_f_num_err = np.flip(input_f_num_err)

    # Load the distance to chip from the JSON file
    dist = np.zeros(5)
    dist_err = np.zeros(5)

    folder_list = [folder for folder in sorted(os.listdir(project_folder))[::-1] if "filter" in folder]

    # Ensure filters are correctly ordered
    if "filter_2" in folder_list:
        print("Realigning filter list")
        folder_list = folder_list[::-1]

    for i, folder in enumerate(folder_list):
        with open(project_folder + f"/{folder}/Measurements/f_number.json") as f:
            data = json.load(f)
        dist[i] = data["distance_to_chip"]
        dist_err[i] = data["distance_to_chip_err"]

    weights = 1 / dist_err ** 2
    #print("Weights:", weights)
    distance_to_chip = np.sum(dist * weights) / np.sum(weights) # Weighted mean
    #print("Distance to chip:", distance_to_chip)
    sigma_weighted_mean = np.sqrt(1 / np.sum(weights))
    #print("Sigma weighted mean:", sigma_weighted_mean)
    std_dev = np.std(dist, ddof=1)
    #print("Standard deviation:", std_dev)
    distance_to_chip_err = np.sqrt(sigma_weighted_mean**2 + std_dev**2)

    #print(distance_to_chip_err)

    ee_list = []
    ee_err_list = []

    for i, folder in enumerate(folder_list):
        print(f"Processing: {folder}")
        # Access project subfolder for each filter
        filter_folder = project_folder + f"/{folder}"

        image_list = os.listdir(filter_folder+ "/REDUCED")
        if "LIGHT_0014_0.08s_reduced.fits" in image_list:
            image_name = "/REDUCED/LIGHT_0014_0.08s_reduced.fits"
        else:
            image_name = f"/REDUCED/{folder}_pos_0_light_reduced.fits"

        # Load the LIGHT_0014_0.08s_reduced.fits file for each filter
        reduced_image_path = filter_folder + image_name
        with fits.open(reduced_image_path) as hdul:
            reduced_image = hdul[0].data.astype(np.float32)

        # Trim data to area of interest
        trimmed_data = core.data_processing.cut_image(reduced_image, margin=1000)  # If margin too low mask won't fit in image

        # Locate center of mass within trimmed image (array)
        com = core.data_processing.locate_focus(trimmed_data)

        ee_sublist = []
        ee_err_sublist = []

        old_data = False
        if old_data:
            distance_to_chip_at_zero_position = 9.9
        else:
            distance_to_chip_at_zero_position = 0

        for fnum, fnum_err in zip(input_f_num, input_f_num_err):
            # Calculate the radius of a circle with input f-ratios
            aperture_radius = (distance_to_chip + distance_to_chip_at_zero_position) / (2 * fnum)  # 9.9: Distance to chip at 0 position
            aperture_radius_err = aperture_radius * np.sqrt(
                (distance_to_chip_err / (distance_to_chip + 9.9)) ** 2 + (fnum_err / fnum) ** 2)

            # Convert to pixels (keep floating-point for error calculations)
            aperture_radius_pix = aperture_radius / 7.52e-3
            aperture_radius_err_pix = aperture_radius_err / 7.52e-3

            #print("aperture_err", aperture_radius_err_pix)

            # Round only for actual mask application
            aperture_radius = int(round(aperture_radius_pix))

            # Create a circle mask
            mask = analysis.sg_analysis.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius, plot_mask=False)

            # Calculate the flux within the mask
            flux = np.sum(mask * trimmed_data)

            # Calculate the flux outside the mask
            mask_outside = np.invert(mask)
            flux_outside = np.sum(mask_outside * trimmed_data)

            # Calculate the encircled energy of the mask
            ee = flux / (flux + flux_outside)

            """# Plot outline of mask with original image and ee value
            from skimage import measure
            plt.figure()
            plt.imshow(trimmed_data, cmap='gray')
            outline = measure.find_contours(mask, 0.5)[0]
            plt.plot(outline[:, 1], outline[:, 0], color='red', linewidth=0.8)
            plt.title(f"Encircled energy: {ee:.3f}")
            plt.axis('off')
            plt.savefig(filter_folder + f"/encircled_energy_{fnum}.png", dpi=300)"""

            # Compute uncertainties (assuming Poisson statistics for flux)
            flux_err = np.sqrt(flux) if flux > 0 else 0  # Avoid division by zero
            flux_outside_err = np.sqrt(flux_outside) if flux_outside > 0 else 0  # Avoid division by zero

            #print("Flux errors", flux_err, flux_outside_err)

            # Compute EE uncertainty
            ee_err = ee * np.sqrt(
                (flux_err / flux) ** 2 +
                (flux_outside_err / flux_outside) ** 2 +
                (aperture_radius_err_pix / aperture_radius_pix) ** 2
            )

            #print("ee, ee_err:", ee, ee_err)
            ee_sublist.append(ee)
            ee_err_sublist.append(ee_err)

        ee_list.append(ee_sublist)
        ee_err_list.append(ee_err_sublist)

    # Make spline fit
    from scipy.interpolate import PchipInterpolator

    # Change order of ee and f-numbers
    ee_list = np.array(ee_list)
    ee_list = np.flip(ee_list, axis=1)
    ee_err_list = np.array(ee_err_list)
    ee_err_list = np.flip(ee_err_list, axis=1)

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
        plt.vlines(input_f_num[4-idx], ee[4-idx], 1.005, color=colors[idx % len(colors)], linestyle='--', linewidth=0.5)

        # Add text to the plot
        alignment = "center" #'right' if idx == 0 else 'left'
        padding = 0 #-0.05 if idx == 0 else 0.05
        plt.text(input_f_num[4 - idx] + padding, 1.02, f"{ee[4 - idx]:.3f}", color=colors[idx % len(colors)],
                 fontsize=8,
                 verticalalignment='top', horizontalalignment=alignment)

    # For the legend
    plt.vlines([], [], [], color='black', linestyle='--', linewidth=0.5, label='Light loss at input = output f-ratio')

    plt.xlabel("Output Aperture f/#")
    plt.ylabel("Relative Encircled Energy")
    plt.title("Encircled Energy vs. Output Aperture f/#")
    plt.grid(True)
    plt.legend()
    plt.savefig(project_folder + "/encircled_energy_vs_output.png")
    plt.close()
    # plt.show()


def plot_f_ratio_circles_on_raw(project_folder):
        from skimage import measure
        from matplotlib.colors import LogNorm

        # Input f-numbers
        input_f_num = np.array(
            [3.637, 4.089, 4.571, 5.103, 6.095])  # These are from the setup_F#_EE_98_ph10_0000 file, 5.5.25
        input_f_num_err = np.array([0.010, 0.009, 0.010, 0.007, 0.030])
        # Change order of input f-numbers
        input_f_num = np.flip(input_f_num)
        input_f_num_err = np.flip(input_f_num_err)

        # Read NA from fiber_data.json
        with open(os.path.join(os.path.dirname(project_folder), 'fiber_data.json')) as f:
            fiber_data = json.load(f)
            NA = fiber_data['numerical_aperature']
            if NA == "":
                NA = 0.22
            else:
                NA = float(NA)


        filter_to_name_dict = {"2": '6.21', "3": '5.103', "4": '4.571', "5": '4.063', "6": '3.597'}

        folder_list = [folder for folder in sorted(os.listdir(project_folder)[::-1]) if "filter" in folder]

        # Ensure filters are correctly ordered
        if "filter_2" in folder_list:
            print("Realigning filter list")
            #folder_list = folder_list[::-1]
            bad_names = True
        else:
            folder_list = folder_list[::-1]
            bad_names = False

        # Load the distance to chip from the JSON file
        dist = np.zeros(5)
        for i, folder in enumerate(folder_list):
            with open(project_folder + f"/{folder}/Measurements/f_number.json") as f:
                data = json.load(f)
            dist[i] = data["distance_to_chip"]

        distance_to_chip = np.mean(dist)

        for i, folder in enumerate(folder_list):
            print(f"Processing {folder}")
            # Access project subfolder for each filter
            filter_folder = project_folder + f"/{folder}"

            f_ratio_images_folder = filter_folder + "/f_ratio_images"
            os.makedirs(f_ratio_images_folder, exist_ok=True)

            image_list = os.listdir(filter_folder + "/REDUCED")
            if "LIGHT_0014_0.08s_reduced.fits" in image_list:
                image_name = "/REDUCED/LIGHT_0014_0.08s_reduced.fits"
            else:
                image_name = f"/REDUCED/{folder}_pos_0_light_reduced.fits"

            # Load the LIGHT_0014_0.08s_reduced.fits file for each filter
            reduced_image_path = filter_folder + image_name
            with fits.open(reduced_image_path) as hdul:
                reduced_image = hdul[0].data.astype(np.float32)

            # Trim data to area of interest (perhaps not necessary with better background reduction)
            trimmed_data = core.data_processing.cut_image(reduced_image, margin=1000)  # If margin too low mask won't fit in image

            # Locate center of mass within trimmed image (array)
            com = core.data_processing.locate_focus(trimmed_data)

            # Calculate radius of NA
            aperture_radius_NA = (distance_to_chip) * NA / np.sqrt(1 - NA ** 2)  #+ 9.9: Distance to chip at 0 position (old setup)
            aperture_radius_NA = aperture_radius_NA // 7.52e-3
            mask_NA = analysis.sg_analysis.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius_NA, plot_mask=False)
            mask_outline_NA = measure.find_contours(mask_NA, 0.5)[0]

            mask_outline_list = []
            filter_name = filter_to_name_dict[str(i+2)]

            for fnum in input_f_num:
                print(f"Processing f/{filter_name}, f/{fnum}")
                # Calculate the radius of a circle with input f-ratios
                aperture_radius = (distance_to_chip) / (2 * fnum)  # 9.9: Distance to chip at 0 position (old setup)

                # Convert to pixels
                aperture_radius = aperture_radius // 7.52e-3

                # Create a circle mask
                mask = analysis.sg_analysis.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius, plot_mask=False)

                mask_outline_list.append(measure.find_contours(mask, 0.5)[0])

            # Boost everything so that the lowest value is 0 for log scaling
            trimmed_data = trimmed_data - np.min(trimmed_data)
            print(np.min(trimmed_data))

            # Save trimmed as fits
            hdu = fits.PrimaryHDU(trimmed_data)
            hdu.writeto(f_ratio_images_folder + f"/trimmed_{filter_name}.fits", overwrite=True)

            # Plot the mask on the raw image
            plt.figure()
            plt.title(f"Input f/{filter_name} with artificial apertures")
            plt.imshow(trimmed_data, cmap='gray', norm=LogNorm())
            # Add color scale for pixel value next to the image
            cbar = plt.colorbar()
            cbar.set_label('Pixel value')
            plt.plot(mask_outline_NA[:, 1], mask_outline_NA[:, 0], color='green', linewidth=0.8, alpha=0.5, dashes=(5, 10),
                     label=f'NA = {NA}')

            # Use a colormap for the outlines
            cmap = ['blue', 'orange', 'purple', 'red', 'brown']
            for idx, mask_outline in enumerate(mask_outline_list):
                color = cmap[idx]
                plt.plot(mask_outline[:, 1], mask_outline[:, 0], color=color, linewidth=0.5, linestyle="--", alpha=0.5, dashes=(5, 10),
                         label=f"f/{input_f_num[idx]}")

            plt.axis('off')
            plt.legend(framealpha=0.5)
            plt.savefig(f_ratio_images_folder + f"/input_{filter_name}_with_artificial_apertures.png", dpi=300)
            plt.close()


def plot_horizontal_cut_ff(project_folder):
    folder_list = [folder for folder in sorted(os.listdir(project_folder)[::-1]) if "filter" in folder]

    for i, folder in enumerate(folder_list):
        print(f"Processing {folder}")
        # Access project subfolder for each filter
        filter_folder = project_folder + f"/{folder}"

        plots_folder = filter_folder + "/plots"
        os.makedirs(plots_folder, exist_ok=True)

        image_list = os.listdir(filter_folder + "/REDUCED")
        if "LIGHT_0014_0.08s_reduced.fits" in image_list:
            image_name = "/REDUCED/LIGHT_0014_0.08s_reduced.fits"
        else:
            image_name = f"/REDUCED/{folder}_pos_0_light_reduced.fits"

        # Load the LIGHT_0014_0.08s_reduced.fits file for each filter
        reduced_image_path = filter_folder + image_name
        with fits.open(reduced_image_path) as hdul:
            reduced_image = hdul[0].data.astype(np.float32)

        # Trim data to area of interest
        trimmed_data = core.data_processing.cut_image(reduced_image, margin=500)

        # Locate center of mass within trimmed image (array)
        com = core.data_processing.locate_focus(trimmed_data)

        # Convert center of mass to integer indices
        com_x = int(com[0])

        # Create a horizontal cut through the center of mass
        cut = trimmed_data[com_x, :]

        # Plot the horizontal cut
        plt.figure()
        plt.plot(cut)
        plt.title(f"Horizontal cut through center of mass")
        plt.xlabel("Pixel")
        plt.ylabel("Flux")
        plt.grid(True)
        plt.savefig(plots_folder + f"/horizontal_cut.png")
        plt.close()


def plot_measurements(filename:str):
    """
    Read frd measurements from HDF5 file and plot the data.
    Args:
        filename: Path to the HDF5 file containing the measurements.

    Returns:

    """
    # Constant input f-number (the same for all measurements)
    input_f_num = np.array(
        [3.637, 4.089, 4.571, 5.103, 6.095])  # These are from the setup_F#_EE_98_ph10_0000 file, 5.5.25
    input_f_num_err = np.array([0.010, 0.009, 0.010, 0.007, 0.030])
    # Change order of input f-numbers
    input_f_num = np.flip(input_f_num)
    input_f_num_err = np.flip(input_f_num_err)

    # Read data from the HDF5 file
    with h5py.File(filename, 'r') as file:
        # Initialize empty lists to store data for plotting
        f_numbers = []
        f_numbers_err = []
        measurement_names = []

        # Loop through all groups (each measurement) in the HDF5 file
        for measurement_name in file.keys():
            measurement_group = file[measurement_name]
            f_number = np.array(measurement_group['f_number'])
            f_number_err = np.array(measurement_group['f_number_err'])

            # Append the data for this measurement
            f_numbers.append(f_number)
            f_numbers_err.append(f_number_err)
            measurement_names.append(measurement_name)

        # Convert lists to numpy arrays for easier handling
        f_numbers = np.array(f_numbers)
        f_numbers_err = np.array(f_numbers_err)

    # Plot the measurements
    plt.figure(figsize=(10, 6))

    # Create a colormap to use for different measurements
    colors = cm.plasma(np.linspace(0, 1, len(measurement_names)))

    # If exists remove "Vincent" from the measurement names
    measurement_names = [name.replace("Vincent\\", "") for name in measurement_names]
    measurement_names = [name.replace("\\FRD", "") for name in measurement_names]

    # Plot each measurement with error bars and dashed lines
    for i, measurement_name in enumerate(measurement_names):
        color = colors[i]  # Get the color for this measurement
        plt.errorbar(input_f_num, f_numbers[i],xerr=input_f_num_err, yerr=f_numbers_err[i], fmt='o', label=measurement_name, color=color)
        plt.plot(input_f_num, f_numbers[i], linestyle='--', color=color)  # Dashed line with the same color

    # Add reference line y = x (no FRD)
    plt.plot([min(input_f_num), max(input_f_num)], [min(input_f_num), max(input_f_num)], 'k--', label='y = x (No FRD)')

    # Labels and title
    plt.xlabel('Input f/#')
    plt.ylabel('Output f/#')
    plt.title('Output f/# vs. Input f/#')

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig("D:/Vincent/FRD_main_plot.png")
    plt.show()


def make_comparison_video(main_folder:str, fiber_diameter, progress_signal=None):
    """
    Creates a video comparing the entrance and exit images.
    Args:
        main_folder: Main working directory.
        fiber_diameter: Fiber diameter in pixels.


    """
    # Define the image folders
    entrance_image_folder = os.path.join(main_folder, "entrance/reduced")
    exit_image_folder = os.path.join(main_folder, "exit/reduced")
    video_prep_exit_folder = os.path.join(main_folder, "video_prep/exit")
    video_prep_entrance_folder = os.path.join(main_folder, "video_prep/entrance")
    sg_parameters_file = os.path.join(main_folder, "scrambling_gain_parameters.json")

    # Get the image files
    entrance_image_files = [f for f in os.listdir(entrance_image_folder) if f.endswith('reduced.png')]
    entrance_image_files = sorted(entrance_image_files)
    exit_image_files = [f for f in os.listdir(exit_image_folder) if f.endswith('reduced.png')]
    exit_image_files = sorted(exit_image_files)

    # Calculate the radius of the fiber in pixels, also handle rectangular case
    if isinstance(fiber_diameter, (tuple,list)):
        fiber_input_radius = (int(fiber_diameter[0] / 0.5169363821005045 / 2), int(fiber_diameter[1] / 0.5169363821005045 / 2))
        fiber_exit_radius = (int(fiber_diameter[0] / 0.439453125 / 2), int(fiber_diameter[1] / 0.439453125 / 2))
    else:
        fiber_input_radius = int(fiber_diameter / 0.5169363821005045 / 2)
        fiber_exit_radius = int(fiber_diameter / 0.439453125 / 2)

    # Margin for better visuals
    margin = 50

    # Create folders if they don't exist
    os.makedirs(video_prep_entrance_folder, exist_ok=True)
    os.makedirs(video_prep_exit_folder, exist_ok=True)

    # Read entrance_comk and exit_comk from json file
    if progress_signal:
        progress_signal.emit("Reading parameters from json")
    with open(sg_parameters_file, 'r') as f:
        parameters = json.load(f)

    entrance_comk = parameters["entrance_comk"]
    exit_comk = parameters["exit_comk"]
    exit_comk = np.array(exit_comk)
    exit_comk_median = np.median(exit_comk, axis=0)


    if progress_signal:
        progress_signal.emit("Cutting images")

    # Cut the images to the same size around the fiber
    for i in range(len(entrance_image_files)):
        # Get comk from the list
        comk = entrance_comk[i]

        # Get the image file name
        image_file = entrance_image_files[i]

        # Read the image
        image_path = os.path.join(entrance_image_folder, image_file)
        image = io.imread(image_path)

        # Cut the image around the fiber
        image = cut_image_around_comk(image, comk, fiber_input_radius, margin)

        # Save the cut image
        io.imsave(os.path.join(video_prep_entrance_folder, image_file.replace(".png","_cut.png")), image)

    for i in range(len(exit_image_files)):
        comk = exit_comk_median
        image_file = exit_image_files[i]

        image_path = os.path.join(exit_image_folder, image_file)
        image = io.imread(image_path)
        image = cut_image_around_comk(image, comk, fiber_exit_radius, margin)
        io.imsave(os.path.join(video_prep_exit_folder, image_file.replace(".png","_cut.png")), image)

    # No scaling needed, done in video creation

    # Create video #Todo: Enhance visuals of the video (looks kinda bad but works for now)
    if progress_signal:
        progress_signal.emit("Creating video")
    from moviepy import VideoFileClip, clips_array
    import imageio

    video_name = os.path.join(main_folder, "comparison_video.mp4")

    video_entrance_files = [f for f in os.listdir(video_prep_entrance_folder) if f.endswith('cut.png')]
    video_entrance_files = sorted(video_entrance_files)
    video_exit_files = [f for f in os.listdir(video_prep_exit_folder) if f.endswith('cut.png')]
    video_exit_files = sorted(video_exit_files)

    # Create entrance video
    entrance_video_path = os.path.join(video_prep_entrance_folder, "entrance_video.mp4")
    imageio.mimsave(entrance_video_path,
                    [io.imread(os.path.join(video_prep_entrance_folder, f)) for f in video_entrance_files],
                    fps=5, quality = 10)

    # Create exit video
    exit_video_path = os.path.join(video_prep_exit_folder, "exit_video.mp4")
    imageio.mimsave(exit_video_path,
                    [io.imread(os.path.join(video_prep_exit_folder, f)) for f in video_exit_files],
                    fps=5, quality = 10)

    # Load videos
    entrance_clip = VideoFileClip(entrance_video_path)
    exit_clip = VideoFileClip(exit_video_path)

    # Combine videos side by side
    final_clip = clips_array([[entrance_clip, exit_clip]])
    final_clip.write_videofile(video_name, fps=5, codec='libx264', bitrate='100000k')


def plot_masks(main_folder:str, fiber_diameter:int, progress_signal=None):
    """
    Plot the masks overlaid on the images
    Args:
        main_folder: Main folder of the fiber.
        fiber_diameter: Fiber diameter or size of the fiber in pixels
        progress_signal: Progress signal to update the progress.


    """
    # Calculate the radius of the fiber in pixels, also handle rectangular case
    if isinstance(fiber_diameter, tuple):
        fiber_input_radius = (int(fiber_diameter[0] / 0.5169363821005045 / 2), int(fiber_diameter[1] / 0.5169363821005045 / 2))
        fiber_exit_radius = (int(fiber_diameter[0] / 0.439453125 / 2), int(fiber_diameter[1] / 0.439453125 / 2))
    else:
        fiber_input_radius = int(fiber_diameter / 0.5169363821005045 / 2)
        fiber_exit_radius = int(fiber_diameter / 0.439453125 / 2)

    # Define the image folders
    entrance_mask_folder = os.path.join(main_folder, "entrance/mask")
    exit_mask_folder = os.path.join(main_folder, "exit/mask")
    plot_folder = os.path.join(main_folder, "plots")
    entrance_overlay_folder = os.path.join(plot_folder, "entrance_overlay")
    exit_overlay_folder = os.path.join(plot_folder, "exit_overlay")

    # Create plot folder if it doesn't exist
    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(entrance_overlay_folder, exist_ok=True)
    os.makedirs(exit_overlay_folder, exist_ok=True)

    # Get the mask files
    entrance_mask_files = [f for f in os.listdir(entrance_mask_folder) if f.endswith('mask.png')]
    entrance_mask_files = sorted(entrance_mask_files)
    exit_mask_files = [f for f in os.listdir(exit_mask_folder) if f.endswith('mask.png')]
    exit_mask_files = sorted(exit_mask_files)

    # Read entrance coms from json file
    if progress_signal:
        progress_signal.emit("Reading parameters from json")
    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    # Get the parameters
    entrance_coms = parameters["entrance_coms"]
    entrance_comk = parameters["entrance_comk"]
    exit_comk = parameters["exit_comk"]

    # Calculate the median of entrance_coms and exit_comk
    entrance_coms = np.array(entrance_coms)
    exit_comk = np.array(exit_comk)
    entrance_com_median = np.median(entrance_coms, axis=0)
    exit_comk_median = np.median(exit_comk, axis=0)

    # Margin for better visuals
    margin = 50

    # Plot Mask outline overlaid on the image
    for i in range(len(entrance_mask_files)):
        # Get the center of mass and center of mask
        com = entrance_com_median
        comk = entrance_comk[i]

        # Read the mask and image
        entrance_mask = io.imread(os.path.join(entrance_mask_folder, entrance_mask_files[i]))
        entrance_image = io.imread(os.path.join(main_folder, "entrance/reduced", entrance_mask_files[i].replace("_mask", "")))

        # Cut image to the fiber size
        entrance_image_cut = cut_image_around_comk(entrance_image, comk, fiber_input_radius, margin)

        # Adjust com to the cut image, also handle rectangular case
        if isinstance(fiber_diameter, tuple):
            bigger_side = max(fiber_input_radius)
            com = [com[0] - comk[0] + bigger_side + margin, com[1] - comk[1] + bigger_side + margin]
        else:
            com = [com[0] - comk[0] + fiber_input_radius + margin, com[1] - comk[1] + fiber_input_radius + margin]

        # Cut the mask to the fiber size
        entrance_mask_cut = cut_image_around_comk(entrance_mask, comk, fiber_input_radius, margin)

        # Find the mask outline
        entrance_mask_outline = measure.find_contours(entrance_mask_cut, 0.5)[0]

        # Plot the image with the mask overlay
        # Define quality of the plot
        dpi = 100
        size = [entrance_image_cut.shape[1] / dpi, float(entrance_image_cut.shape[0] / dpi)]

        # noinspection PyTypeChecker
        plt.figure(figsize=size, dpi = dpi)
        plt.imshow(entrance_image_cut, cmap='gray')
        plt.scatter(com[1], com[0], color='r', s=0.5)
        plt.plot(entrance_mask_outline[:, 1], entrance_mask_outline[:, 0], 'r', linewidth=0.5)
        plt.title('Entrance Mask Overlay')
        plt.axis('off')
        plt.savefig(os.path.join(entrance_overlay_folder, entrance_mask_files[i].replace(".png", "_overlay.png")), dpi="figure")
        plt.close() # Plots are generally saved and closed, because code is run in threads which can cause issues.

    # Send progress signal
    if progress_signal:
        progress_signal.emit("Entrance Masks plotted. Continuing with exit masks")

    # Same for exit masks
    for i in range(len(exit_mask_files)):
        comk = exit_comk_median
        exit_mask = io.imread(os.path.join(exit_mask_folder, exit_mask_files[i]))
        exit_image = io.imread(os.path.join(main_folder, "exit/reduced", exit_mask_files[i].replace("_mask", "")))

        # Cut image to the fiber size
        exit_image_cut = cut_image_around_comk(exit_image, comk, fiber_exit_radius, margin)

        # Cut the mask to the fiber size
        exit_mask_cut = cut_image_around_comk(exit_mask, comk, fiber_exit_radius, margin)

        exit_mask_outline = measure.find_contours(exit_mask_cut, 0.5)[0]

        dpi = 100
        size = [exit_image_cut.shape[1] / dpi, exit_image_cut.shape[0] / dpi]

        # noinspection PyTypeChecker
        plt.figure(figsize=size, dpi=dpi)
        plt.imshow(exit_image_cut, cmap='gray')
        plt.plot(exit_mask_outline[:, 1], exit_mask_outline[:, 0], 'r', linewidth=0.5)
        plt.title('Exit Mask Overlay')
        plt.axis('off')
        plt.savefig(os.path.join(exit_overlay_folder, exit_mask_files[i].replace(".png", "_overlay.png")), dpi="figure")
        plt.close()
    if progress_signal:
        progress_signal.emit("Exit Masks plotted.")


def plot_coms(main_folder, progress_signal=None):
    """
    Plot the COMs and COMKs of the entrance and exit masks.
    Args:
        main_folder: Main folder of the fiber.
        progress_signal: Progress signal to update the progress.

    """
    if progress_signal:
        progress_signal.emit("Reading parameters from json")

    # Read from json file
    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    entrance_coms = parameters["entrance_coms"]
    exit_coms = parameters["exit_coms"]
    entrance_comk = parameters["entrance_comk"]
    exit_comk = parameters["exit_comk"]
    reference_index = parameters["reference_index"]

    # Convert lists to NumPy arrays
    entrance_coms = np.array(entrance_coms)
    exit_coms = np.array(exit_coms)
    entrance_comk = np.array(entrance_comk)
    exit_comk = np.array(exit_comk)

    # Plot entrance COMs and COMKs
    plt.scatter(entrance_coms[:, 1], entrance_coms[:, 0], label="COMs")
    for i, com in enumerate(entrance_coms):
        plt.text(com[1], com[0], str(i), fontsize=8, ha='right')
    plt.scatter(entrance_comk[:, 1], entrance_comk[:, 0], label="COMKs")
    plt.legend()
    plt.title("Entrance")
    plt.savefig(os.path.join(main_folder, "plots/Entrance_coms.png"))
    plt.close()

    plt.figure()
    plt.scatter(exit_coms[:, 1], exit_coms[:, 0], label="COMs")
    plt.scatter(exit_comk[:, 1], exit_comk[:, 0], label="COMKs")
    ref_com = exit_coms[reference_index]
    # noinspection PyTypeChecker
    plt.text(ref_com[1], ref_com[0], "Ref", fontsize=8, ha='right')
    ref_comk = exit_comk[reference_index]
    # noinspection PyTypeChecker
    plt.text(ref_comk[1], ref_comk[0], "Ref", fontsize=8, ha='right')
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.legend()
    plt.title("Exit")
    plt.savefig(os.path.join(main_folder, "plots/Exit_coms.png"))
    plt.close()

    # Send progress signal
    if progress_signal:
        progress_signal.emit("Plotting COMs done")


def plot_horizontal_cut_nf(project_folder):
    """
    Plot the horizontal cut of the image.
    Args:
        project_folder: Main folder of the project.

    """
    plots_folder = os.path.join(project_folder, "plots")

    # Define the image folders
    exit_image_folder = os.path.join(project_folder, "exit/reduced")

    # Get the image and mask files
    exit_image_files = [f for f in os.listdir(exit_image_folder) if f.endswith('.png')]

    # Read the first image and mask
    exit_image = io.imread(os.path.join(exit_image_folder, exit_image_files[0]))

    # Trim the image
    trimmed_data = core.data_processing.cut_image(exit_image, margin=200)

    # Locate center of mass within trimmed image (array)
    com = core.data_processing.locate_focus(trimmed_data)

    # Convert center of mass to integer indices
    com_x = int(com[0])

    # Create a horizontal cut through the center of mass
    cut = trimmed_data[com_x, :]

    # Plot the horizontal cut
    plt.figure()
    plt.plot(cut)
    plt.title(f"Horizontal cut through center of mass")
    plt.xlabel("Pixel")
    plt.ylabel("Flux")
    plt.grid(True)
    #plt.show()
    plt.savefig(plots_folder + f"/horizontal_cut.png")
    plt.close()


def plot_com_comk_on_image_cut(project_folder, progress_signal=None):
    # Define folders
    video_prep_folder = os.path.join(project_folder, "video_prep")
    entrance_folder = os.path.join(video_prep_folder, "entrance")
    exit_folder = os.path.join(video_prep_folder, "exit")
    plots_folder = os.path.join(project_folder, "plots")
    com_comk_on_image_entr_folder = os.path.join(plots_folder, "com_comk_on_image_entrance")
    com_comk_on_image_exit_folder = os.path.join(plots_folder, "com_comk_on_image_exit")

    # Create folders if they don't exist
    os.makedirs(com_comk_on_image_entr_folder, exist_ok=True)
    os.makedirs(com_comk_on_image_exit_folder, exist_ok=True)

    # Check if video_prep folder exists
    if not os.path.exists(video_prep_folder):
        if progress_signal:
            progress_signal.emit("Video prep folder does not exist, exiting...")
        print(f"Video prep folder does not exist: {video_prep_folder}")
        return

    # Read sg parameter json
    if progress_signal:
        progress_signal.emit("Reading parameters from json")
    with open(os.path.join(project_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    exit_radii = parameters["exit_radii"]
    entrance_coms_list = parameters["entrance_coms"]
    entrance_comk_list = parameters["entrance_comk"]
    exit_coms_list = parameters["exit_coms"]
    exit_comk_list = parameters["exit_comk"]
    entrance_coms_list = np.array(entrance_coms_list)
    entrance_comk_list = np.array(entrance_comk_list)
    exit_coms_list = np.array(exit_coms_list)
    exit_comk_list = np.array(exit_comk_list)

    # Get the first image to get the size
    entrance_image = io.imread(os.path.join(entrance_folder, f"entrance_cam_image000_reduced_cut.png"))
    entr_img_h_size = entrance_image.shape[0] // 2

    exit_image = io.imread(os.path.join(exit_folder, f"exit_cam_image000_reduced_cut.png"))
    exit_img_h_size = exit_image.shape[0] // 2

    # Calculate entrance com median
    entrance_coms_median = np.median(entrance_coms_list, axis=0)

    # Translate coms and comk to cut image system by referencing it to the int(comk) of the first image which then is the center of the cut image
    entrance_comk_list_cut = [tuple([entr_img_h_size, entr_img_h_size]) for _ in range(len(entrance_coms_list))]
    entrance_norm = entrance_comk_list - entrance_coms_median
    entrance_coms_list_cut = entrance_comk_list_cut - entrance_norm

    # Exit calculations
    exit_norm = exit_coms_list - np.array([int(exit_comk_list[0][0]), int(exit_comk_list[0][1])])

    # Calculate the maximum distance between COM and COMK
    distances = np.sqrt((exit_norm[:, 0] ** 2) + (exit_norm[:, 1] ** 2))
    max_distance = np.max(distances)

    # Define the target distance for visualization
    exit_radii = np.array(exit_radii)
    exit_radii_mean = np.mean(exit_radii)
    target_distance = exit_radii_mean * 0.8

    # Compute the visualization factor
    visualization_factor = int(max(target_distance / max_distance, 1))
    print(f"Target distance: {target_distance}")
    print(f"Max distance: {max_distance}")
    print(f"Visualization factor: {visualization_factor}")

    exit_coms_list_cut = exit_norm * visualization_factor + np.array([exit_img_h_size, exit_img_h_size])
    exit_comk_list_cut = exit_comk_list - np.array([int(exit_comk_list[0][0]), int(exit_comk_list[0][1])]) + np.array([exit_img_h_size, exit_img_h_size])

    # Calculate median of exit comks
    exit_comk_median_cut = np.median(exit_comk_list_cut, axis=0)

    # Plot entrance COMs and COMKs
    if progress_signal:
        progress_signal.emit("Starting plots")
    for i in range(len(entrance_coms_list)):
        entrance_image = io.imread(os.path.join(entrance_folder, f"entrance_cam_image{i:03d}_reduced_cut.png"))
        entrance_com = entrance_coms_list_cut[i]
        entrance_comk = entrance_comk_list_cut[i]

        # Draw a line between the COM and COMK
        plt.plot([entrance_com[1], entrance_comk[1]], [entrance_com[0], entrance_comk[0]], color='blue',
                 linestyle='--', linewidth=0.5, label='Distance')

        # Annotate the distance between the dots
        distance = np.sqrt(entrance_norm[i][0] ** 2 + entrance_norm[i][1] ** 2)
        plt.text((entrance_com[1] + entrance_comk[1]) / 2, (entrance_com[0] + entrance_comk[0]) / 2, f"{distance:.2f} px",
                 color='blue', fontsize=6)

        # Plot the image with the COM and COMK
        plt.imshow(entrance_image, cmap='gray')
        plt.scatter(entrance_com[1], entrance_com[0], color='r', s=10, label='Center of Mass')
        plt.scatter(entrance_comk[1], entrance_comk[0], color='g', s=10, label='Center of Mask')
        plt.title(f"Entrance Image {i}")
        plt.axis('off')
        plt.legend(loc='upper right', fontsize=6)
        plt.savefig(os.path.join(com_comk_on_image_entr_folder, f"entrance_image_{i}_com.png"), dpi="figure")
        plt.close()

    # Plot exit COMs and COMKs
    for i in range(len(exit_coms_list)):
        exit_image = io.imread(os.path.join(exit_folder, f"exit_cam_image{i:03d}_reduced_cut.png"))
        exit_com = exit_coms_list_cut[i]
        exit_comk = exit_comk_median_cut

        # Plot the image with the COM and COMK
        plt.imshow(exit_image, cmap='gray')
        plt.scatter(exit_com[1], exit_com[0], color='r', s=10, label='Center of Mass')
        plt.scatter(exit_comk[1], exit_comk[0], color='g', s=10, label='Center of Mask')

        # Draw a line between the COM and COMK
        plt.plot([exit_com[1], exit_comk[1]], [exit_com[0], exit_comk[0]], color='blue', linestyle='--', linewidth=0.5,
                 label=f'Distance (visually exaggerated by a factor of {visualization_factor})')

        # Annotate the distance between the dots
        distance = np.sqrt(exit_norm[i][0] ** 2 + exit_norm[i][1] ** 2)
        plt.text((exit_com[1] + exit_comk[1]) / 2, (exit_com[0] + exit_comk[0]) / 2, f"{distance:.2f} (* {visualization_factor}) px", color='blue',
                 fontsize=6)

        plt.title(f"Exit Image {i}")
        plt.axis('off')
        plt.legend(loc='upper right', fontsize=6)
        plt.savefig(os.path.join(com_comk_on_image_exit_folder, f"exit_image_{i}_com.png"), dpi="figure")
        plt.close()
