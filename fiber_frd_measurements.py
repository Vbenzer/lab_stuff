import numpy as np
import json
import matplotlib.pyplot as plt
import file_save_managment
import analyse_main
from astropy.io import fits
import os

def main_measure_all_filters(project_folder:str, progress_signal=None, base_directory=None):
    """
    Run the measuring pipeline for all filters.
    Args:
        project_folder: Path to the project folder.
        progress_signal: Progress signal.
        base_directory: Base directory of the project.

    Returns:

    """
    for i in range(2, 7):
        # Create project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        progress_signal.emit(f"Starting measurement for filter {i}")

        # Run the main measuring pipeline for each filter
        analyse_main.main_measure(filter_folder, progress_signal,
                                  batch_file_path=f"D:\\stepper_motor\\start_nina_with_fstop_filter{i}.bat",
                                  base_directory=base_directory)

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

    folder_list = [folder for folder in sorted(os.listdir(project_folder)) if "filter" in folder]
    for i, folder in enumerate(folder_list):
        # Define project subfolder for each filter
        filter_folder = os.path.join(project_folder, folder)

        if progress_signal:
            progress_signal.emit(f"Starting analysis for: {folder}")

        analyse_main.run_from_existing_files(filter_folder, progress_signal)

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

    #file_save_managment.save_measurement_hdf5("D:/Vincent/frd_measurements.h5", measurement_name, f_num, f_num_err)

def plot_main(project_folder:str):
    # Load the output f-numbers from the JSON file
    with open(project_folder + "/f_number.json") as f:
        data = json.load(f)
        f_num = np.array(data["f_number"])
        f_num_err = np.array(data["f_number_err"])

    # Input f-numbers
    input_f_num = np.array([6.21, 5.103, 4.571, 4.063, 3.597])  # These are from the setup_F#_EE_98 file, 18.2.25
    input_f_num_err = np.array([0.04, 0.007, 0.01, 0.005, 0.013])

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
    input_f_num = np.array([6.21, 5.103, 4.571, 4.063, 3.597]) # These are from the setup_F#_EE_98 file, 18.2.25
    input_f_num_err = np.array([0.04, 0.007, 0.01, 0.005, 0.013])

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
    print("Weights:", weights)
    distance_to_chip = np.sum(dist * weights) / np.sum(weights) # Weighted mean
    print("Distance to chip:", distance_to_chip)
    sigma_weighted_mean = np.sqrt(1 / np.sum(weights))
    print("Sigma weighted mean:", sigma_weighted_mean)
    std_dev = np.std(dist, ddof=1)
    print("Standard deviation:", std_dev)
    distance_to_chip_err = np.sqrt(sigma_weighted_mean**2 + std_dev**2)

    print(distance_to_chip_err)

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

        import image_analysation as ia
        # Trim data to area of interest
        trimmed_data = ia.cut_image(reduced_image, margin=1000)  # If margin too low mask won't fit in image

        # Locate center of mass within trimmed image (array)
        com = ia.LocateFocus(trimmed_data)

        ee_sublist = []
        ee_err_sublist = []

        for fnum, fnum_err in zip(input_f_num, input_f_num_err):
            # Calculate the radius of a circle with input f-ratios
            aperture_radius = (distance_to_chip + 9.9) / (2 * fnum)  # 9.9: Distance to chip at 0 position
            aperture_radius_err = aperture_radius * np.sqrt(
                (distance_to_chip_err / (distance_to_chip + 9.9)) ** 2 + (fnum_err / fnum) ** 2)

            # Convert to pixels (keep floating-point for error calculations)
            aperture_radius_pix = aperture_radius / 7.52e-3
            aperture_radius_err_pix = aperture_radius_err / 7.52e-3

            print("aperture_err", aperture_radius_err_pix)

            # Round only for actual mask application
            aperture_radius = int(round(aperture_radius_pix))

            # Create a circle mask
            import sg_pipeline
            mask = sg_pipeline.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius, plot_mask=True)

            # Calculate the flux within the mask
            flux = np.sum(mask * trimmed_data)

            # Calculate the flux outside the mask
            mask_outside = np.invert(mask)
            flux_outside = np.sum(mask_outside * trimmed_data)

            # Calculate the encircled energy of the mask
            ee = flux / (flux + flux_outside)

            # Compute uncertainties (assuming Poisson statistics for flux)
            flux_err = np.sqrt(flux) if flux > 0 else 0  # Avoid division by zero
            flux_outside_err = np.sqrt(flux_outside) if flux_outside > 0 else 0  # Avoid division by zero

            print("Flux errors", flux_err, flux_outside_err)

            # Compute EE uncertainty
            ee_err = ee * np.sqrt(
                (flux_err / flux) ** 2 +
                (flux_outside_err / flux_outside) ** 2 +
                (aperture_radius_err_pix / aperture_radius_pix) ** 2
            )

            print("ee, ee_err:", ee, ee_err)
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
        plt.vlines(input_f_num[4-idx], ee[4-idx], 1, color=colors[idx % len(colors)], linestyle='--', linewidth=0.5)

        # Add text to the plot
        alignment = "center" #'right' if idx == 0 else 'left'
        padding = 0 #-0.05 if idx == 0 else 0.05
        plt.text(input_f_num[4 - idx] + padding, 1, f"{ee[4 - idx]:.3f}", color=colors[idx % len(colors)],
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
        import sg_pipeline
        import image_analysation as ia
        from skimage import measure
        from matplotlib.colors import LogNorm

        # Input f-numbers
        input_f_num = np.array([6.21, 5.103, 4.571, 4.063, 3.597])  # These are from the setup_F#_EE_98 file, 18.2.25
        input_f_num_err = np.array([0.04, 0.007, 0.01, 0.005, 0.013])

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
            trimmed_data = ia.cut_image(reduced_image, margin=1000)  # If margin too low mask won't fit in image

            # Locate center of mass within trimmed image (array)
            com = ia.LocateFocus(trimmed_data)

            # Calculate radius of NA
            aperture_radius_NA = (distance_to_chip + 9.9) * NA / np.sqrt(1 - NA ** 2)  # 9.9: Distance to chip at 0 position
            aperture_radius_NA = aperture_radius_NA // 7.52e-3
            mask_NA = sg_pipeline.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius_NA, plot_mask=False)
            mask_outline_NA = measure.find_contours(mask_NA, 0.5)[0]

            mask_outline_list = []
            filter_name = filter_to_name_dict[str(i+2)]

            for fnum in input_f_num:
                print(f"Processing f/{filter_name}, f/{fnum}")
                # Calculate the radius of a circle with input f-ratios
                aperture_radius = (distance_to_chip + 9.9) / (2 * fnum)  # 9.9: Distance to chip at 0 position

                # Convert to pixels
                aperture_radius = aperture_radius // 7.52e-3

                # Create a circle mask
                mask = sg_pipeline.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius, plot_mask=False)

                mask_outline_list.append(measure.find_contours(mask, 0.5)[0])

            """# Boost everything so that the lowest value is 0 for log scaling
            trimmed_data = trimmed_data - np.min(trimmed_data)
            print(np.min(trimmed_data))"""

            # Save trimmed as fits
            hdu = fits.PrimaryHDU(trimmed_data)
            hdu.writeto(f_ratio_images_folder + f"/trimmed_{filter_name}.fits", overwrite=True)

            # Plot the mask on the raw image
            plt.figure()
            plt.title(f"Input f/{filter_name} with artificial apertures")
            plt.imshow(trimmed_data, cmap='gray')#, norm=LogNorm())
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


def plot_horizontal_cut(project_folder):
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
        import image_analysation as ia
        trimmed_data = ia.cut_image(reduced_image, margin=500)

        # Locate center of mass within trimmed image (array)
        com = ia.LocateFocus(trimmed_data)

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



if __name__ == "__main__":
    project_folder = "/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/O_50_0000_0000/FRD"
    #sutherland_plot(project_folder)
    plot_f_ratio_circles_on_raw(project_folder)