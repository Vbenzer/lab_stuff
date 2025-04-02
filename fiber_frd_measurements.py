import numpy as np
import json
import matplotlib.pyplot as plt
import file_save_managment
import analyse_main
from astropy.io import fits
import os

fw = None
cam = None

def init_camera(exp_time:int):
    import qhy_ccd_take_image
    global cam
    cam = qhy_ccd_take_image.Camera(exp_time=exp_time)

def init_filter_wheel():
    global fw
    import qhycfw3_filter_wheel_control
    fw = qhycfw3_filter_wheel_control.FilterWheel('COM5')

def nf_ff_capture(project_folder:str, fiber_diameter:[int, tuple[int,int]], exposure_times:dict[str, str]=None,
                         progress_signal=None):
    import step_motor_control as smc
    import move_to_filter as mtf
    import qhy_ccd_take_image
    import qhycfw3_filter_wheel_control
    import thorlabs_cam_control as tcc
    import threading

    if exposure_times is None:
        raise ValueError("Exposure times must be provided.")

    # Create and check project folder
    os.makedirs(project_folder)

    # Connect filter wheel and cameras in thread
    fw_thread = threading.Thread(target=init_filter_wheel)
    cam_thread = threading.Thread(target=init_camera, args=([exposure_times["exit_cam"]]))

    fw_thread.start()
    cam_thread.start()

    # Make sure everything is ready
    fw_thread.join()
    cam_thread.join()

    # Set filter wheel to f/3.5
    fw.move_to_filter("3.5")

    # Define folders
    entrance_folder = os.path.join(project_folder, "entrance")
    exit_folder = os.path.join(project_folder, "exit")
    os.makedirs(entrance_folder, exist_ok=True)
    os.makedirs(exit_folder, exist_ok=True)

    entrance_folder_light = os.path.join(entrance_folder, "light")
    exit_folder_light = os.path.join(exit_folder, "light")
    os.makedirs(entrance_folder_light, exist_ok=True)
    os.makedirs(exit_folder_light, exist_ok=True)

    entrance_folder_dark = os.path.join(entrance_folder, "dark")
    exit_folder_dark = os.path.join(exit_folder, "dark")
    os.makedirs(entrance_folder_dark, exist_ok=True)
    os.makedirs(exit_folder_dark, exist_ok=True)

    entrance_folder_reduced = os.path.join(entrance_folder, "reduced")
    exit_folder_reduced = os.path.join(exit_folder, "reduced")
    os.makedirs(entrance_folder_reduced, exist_ok=True)
    os.makedirs(exit_folder_reduced, exist_ok=True)


    number_of_positions = 11
    # Calculate the step size and leftmost position. Also handle rectangular case
    if isinstance(fiber_diameter, (tuple, list)):
        max_size = max(fiber_diameter)
        step_size = max_size / 1000 * 0.8 / (number_of_positions - 1)  # Step size in mm
        pos_left = 5 - max_size / 1000 * 0.8 / 2  # Leftmost position in mm
    else:
        step_size = fiber_diameter / 1000 * 0.8 / (number_of_positions - 1)  # Step size in mm
        pos_left = 5 - fiber_diameter / 1000 * 0.8 / 2  # Leftmost position in mm

    # Take images
    for i in range(number_of_positions):
        print("Taking image:", i, ", at position:", pos_left + i * step_size)

        # Move the motor to the next position
        smc.move_motor_to_position(pos_left + i * step_size)

        # Take darks
        mtf.move("Closed")
        tcc.take_image("entrance_cam", entrance_folder_dark + f"/entrance_cam_dark{i:03d}.fits",
                   exposure_time=exposure_times["entrance_cam"], save_fits=True)
        cam.take_single_frame(exit_folder_dark, f"exit_cam_dark{i:03d}.fits")


        # Take images
        mtf.move("Open")
        tcc.take_image("entrance_cam", entrance_folder_light + f"/entrance_cam_image{i:03d}.fits",
                       exposure_time=exposure_times["entrance_cam"], save_fits=True)
        cam.take_single_frame(exit_folder_light, f"exit_cam_image{i:03d}.fits")
    smc.move_motor_to_position(5)
    print("All images taken!")

def nf_ff_process(project_folder:str, fiber_diameter:[int, tuple[int,int]], progress_signal=None):
    import image_reduction as ir
    import image_analysation as ia
    # Define folders
    entrance_folder = os.path.join(project_folder, "entrance")
    exit_folder = os.path.join(project_folder, "exit")

    entrance_folder_light = os.path.join(entrance_folder, "light")
    exit_folder_light = os.path.join(exit_folder, "light")

    entrance_folder_dark = os.path.join(entrance_folder, "dark")
    exit_folder_dark = os.path.join(exit_folder, "dark")

    entrance_folder_reduced = os.path.join(entrance_folder, "reduced")
    exit_folder_reduced = os.path.join(exit_folder, "reduced")
    os.makedirs(entrance_folder_reduced, exist_ok=True)
    os.makedirs(exit_folder_reduced, exist_ok=True)

    entrance_folder_cut = os.path.join(entrance_folder, "cut")
    exit_folder_cut = os.path.join(exit_folder, "cut")
    os.makedirs(entrance_folder_cut, exist_ok=True)
    os.makedirs(exit_folder_cut, exist_ok=True)

    # Get the list of light and dark images
    entrance_light_images = sorted(os.listdir(entrance_folder_light))
    exit_light_images = sorted(os.listdir(exit_folder_light))

    entrance_dark_images = sorted(os.listdir(entrance_folder_dark))
    exit_dark_images = sorted(os.listdir(exit_folder_dark))

    # Reduce the images
    for i in range(len(entrance_light_images)):
        print("Reducing image", i)
        output_file_path_entrance = os.path.join(entrance_folder_reduced, f"entrance_cam_reduced{i:03d}.fits")
        # Load the entrance light and dark images
        with fits.open(os.path.join(entrance_folder_light, entrance_light_images[i])) as hdul:
            entrance_light_data = hdul[0].data.astype(np.float32)

        with fits.open(os.path.join(entrance_folder_dark, entrance_dark_images[i])) as hdul:
            entrance_dark_data = hdul[0].data.astype(np.float32)

        ir.reduce_image_with_dark(entrance_light_data, entrance_dark_data, output_file_path_entrance, save=True)

        # Load the exit light and dark images
        with fits.open(os.path.join(exit_folder_light, exit_light_images[i])) as hdul:
            exit_light_data = hdul[0].data.astype(np.float32)

        with fits.open(os.path.join(exit_folder_dark, exit_dark_images[i])) as hdul:
            exit_dark_data = hdul[0].data.astype(np.float32)

        output_file_path_exit = os.path.join(exit_folder_reduced, f"exit_cam_reduced{i:03d}.fits")
        ir.reduce_image_with_dark(exit_light_data, exit_dark_data, output_file_path_exit, save=True)
    print("All images reduced!")

    # Get list of reduced images
    entrance_reduced_images = sorted(os.listdir(entrance_folder_reduced))
    exit_reduced_images = sorted(os.listdir(exit_folder_reduced))

    # Calculate the radius of the fiber in pixels, also handle rectangular case
    if isinstance(fiber_diameter, (tuple, list)):
        fiber_input_radius = (
        int(fiber_diameter[0] / 0.5169363821005045 / 2), int(fiber_diameter[1] / 0.5169363821005045 / 2))
    else:
        fiber_input_radius = int(fiber_diameter / 0.5169363821005045 / 2)

    # Cut the images to size
    for i in range(len(entrance_light_images)):
        print("Cutting image", i)
        # Cut entrance images
        from sg_pipeline import cut_image_around_comk

        # Load the reduced entrance images
        with fits.open(os.path.join(entrance_folder_reduced, entrance_reduced_images[i])) as hdul:
            entrance_reduced_data = hdul[0].data.astype(np.float32)

        com = ia.LocateFocus(entrance_reduced_data)
        cut_image = cut_image_around_comk(entrance_reduced_data, com, fiber_input_radius, margin=50)
        # Save the cut image as png
        plt.imshow(cut_image, cmap='gray', origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(entrance_folder_cut, f"entrance_cam_cut{i:03d}.png"))
        plt.close()

        # Cut exit images
        # Load the reduced exit images
        with fits.open(os.path.join(exit_folder_reduced, exit_reduced_images[i])) as hdul:
            exit_reduced_data = hdul[0].data.astype(np.float32)
        trimmed_data = ia.cut_image(exit_reduced_data, margin=500)
        plt.imshow(trimmed_data, cmap='gray', origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(exit_folder_cut, f"exit_cam_cut{i:03d}.png"))
        plt.close()

    print("All images reduced and cut!")





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
    input_f_num = np.array(
        [6.152, 5.103, 4.57, 4.089, 3.578])  # These are from the setup_F#_EE_98_Measurement_2 file, 31.3.25
    input_f_num_err = np.array([0.003, 0.007, 0.03, 0.01, 0.021])

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
        [6.152, 5.103, 4.57, 4.089, 3.578])  # These are from the setup_F#_EE_98_Measurement_2 file, 31.3.25
    input_f_num_err = np.array([0.003, 0.007, 0.03, 0.01, 0.021])

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

        import image_analysation as ia
        # Trim data to area of interest
        trimmed_data = ia.cut_image(reduced_image, margin=1000)  # If margin too low mask won't fit in image

        # Locate center of mass within trimmed image (array)
        com = ia.LocateFocus(trimmed_data)

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
            import sg_pipeline
            mask = sg_pipeline.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius, plot_mask=False)

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
        import sg_pipeline
        import image_analysation as ia
        from skimage import measure
        from matplotlib.colors import LogNorm

        # Input f-numbers
        input_f_num = np.array(
            [6.152, 5.103, 4.57, 4.089, 3.578])  # These are from the setup_F#_EE_98_Measurement_2 file, 31.3.25
        input_f_num_err = np.array([0.003, 0.007, 0.03, 0.01, 0.021])

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
            aperture_radius_NA = (distance_to_chip) * NA / np.sqrt(1 - NA ** 2)  #+ 9.9: Distance to chip at 0 position (old setup)
            aperture_radius_NA = aperture_radius_NA // 7.52e-3
            mask_NA = sg_pipeline.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius_NA, plot_mask=False)
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
                mask = sg_pipeline.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius, plot_mask=False)

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
    project_folder = "D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5_measurement_2/NF_FF"
    #sutherland_plot(project_folder)
    #plot_f_ratio_circles_on_raw(project_folder)
    nf_ff_capture(project_folder, 28, {"entrance_cam": "10ms", "exit_cam": 25000})
    nf_ff_process(project_folder, 28)