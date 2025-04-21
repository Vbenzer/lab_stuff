import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from skimage import io, transform

import core.data_processing
import qhy_ccd_take_image

import thorlabs_cam_control as tcc


def measure_fiber_size(project_folder:str, exposure_times:dict[str, str]=None):
    if exposure_times is None:
        raise ValueError("Exposure times must be provided.")

    os.makedirs(project_folder, exist_ok=True)

    # Take entrance image
    tcc.take_image("exit_cam", project_folder + "/exit_cam_image.fits",
                   exposure_time=exposure_times["exit_cam"], save_fits=True)

    # Load the image
    exit_image = fits.open(project_folder + "/exit_cam_image.fits")[0].data.astype(np.float32)

    # Get fiber dimension
    fiber_data = core.data_processing.measure_fiber_dimensions(exit_image)

    print(fiber_data[0]["dimensions_mu"])


fw = None
cam = None


def init_camera(exp_time:int):
    import qhy_ccd_take_image
    global cam
    cam = qhy_ccd_take_image.Camera(exp_time=exp_time)


def init_filter_wheel():
    global fw
    fw = qhycfw3_filter_wheel_control.FilterWheel('COM5')


def nf_ff_capture(project_folder:str, fiber_diameter:[int, tuple[int,int]], exposure_times:dict[str, str]=None,
                         progress_signal=None):
    from core.hardware import motor_control as smc
    from core.hardware import filter_wheel_color as mtf
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

        core.data_processing.reduce_image_with_dark(entrance_light_data, entrance_dark_data, output_file_path_entrance, save=True)

        # Load the exit light and dark images
        with fits.open(os.path.join(exit_folder_light, exit_light_images[i])) as hdul:
            exit_light_data = hdul[0].data.astype(np.float32)

        with fits.open(os.path.join(exit_folder_dark, exit_dark_images[i])) as hdul:
            exit_dark_data = hdul[0].data.astype(np.float32)

        output_file_path_exit = os.path.join(exit_folder_reduced, f"exit_cam_reduced{i:03d}.fits")
        core.data_processing.reduce_image_with_dark(exit_light_data, exit_dark_data, output_file_path_exit, save=True)
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
        from analysis.sg_analysis import cut_image_around_comk

        # Load the reduced entrance images
        with fits.open(os.path.join(entrance_folder_reduced, entrance_reduced_images[i])) as hdul:
            entrance_reduced_data = hdul[0].data.astype(np.float32)

        com = core.data_processing.locate_focus(entrance_reduced_data)
        cut_image = cut_image_around_comk(entrance_reduced_data, com, fiber_input_radius, margin=50)
        # Save the cut image as png
        cut_image = (cut_image - np.min(cut_image)) / (np.max(cut_image) - np.min(cut_image)) * 255
        io.imsave(os.path.join(entrance_folder_cut, f"entrance_cam_cut{i:03d}.png"), cut_image.astype(np.uint8))

        # Cut exit images
        # Load the reduced exit images
        with fits.open(os.path.join(exit_folder_reduced, exit_reduced_images[i])) as hdul:
            exit_reduced_data = hdul[0].data.astype(np.float32)
        trimmed_data = core.data_processing.cut_image(exit_reduced_data, margin=500)

        # Raise values so that minimum is 0
        trimmed_data = (trimmed_data - np.min(trimmed_data)) / (np.max(trimmed_data) - np.min(trimmed_data)) * 255
        # Resize the image to 1/10
        trimmed_data = transform.resize(trimmed_data, (trimmed_data.shape[0] // 10, trimmed_data.shape[1] // 10),
                                        anti_aliasing=True)


        io.imsave(os.path.join(exit_folder_cut, f"exit_cam_cut{i:03d}.png"), trimmed_data.astype(np.uint8))

    print("All images reduced and cut!")


def get_ff_with_all_filters(working_directory):
    """
    Takes and reduces one far field image for each color filter
    Args:
        working_directory: Working directory where the images will be saved

    Returns:

    """
    # Define Folders
    dark_folder = os.path.join(working_directory, "darks")
    science_folder = os.path.join(working_directory, "science")
    reduced_folder = os.path.join(working_directory, "reduced")

    # Create folders if they do not exist
    os.makedirs(dark_folder, exist_ok=False)
    os.makedirs(science_folder, exist_ok=False)
    os.makedirs(reduced_folder, exist_ok=False)

    # Define filters
    filters = ["400", "450", "500", "600", "700", "800"]

    # Initialize camera
    camera = qhy_ccd_take_image.Camera(exp_time=2000000)

    # Go to block filter and take darks
    move_to_filter.move("none")
    camera.take_multiple_frames(dark_folder, "dark", num_frames=5)

    for filter_name in filters:
        move_to_filter.move(filter_name)
        camera.take_single_frame(science_folder, filter_name)

    # Close camera
    camera.close()

    # Return to no filter
    move_to_filter.move("0")

    # Create master dark
    mdark = core.data_processing.create_master_dark(dark_folder)

    # Reduce all science images
    for image in os.listdir(science_folder):
        if image.endswith(".fits"):
            image_path = os.path.join(science_folder, image)
            with fits.open(image_path) as hdul:
                science_data = hdul[0].data.astype(np.float32)
                reduced_path = os.path.join(reduced_folder, image)
                core.data_processing.reduce_image_with_dark(science_data, mdark, output_file=reduced_path, save=True)


def inf_fiber_original_tp_plot(csv_file:str, show=False):
    """
    Created for infrared fiber throughput. Returns data from original datasheet, which was extracted to csv file
    manually beforehand.
    Args:
        csv_file: Path to the csv file containing the original data.
        show: Boolean flag to show the plot or not.

    Returns: list1, throughput
        list1: Wavelengths from the csv file.
        throughput: Throughput values calculated from the csv file data.

    """
    import pandas as pd
    # Read data from csv file
    print(csv_file)
    data = pd.read_csv(csv_file, header=None)
    list1 = data[0].tolist()
    print(list1)
    #list1 = [float(s.strip(',')) for s in list1]
    list2 = data[1].tolist()
    print(list2)

    if show:
        plt.figure()
        plt.plot(list1, list2, label="Original")
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Throughput")
        plt.show()

    # Convert attenuation to throughput
    fiber_length = 0.005  # km
    attenuation = np.array(list2) * fiber_length
    throughput = 10**(-attenuation/10)


    plt.figure()
    plt.plot(list1, throughput, label="Original in Throughput")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Throughput")
    plt.savefig("D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/Throughput/throughput_original.png")
    if show:
        plt.show()
    plt.close()



    return list1, throughput


def overplot_original_and_data(data:str, original_data:str, show=False):
    """
    Compares the original throughput data with the measured data from the json file.
    Args:
        data: Path to the json file containing the measured data.
        original_data: Path to the csv file containing the original data.
        show: Boolean flag to show the intermediate plot of the original data or not.

    Returns:

    """
    import json
    wavelength_or, tp_or = inf_fiber_original_tp_plot(original_data, show)

    # get other data from json file
    with open(data) as f:
        data = json.load(f)

    filters = ["400", "450", "500", "600", "700", "800"]
    wavelength = []
    throughput = []
    for filter in filters:
        wavelength.append(int(filter)*1e-3)
        throughput.append(data[filter])

    plt.figure()
    plt.plot(wavelength_or, tp_or, label="Original")

    # Column plot of measured data
    plt.bar(wavelength, throughput,width=0.05, label="Data")

    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Throughput")
    plt.legend()
    plt.savefig("D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/Throughput/throughput_comparison.png")
    plt.close()
