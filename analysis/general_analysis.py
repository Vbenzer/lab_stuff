"""Module general_analysis.py.

Auto-generated docstring for better readability.
"""
import os
import threading

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from skimage import io, transform

import core.data_processing
import core.hardware.filter_wheel_fratio as qhycfw3_filter_wheel_control
import core.hardware.cameras.qhyccd_control as qhy_ccd_take_image

from core.hardware.cameras import thorlabs_cam_control as tcc


def measure_fiber_size(project_folder: str, exposure_times: dict[str, str] = None, progress_signal=None):
    if exposure_times is None:
        raise ValueError("Exposure times must be provided.")

    os.makedirs(project_folder, exist_ok=True)

    if progress_signal:
        progress_signal.emit("Taking image with entrance camera")

    # Take entrance image
    tcc.take_image("exit_cam", project_folder + "/exit_cam_image.fits", exposure_time=exposure_times["exit_cam"],
                   save_fits=True)

    if progress_signal:
        progress_signal.emit("Loading image")

    # Load the image
    exit_image = fits.open(project_folder + "/exit_cam_image.fits")[0].data.astype(np.float32)

    if progress_signal:
        progress_signal.emit("Processing image")

    # Get fiber dimension
    fiber_data = core.data_processing.measure_fiber_dimensions(exit_image)

    print(fiber_data[0]["dimensions_mu"])


fw = None
cam = None


def init_camera(exp_time:int):
    global cam
    cam = qhy_ccd_take_image.Camera(exp_time=exp_time)


def init_filter_wheel():
    global fw
    fw = qhycfw3_filter_wheel_control.FilterWheel('COM5')


def _initialize_devices(exit_cam_exp: int, progress_signal=None) -> None:
    """Initialize camera and filter wheel in parallel."""
    if progress_signal:
        progress_signal.emit("Initializing filter wheel and cameras")

    fw_thread = threading.Thread(target=init_filter_wheel)
    cam_thread = threading.Thread(target=init_camera, args=(exit_cam_exp,))

    fw_thread.start()
    cam_thread.start()
    fw_thread.join()
    cam_thread.join()


def _create_capture_folders(project_folder: str) -> dict[str, str]:
    """Create and return all subfolders used during capture."""
    entrance_folder = os.path.join(project_folder, "entrance")
    exit_folder = os.path.join(project_folder, "exit")
    os.makedirs(entrance_folder, exist_ok=True)
    os.makedirs(exit_folder, exist_ok=True)

    folders = {
        "entrance_light": os.path.join(entrance_folder, "light"),
        "exit_light": os.path.join(exit_folder, "light"),
        "entrance_dark": os.path.join(entrance_folder, "dark"),
        "exit_dark": os.path.join(exit_folder, "dark"),
        "entrance_reduced": os.path.join(entrance_folder, "reduced"),
        "exit_reduced": os.path.join(exit_folder, "reduced"),
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders


def _capture_position(index: int, position: float, exposure_times: dict[str, int],
                      folders: dict[str, str], progress_signal=None) -> None:
    """Capture dark and light frames for a single motor position."""
    from core.hardware import motor_control as smc
    from core.hardware import filter_wheel_color as mtf

    if progress_signal:
        progress_signal.emit(f"Moving Motor to position {position}")
    smc.move_motor_to_position(position)

    if progress_signal:
        progress_signal.emit("Closing Shutter")
    mtf.move("Closed")
    if progress_signal:
        progress_signal.emit("Taking darks")

    tcc.take_image(
        "entrance_cam",
        os.path.join(folders["entrance_dark"], f"entrance_cam_dark{index:03d}.fits"),
        exposure_time=exposure_times["entrance_cam"],
        save_fits=True,
    )
    cam.take_single_frame(folders["exit_dark"], f"exit_cam_dark{index:03d}.fits")

    if progress_signal:
        progress_signal.emit("Opening Shutter")
    mtf.move("Open")
    if progress_signal:
        progress_signal.emit("Taking images")

    tcc.take_image(
        "entrance_cam",
        os.path.join(folders["entrance_light"], f"entrance_cam_image{index:03d}.fits"),
        exposure_time=exposure_times["entrance_cam"],
        save_fits=True,
    )
    cam.take_single_frame(folders["exit_light"], f"exit_cam_image{index:03d}.fits")


def nf_ff_capture(project_folder: str, fiber_diameter: [int, tuple[int, int]], exposure_times: dict[str, int] | None = None,
                  progress_signal=None, driving_width: float | None = None, number_of_positions: int = 11) -> None:
    """
    Takes far field and near field images with the entrance and exit cameras. The images are taken at different
    positions of the input spot.
    Args:
        project_folder: Project folder where the images will be saved
        fiber_diameter: Fiber diameter in microns. If a tuple is provided, the first element is the width and the
                        second element is the height.
        exposure_times: Exposure times for the cameras in seconds. The keys are "entrance_cam" and "exit_cam".
        progress_signal: Signal to update the progress of the process. If None, no progress signal is emitted.
        driving_width: Distance to be set between first and last image. If None, the fiber diameter is used.
                        When analyzing effects of outer layers carrying light, this should be set.
        number_of_positions: Number of positions to take images at. Default is 11.

    Returns:

    """
    if exposure_times is None:
        raise ValueError("Exposure times must be provided.")

    os.makedirs(project_folder, exist_ok=True)

    _initialize_devices(exposure_times["exit_cam"], progress_signal)

    if progress_signal:
        progress_signal.emit("Moving to initial position (f/3.5)")

    fw.move_to_filter("3.5")

    folders = _create_capture_folders(project_folder)

    if driving_width is not None:
        fiber_diameter = driving_width

    if isinstance(fiber_diameter, (tuple, list)):
        max_size = max(fiber_diameter)
    else:
        max_size = fiber_diameter
    step_size = max_size / 1000 * 0.8 / (number_of_positions - 1)
    pos_left = 5 - max_size / 1000 * 0.8 / 2

    for i in range(number_of_positions):
        if progress_signal:
            progress_signal.emit(f"Starting image process {i + 1} of {number_of_positions}")

        current_pos = pos_left + i * step_size
        print("Taking image:", i, ", at position:", current_pos)

        _capture_position(i, current_pos, exposure_times, folders, progress_signal)

        if progress_signal:
            progress_signal.emit(f"Image process {i + 1} of {number_of_positions} done")

    if progress_signal:
        progress_signal.emit("Resetting motor position to 5mm")

    from core.hardware import motor_control as smc
    smc.move_motor_to_position(5)
    print("All images taken!")


def nf_ff_process(project_folder:str, fiber_diameter:[int, tuple[int,int]], progress_signal=None, output_scale:str="lin"):
    """
    Processes the images taken with the entrance and exit cameras. The images are reduced and cut to size.
    Args:
        project_folder: Folder where the images are saved
        fiber_diameter: Diameter of the fiber in microns. If a tuple is provided, the first element is the width and the second
                        element is the height.
        progress_signal: Signal to update the progress of the process. If None, no progress signal is emitted.
        output_scale: Scale of the output images. Can be either "lin" or "log". Default is "lin".

    Returns:

    """
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

        if progress_signal:
            progress_signal.emit(f"Reducing images {i + 1} of {len(entrance_light_images)}")

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

    if progress_signal:
        progress_signal.emit("Image reduction done")

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

        if progress_signal:
            progress_signal.emit(f"Cutting images {i + 1} of {len(entrance_light_images)}")

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
        if output_scale == "lin":
            scaled_data = trimmed_data
        elif output_scale == "log":
            # Scale the data to 0-255 using logarithmic scaling
            scaled_data = (np.log1p(trimmed_data) / np.log1p(np.max(trimmed_data)) * 255)
        else:
            print("Output scale must be either 'lin' or 'log'")
            scaled_data = trimmed_data

        io.imsave(os.path.join(exit_folder_cut, f"exit_cam_cut{i:03d}.png"), scaled_data.astype(np.uint8))

    print("All images reduced and cut!")

    if progress_signal:
        progress_signal.emit("Image cutting done")


def get_ff_with_all_filters(working_directory, progress_signal=None):
    """
    Takes and reduces one far field image for each color filter
    Args:
        progress_signal:
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
    if progress_signal:
        progress_signal("Initializing camera")

    camera = qhy_ccd_take_image.Camera(exp_time=2000000)

    # Go to block filter and take darks
    if progress_signal:
        progress_signal("Closing Shutter")
    move_to_filter.move("none")

    if progress_signal:
        progress_signal("Taking darks")

    camera.take_multiple_frames(dark_folder, "dark", num_frames=5)

    for filter_name in filters:
        if progress_signal:
            progress_signal(f"Moving to filter {filter_name}")

        move_to_filter.move(filter_name)

        if progress_signal:
            progress_signal(f"Taking image with filter {filter_name}")

        camera.take_single_frame(science_folder, filter_name)

    # Close camera
    camera.close()

    # Return to no filter
    if progress_signal:
        progress_signal("Closing Shutter")
    move_to_filter.move("0")

    # Create master dark
    if progress_signal:
        progress_signal("Creating master dark")
    mdark = core.data_processing.create_master_dark(dark_folder)

    # Reduce all science images
    if progress_signal:
        progress_signal.emit("Starting image reduction")

    for image in os.listdir(science_folder):
        if image.endswith(".fits"):
            if progress_signal:
                progress_signal.emit(f"Reducing image {image}")
            image_path = os.path.join(science_folder, image)
            with fits.open(image_path) as hdul:
                science_data = hdul[0].data.astype(np.float32)
                reduced_path = os.path.join(reduced_folder, image)
                core.data_processing.reduce_image_with_dark(science_data, mdark, output_file=reduced_path, save=True)

    if progress_signal:
        progress_signal.emit("Image reduction done")


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
