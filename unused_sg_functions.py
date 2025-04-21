import time

#from cProfile import label

import numpy as np
import os
import json

import matplotlib.pyplot as plt
from skimage import io, filters, morphology
from skimage.draw import circle_perimeter
from scipy import ndimage as ndi

from analysis.sg_analysis import detect_circle, get_sg_params, \
    check_mask_flux_single, capture_images_and_reduce
from analysis.visualization import plot_com_comk_on_image_cut


def binary_filtering(image:np.ndarray, plot:bool=False): # Unused
    """
    Apply binary filtering to the given image.
    Args:
        image: Input image as a NumPy array.
        plot: If True, plot the filtering steps.

    """
    smooth = filters.gaussian(image, sigma=1.6)

    thresh_value = filters.threshold_otsu(smooth)
    thresh = smooth > thresh_value

    fill = ndi.binary_fill_holes(thresh)

    # Skip clear border step

    dilate = morphology.binary_dilation(fill)
    erode = morphology.binary_erosion(fill)

    mask = np.logical_and(dilate, ~erode)

    if plot:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6), sharey=True)

        ax[0, 0].imshow(image, cmap="gray")
        ax[0, 0].set_title('a) Raw')

        ax[0, 1].imshow(smooth, cmap="gray")
        ax[0, 1].set_title('b) Blur')

        ax[0, 2].imshow(thresh, cmap="gray")
        ax[0, 2].set_title('c) Threshold')

        ax[0, 3].imshow(fill, cmap="gray")
        ax[0, 3].set_title('c-1) Fill in')

        ax[1, 1].imshow(dilate, cmap="gray")
        ax[1, 1].set_title('d) Dilate')

        ax[1, 2].imshow(erode, cmap="gray")
        ax[1, 2].set_title('e) Erode')

        ax[1, 3].imshow(mask, cmap="gray")
        ax[1, 3].set_title('f) Nucleus Rim')

        for a in ax.ravel():
            a.set_axis_off()

        fig.tight_layout()
        plt.show()


def plot_original_with_mask_unfilled(image:np.ndarray, center_y:int, center_x:int, radius:int): # Unused
    """
    Plots the original image with an unfilled circle mask overlay.

    Parameters:
        image (np.ndarray): The original image as a NumPy array.
        center_y (int): The y-coordinate of the circle center.
        center_x (int): The x-coordinate of the circle center.
        radius (int): The radius of the circle.
    """
    overlay_image = image.copy()
    rr, cc = circle_perimeter(center_y, center_x, radius)
    overlay_image[rr, cc] = 255

    plt.imshow(overlay_image, cmap='gray')
    plt.imshow(overlay_image, cmap='gray')
    plt.title('Original Image with Unfilled Mask Overlay')
    plt.axis('off')
    plt.show()


def plot_circle_movement(image_folder:str, fiber_px_radius:int): # Todo: This function doesn't make sense for now,
    # due to precision issues with the circle detection also needs to be updated generally
    # Todo: Perhaps obsolete due to the new pipeline
    """
    Plots the movement of the center of circle of the fibers in the given image folder.
    Args:
        image_folder: Path to the folder containing the images.
        fiber_px_radius: Radius of the fiber in pixels.

    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files = sorted(image_files)

    cpos = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = io.imread(image_path)
        center_y, center_x, radius = detect_circle(image, fiber_px_radius)
        center = center_y, center_x
        cpos.append(center)

    cpos = np.array(cpos)
    print(cpos)
    print(cpos[:, 0])

    # Calculate average pos and shift around that
    cpos = cpos - np.mean(cpos, axis=0)

    # Convert to micrometers
    cpos = cpos * 0.45 # Todo: Value correct?

    time = [0,1,2] # Todo: Get the time from the image header (or somewhere else)

    plt.scatter(time, cpos[:, 1], label = 'x-shift')
    plt.scatter(time, cpos[:, 0], label = 'y-shift')
    #plt.gca().invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Shift [μm]')
    plt.title('Center of Mass Movement')
    plt.legend()
    plt.show()


def calc_sg(main_folder:str, progress_signal=None, plot_result:bool=False): # Todo: Function is probably obsolete
    """
    Calculate the scrambling gain from the given parameters
    Args:
        main_folder: Main working directory.
        progress_signal: Progress signal to update the progress.
        plot_result: If True, plot the results.

    """

    # Load the parameters
    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    # Get the parameters
    entrance_distances_y = parameters["entrance_distances_y"]
    entrance_distances = parameters["entrance_distances"]
    exit_distances_x = parameters["exit_distances_x"]
    exit_distances_y = parameters["exit_distances_y"]
    exit_distances = parameters["exit_distances"]
    reference_index = parameters["reference_index"]
    is_bad = parameters["bad_measurements"]
    fiber_px_radius_entrance = parameters["fiber_px_radius_entrance"]
    fiber_px_radius_exit = parameters["fiber_px_radius_exit"]

    # Convert lists to NumPy arrays
    entrance_distances = np.array(entrance_distances)
    exit_distances = np.array(exit_distances)
    exit_distances_x = np.array(exit_distances_x)
    exit_distances_y = np.array(exit_distances_y)

    # Rescale values with reference index as zero
    entrance_distances = entrance_distances - entrance_distances[reference_index]
    exit_distances = exit_distances - exit_distances[reference_index]
    exit_distances_x = exit_distances_x - exit_distances_x[reference_index]
    exit_distances_y = exit_distances_y - exit_distances_y[reference_index]

    # Change sign of entrance distance depending on the direction of the movement
    for i in range(len(entrance_distances)):
        if entrance_distances_y[i] < 0:
            entrance_distances[i] = -entrance_distances[i]

    # Calculate the scrambling gain for each pair
    scrambling_gain = []
    for i in range(len(entrance_distances)):
        if i == reference_index:
            continue
        if isinstance(fiber_px_radius_entrance, (tuple, list)):
            fiber_px_radius_entrance = fiber_px_radius_entrance[0]
            fiber_px_radius_exit = fiber_px_radius_exit[0]

        scrambling_gain.append(
            (entrance_distances[i] / 2 * fiber_px_radius_entrance) / (exit_distances[i] / 2 * fiber_px_radius_exit))

    sg_min = np.max(entrance_distances / fiber_px_radius_entrance) / np.max(exit_distances / fiber_px_radius_exit)
    print(f"SG_min value: {sg_min}")

    scrambling_gain = np.array(scrambling_gain)
    scrambling_gain_rounded = np.round(scrambling_gain, 2)
    print(f"Scrambling gain: ", scrambling_gain_rounded)

    # Delete the reference element from the arrays
    scrambling_gain = np.delete(scrambling_gain, reference_index)

    if progress_signal:
        progress_signal.emit("Scrambling gain calculations completed.")

    # Plot the results
    if plot_result:
        entrance_distances = np.delete(entrance_distances, np.where(is_bad))
        exit_distances_y = np.delete(exit_distances_y, np.where(is_bad))
        exit_distances_x = np.delete(exit_distances_x, np.where(is_bad))

        # reference_index = np.argmin(entrance_distances)

        # Plot COM x and y movement of the fiber output with different spot positions as colorbar
        scatter = plt.scatter(exit_distances_x, exit_distances_y, c=entrance_distances, cmap='plasma')
        plt.colorbar(scatter, label='Relative Spot Displacement [px]')
        plt.xlabel('Exit COM x-distance [px]')
        plt.ylabel('Exit COM y-distance [px]')
        plt.title('COM Movement of Fiber Output')
        plt.grid(True)
        result_name = os.path.join(main_folder, "scrambling_gain_result.png")
        plt.savefig(result_name)
        plt.show()

    # Write the scrambling gain to a json file
    sg_params = {
        "scrambling_gain": scrambling_gain.tolist(),
        "sg_min": sg_min
    }

    with open(os.path.join(main_folder, "scrambling_gain.json"), 'w') as f:
        json.dump(sg_params, f, indent=4)

    if progress_signal:
        progress_signal.emit("Scrambling gain results saved and plotted.")

def main(fiber_diameter:int, fiber_shape:str, number_of_positions:int=11): # Todo: Function is obsolete
    """
    Main function to run the scrambling gain calculation pipeline.

    Args:
        fiber_diameter: Diameter of the fiber in micrometers.
        fiber_shape: Shape of the fiber. Must be either "circle" or "octagon".
        number_of_positions: Number of positions to take images at.
    """

    main_image_folder = "D:/Vincent/thorlabs_cams_images_oct_89_other_way+camclean"

    capture_images_and_reduce(fiber_diameter, main_image_folder, number_of_positions)

    get_sg_params(main_image_folder, fiber_diameter, fiber_shape)

    calc_sg(main_image_folder, plot_result=True)


def check_mask_flux_all(main_folder:str):
    """
    Check the flux of the mask in all images.
    Args:
        main_folder: Main folder of the fiber.

    """
    # Define the image folders
    exit_mask_folder = os.path.join(main_folder, "exit/mask")
    exit_image_folder = os.path.join(main_folder, "exit/reduced")

    # Get the mask and image files
    exit_mask_files = [f for f in os.listdir(exit_mask_folder) if f.endswith('mask.png')]
    exit_mask_files = sorted(exit_mask_files)
    exit_image_files = [f for f in os.listdir(exit_image_folder) if f.endswith('reduced.png')]
    exit_image_files = sorted(exit_image_files)

    # Check the flux of the mask in all images
    for i in range(len(exit_mask_files)):
        mask = io.imread(os.path.join(exit_mask_folder, exit_mask_files[i]))
        image = io.imread(os.path.join(exit_image_folder, exit_image_files[i]))

        check_mask_flux_single(image, mask)


def plot_sg_cool_like(main_folder:str, fiber_diameter:[int, tuple[int,int]], progress_signal=None): #Todo: Function is probably obsolete
    """
    Plot the scrambling gain in a cool way.
    Args:
        main_folder: Path to the main folder containing the images.
        fiber_diameter: Diameter of the fiber in micrometers.
        progress_signal: Signal to send progress updates.
    """
    plot_folder = os.path.join(main_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)

    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    entrance_distances_x = parameters["entrance_distances_x"]
    entrance_distances_y = parameters["entrance_distances_y"]
    exit_distances_x = parameters["exit_distances_x"]
    exit_distances_y = parameters["exit_distances_y"]
    is_bad = parameters["bad_measurements"]

    # Flag bad measurements
    exit_distances = []
    for i in range(len(exit_distances_x)):
        exit_distance = np.sqrt(exit_distances_x[i] ** 2 + exit_distances_y[i] ** 2)
        exit_distances.append(exit_distance)

    print("Bad measurements:", is_bad)

    # Skip bad measurements
    entrance_distances_x = np.delete(entrance_distances_x, np.where(is_bad))
    entrance_distances_y = np.delete(entrance_distances_y, np.where(is_bad))
    exit_distances_x = np.delete(exit_distances_x, np.where(is_bad))
    exit_distances_y = np.delete(exit_distances_y, np.where(is_bad))
    exit_distances = np.delete(exit_distances, np.where(is_bad))

    # Calculate total distances with sign differences depending on the direction of the movement
    entrance_distances = np.sqrt(np.array(entrance_distances_x) ** 2 + np.array(entrance_distances_y) ** 2)

    # Update reference index
    reference_index = np.argmin(entrance_distances)

    # Change sign of entrance distance depending on the direction of the movement
    for i in range(len(entrance_distances)):
        if entrance_distances_y[i] < 0:
            entrance_distances[i] = -entrance_distances[i]

    """plt.scatter(entrance_distances_x, entrance_distances_y, label='Entrance')
    plt.scatter(exit_distances_x, exit_distances_y, label='Exit')
    plt.legend()
    plt.show()"""

    # Rebalance distances with zero as the reference index
    exit_distances_x = np.array(exit_distances_x) - exit_distances_x[reference_index]
    exit_distances_y = np.array(exit_distances_y) - exit_distances_y[reference_index]



    exit_distances = np.array(exit_distances) - exit_distances[reference_index]

    # Change unit to mu
    entrance_distances = np.array(entrance_distances) * 0.439453125
    exit_distances = np.array(exit_distances) * 0.5169363821005045
    exit_distances_x = np.array(exit_distances_x) * 0.5169363821005045
    exit_distances_y = np.array(exit_distances_y) * 0.5169363821005045

    def func(x, a, b):
        return a * x + b

    # noinspection PyPep8Naming
    def sg_func(d_in, D_in, d_out, D_out):
        return (d_in / D_in) / (d_out / D_out)

    print(entrance_distances)

    for i in range(len(entrance_distances)):
        if i == reference_index:
            continue
        if isinstance(fiber_diameter, list):
            fiber_diameter = fiber_diameter[0]

        sgx = sg_func(entrance_distances[i], fiber_diameter, exit_distances_x[i], fiber_diameter)
        sgy = sg_func(entrance_distances[i], fiber_diameter, exit_distances_y[i], fiber_diameter)
        sg = sg_func(entrance_distances[i], fiber_diameter, exit_distances[i], fiber_diameter)
        print(f"Scrambling gain {i}: {sgx}, {sgy}, {sg}")

    plt.figure(figsize=(10, 5), dpi=100)
    plt.scatter(entrance_distances, exit_distances_x, label='X movement')
    plt.scatter(entrance_distances, exit_distances_y, label='Y movement')
    plt.scatter(entrance_distances, exit_distances, label='Total movement')
    plt.plot(entrance_distances, func(entrance_distances, 5e-4, 0), 'g--', label='SG 2000', alpha=1, linewidth=0.5)
    plt.plot(entrance_distances, func(entrance_distances, -5e-4, 0), 'g--', alpha=1, linewidth=0.5)
    plt.fill_between(entrance_distances, func(entrance_distances, 5e-4, 0), func(entrance_distances, -5e-4, 0), color='g', alpha=0.3, hatch="//", zorder=2)
    plt.plot(entrance_distances, func(entrance_distances, 2e-3, 0), 'r--', label='SG 500', alpha=1, linewidth=0.5)
    plt.plot(entrance_distances, func(entrance_distances, -2e-3, 0), 'r--', alpha=1, linewidth=0.5)
    plt.fill_between(entrance_distances, func(entrance_distances, 2e-3, 0), func(entrance_distances, -2e-3, 0), color='r', alpha=0.2, hatch="/", zorder=1)
    plt.legend()
    plt.ylim(-0.08, 0.08)
    plt.ylabel('Exit COM distance [mu]')
    plt.xlabel('Entrance Spot displacement [mu]')
    plt.title('Scrambling Gain')
    plt.savefig(os.path.join(plot_folder, "scrambling_gain_plot.png"))
    plt.close()

    if progress_signal:
        progress_signal.emit("Scrambling gain plot done!")


def calc_px_to_mu(image_folder, fiber_diameter, plot=False):
    """
    Calculates the px to mu ratio from images of fibers with known size.
    Args:

    Returns:

    """
    plot_folder = os.path.join(image_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)

    # Get image paths
    images = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.endswith('.png')]

    px_to_mu_list = []

    # Check if its entrance or exit by searching for the word entrance or exit in the folder path
    if "entrance" in image_folder:
        fiber_px_radius = fiber_diameter / 0.5169363821005045 / 2
    elif "exit" in image_folder:
        fiber_px_radius = fiber_diameter / 0.439453125 / 2
    else:
        raise ValueError("Invalid folder name. Must contain either 'entrance' or 'exit'.")

    for image_path in images:
        image = io.imread(image_path)

        # Detect circle on image
        cy, cx, rad = detect_circle(image, fiber_px_radius)

        print(f"Radius: {rad}")

        if plot:
            plt.imshow(image, cmap='gray')
            plt.scatter(cx, cy, color='r', s=0.5)
            circle = plt.Circle((cx, cy), rad, color='r', fill=False)
            plt.gca().add_artist(circle)
            plt.savefig(os.path.join(plot_folder, os.path.basename(image_path).replace(".png", "_circle.png")))
            plt.close()

        # Calculate the px to mu ratio
        px_to_mu = fiber_diameter / (2 * rad)

        print(f"px_to_mu: {px_to_mu}")

        px_to_mu_list.append(px_to_mu)

    return px_to_mu_list

def capture_px_mu_calib_images(main_folder, number_of_images):
    import thorlabs_cam_control as tcc
    os.makedirs(main_folder, exist_ok=False)
    exit_folder = os.path.join(main_folder, "exit")
    os.makedirs(exit_folder, exist_ok=True)
    entrance_folder = os.path.join(main_folder, "entrance")
    os.makedirs(entrance_folder, exist_ok=True)

    # Capture images
    print("Starting entrance image capture in 5s")
    time.sleep(5)
    for i in range(number_of_images):
        tcc.take_image("entrance_cam", entrance_folder + f"/entrance_cam{i:03d}.png", exposure_time="1ms")

    print("Starting exit image capture in 30s")
    time.sleep(30)
    for i in range(number_of_images):
        tcc.take_image("exit_cam", exit_folder + f"/entrance_cam{i:03d}.png", exposure_time="1ms")

def get_px_to_mu(main_folder, fiber_diameter):

    entrance_folder = os.path.join(main_folder, "entrance")
    exit_folder = os.path.join(main_folder, "exit")

    px_to_mu_entrance = calc_px_to_mu(entrance_folder, fiber_diameter, plot=True)
    px_to_mu_exit = calc_px_to_mu(exit_folder, fiber_diameter, plot=True)

    # Convert to numpy arrays
    px_to_mu_entrance = np.array(px_to_mu_entrance)
    px_to_mu_exit = np.array(px_to_mu_exit)

    # Calculate average px to mu ratio
    px_to_mu_entrance_avg = np.mean(px_to_mu_entrance)
    px_to_mu_exit_avg = np.mean(px_to_mu_exit)

    print(f"Entrance px to mu ratios: {px_to_mu_entrance}")
    print(f"Exit px to mu ratios: {px_to_mu_exit}")

    print(f"Average px to mu ratio for entrance: {px_to_mu_entrance_avg}")
    print(f"Average px to mu ratio for exit: {px_to_mu_exit_avg}")

    # Write to json file
    px_to_mu = {"px_to_mu_entrance": px_to_mu_entrance_avg, "px_to_mu_exit": px_to_mu_exit_avg}

    with open(os.path.join(main_folder, "px_to_mu.json"), 'w') as f:
        json.dump(px_to_mu, f)


if __name__ == '__main__':

    image_path = 'D:/Vincent/40x120_300A_test/SG/exit/dark/exit_cam_dark000.png'
    #image = io.imread(image_path)
    #print(com_of_spot(image, plot=True))

    #cy, cx, rad = detect_circle(image, 55)
    #print(cy,cx,rad)
    #filled_mask = create_circular_mask(image, (cy, cx), rad, plot_mask=True)

    #image_to_fits(image_path)

    #main_folder = r"/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/R_25x40_0000_0001/SG"

    #plot_horizontal_cut_nf(main_folder)

    #reduce_images(main_folder, 11)

    # entrance_folder = "entrance_images"
    # exit_folder = "exit_images"

    #fiber_diameter = [40, 120]  # Value in micrometers
    """
    # Plot com movement for entrance images
    plot_circle_movement(entrance_folder, fiber_diameter, 'entrance')
    
    # Plot com movement for exit images
    plot_circle_movement(exit_folder, fiber_diameter, 'exit')"""

    #entrance_folder = "D:/Vincent/thorlabs_cams_images/entrance/reduced"
    #exit_folder = "D:/Vincent/thorlabs_cams_images/exit/reduced"

    #get_sg_params(main_folder, [25, 40], fiber_shape="rectangular", plot_all=True, plot_mask=True, save_mask=False)

    #calc_sg(main_folder, plot_result=True)

    #plot_masks(main_folder, fiber_diameter)

    #_, _ = capture_images_and_reduce(fiber_diameter, 11)

    #main(fiber_diameter, "octagon", number_of_positions=11)     # Always check if main folder is empty or if
    # files are important before running

    #make_comparison_video(main_folder, fiber_diameter)
    #check_mask_flux_all(main_folder)
    #main_folder = "D:/Vincent/40x120_300A_test/SG"
    #make_comparison_video(main_folder, fiber_diameter)
    #get_sg_params(main_folder, fiber_diameter, fiber_shape="rectangular", plot_all=True, plot_mask=True, save_mask=False)
    #sg_new(main_folder)
    #plot_sg_cool_like(main_folder, [40, 120])
    #make_shape("rectangular", [40, 120])

    #angle, position, radius = match_shape(image, 96, "octagon", plot_all=False, plot_best=False)
    #grow_mask(image, position, radius, "octagon", angle)

    #match_shape(image, 89, "octagon", plot_all=True, plot_best=True)

    #plot_coms(main_folder)

    #image_path = "D:/Vincent/oct_89_good/SG/exit/reduced/exit_cam_image000_reduced.png"
    #image_to_fits(image_path)

    project_folder = "D:/Vincent/oct_89_good/SG"
    plot_com_comk_on_image_cut(project_folder)

    """calib_folder = "D:/Vincent/fiber_size_calibration2"
    capture_px_mu_calib_images(calib_folder, 10)
    get_px_to_mu(calib_folder,  200)"""
