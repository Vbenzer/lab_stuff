"""Module sg_analysis.py.

Auto-generated docstring for better readability.
"""
import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import feature, transform, morphology, measure, io
from skimage.draw import disk, polygon

import core.data_processing
import core.file_management
from core.data_processing import com_of_spot, png_to_numpy, detect_circle
import core.hardware.filter_wheel_color as mtf


def make_shape(shape:str, radius:[int, tuple[int, int]]):
    """
    Create a shape for the fiber with the given radius.
    Args:
        shape: Shape form. Must be either "circular", "rectangular" or "octagonal".
        radius: Fiber radius in pixels. Int for circular or octagonal, tuple [size x, size y] for rectangular.

    Returns:
        np.ndarray: The shape as a NumPy array.
    """
    if shape == "circular":
        mask = np.zeros((2 * radius, 2 * radius), dtype=bool)
        rr, cc = disk((radius, radius), radius)
        mask[rr, cc] = True
        """plt.imshow(mask, cmap='gray')
        plt.axis("off")
        plt.show()"""
        return mask

    if shape == "rectangular":
        # Get values directly from radius tuple
        width = radius[0]
        height = radius[1]

        # Create a fitting mask
        bigger_side = max(width, height)
        mask = np.zeros((4 * bigger_side, 4 * bigger_side), dtype=bool) # Total size of the mask, must be even and big enough to fit rotations of the mask
        #aspect_ratio = 3 # >1

        # Calculate the start and end points of the rectangle
        start_x = bigger_side*2 - width
        end_x = bigger_side*2 + width
        start_y = bigger_side*2 - height
        end_y = bigger_side*2 + height

        #print(start_x, end_x, start_y, end_y)

        mask[start_x:end_x, start_y:end_y] = True

        #print(mask.shape)

        #plt.imshow(mask, cmap='gray')
        #plt.axis("off")
        #plt.show()

        return mask

    if shape == "octagonal":
        # Calculate side length from the radius
        size = 2 * radius * (np.sqrt(2) - 1)

        # Total size of the mask (to fit the octagon comfortably)
        total_size = int(radius * 4)  # Total size of the mask, must be even and big enough to fit rotations of the mask
        mask = np.zeros((total_size, total_size), dtype=np.uint8)

        # Calculate center of the mask
        center = total_size / 2

        # Calculate vertices of the octagon
        vertices = np.array([
            (center - size / 2, center - radius),  # Top-left
            (center + size / 2, center - radius),  # Top-right
            (center + radius, center - size / 2),  # Right-top
            (center + radius, center + size / 2),  # Right-bottom
            (center + size / 2, center + radius),  # Bottom-right
            (center - size / 2, center + radius),  # Bottom-left
            (center - radius, center + size / 2),  # Left-bottom
            (center - radius, center - size / 2),  # Left-top
        ])

        # Create the polygon (octagon) mask
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], mask.shape)
        mask[rr, cc] = 255

        """plt.imshow(mask, cmap="gray")
        plt.title("Octagon Mask")"""

        return mask


def match_shape(image:np.ndarray, radius:int, shape:str, num_rotations:int=50, radius_threshold:int=5,
                plot_all:bool=False, best_params=None):
    """
    Match the given shape to the image using template matching.
    Args:
        image: Input image as a NumPy array.
        radius: Expected radius of the shape in pixels.
        shape: Form of the shape. Must be either "circular", "rectangular" or "octagonal".
        num_rotations: Number of rotations to try. Total angle depends on shape.
        radius_threshold: Threshold for the radius to try.
        plot_all: If True, plot all the steps.
        best_params: If not None, use the best parameters for the template matching

    Returns:
        best_angle: The angle of the best match.
        com_position: The center of mass of the best match.
        best_radius: The radius of the best match.
    """
    best_match = None
    best_score = -np.inf
    best_angle = 0
    best_location = None
    best_radius = radius

    print("Matching shape")

    # Detect edges using Canny edge detector
    edges = feature.canny(image, sigma=2, low_threshold=1, high_threshold=15)

    # Ensure outline is closed (2 iterations of dilation and erosion)
    erode = morphology.binary_erosion(edges)
    dilate = morphology.binary_dilation(edges)
    mask_1 = np.logical_and(dilate, ~erode)

    erode = morphology.binary_erosion(mask_1)
    dilate = morphology.binary_dilation(mask_1)
    mask_2 = np.logical_and(dilate, ~erode)

    # Plots to check mask
    if plot_all:
        plt.figure(figsize=(12.80, 10.24))
        plt.imshow(edges)
        plt.title("Edges")
        plt.show()

        plt.figure(figsize=(12.80, 10.24))
        plt.imshow(mask_2)
        plt.title("Mask after 2nd iteration of dilation and erosion")
        plt.show()

    # Fill in the holes in the edges
    filled_mask = ndi.binary_fill_holes(mask_2)

    if plot_all:
        plt.figure(figsize=(12.80, 10.24))
        # Plot the filled mask
        plt.imshow(filled_mask)
        plt.title("Filled Mask")
        plt.show()

    # Set the stop value for the rotation depending on the shape
    if shape == "circular":
        stop_value = 0
        num_rotations = 1
    elif shape == "rectangular":
        stop_value = 180
        num_rotations = 180
    elif shape == "octagonal":
        stop_value = 45
        num_rotations = 45
    else:
        raise ValueError("Invalid shape. Must be either 'circle', 'rectangle' or 'octagon")

    # Convert bool to uint8
    uint8_mask = filled_mask.astype(np.uint8)
    #uint8_mask = image

    """if image.dtype != np.uint8:
        edges = (edges * 255).astype(np.uint8)"""

    if best_params is not None:
        radius = best_params[0]
        angle = best_params[1]

        if shape == "rectangular":
            r = [radius[0], radius[1]]
        else:
            r = radius

        # Create the shape
        full_shape = make_shape(shape, r)

        # Convert the full_shape to just the outline of the shape
        erode = morphology.binary_erosion(full_shape)
        dilate = morphology.binary_dilation(full_shape)

        # Create the template
        template = np.logical_and(dilate, ~erode)

        # Rotate the template
        rotated_template = transform.rotate(template, angle, resize=False)
        rotated_template = (rotated_template * 255).astype(np.uint8)

        # Perform template matching
        result = cv2.matchTemplate(uint8_mask, rotated_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        best_match = rotated_template
        best_angle = angle
        best_location = max_loc
        best_radius = r

    else:
        for i in range( -radius_threshold, radius_threshold): # Iterate over different radii
            if shape == "rectangular":
                r = [radius[0] + i, radius[1] + i]
            else:
                r = radius + i

            # Create the shape
            full_shape = make_shape(shape,r)

            # Convert the full_shape to just the outline of the shape
            erode = morphology.binary_erosion(full_shape)
            dilate = morphology.binary_dilation(full_shape)

            # Create the template
            template = np.logical_and(dilate, ~erode)

            #print("Template shape:", template.shape)

            for angle in np.linspace(0, stop_value, num_rotations, endpoint=False): # Iterate over different rotations
                print("Angle:", angle)
                # Rotate the template
                rotated_template = transform.rotate(template, angle, resize=False)
                rotated_template = (rotated_template * 255).astype(np.uint8)

                #print("Rotated template shape:", rotated_template.shape)

                # Perform template matching
                result = cv2.matchTemplate(uint8_mask, rotated_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > best_score:
                    # Update the best match
                    best_score = max_val
                    best_match = rotated_template
                    best_angle = angle
                    best_location = max_loc
                    best_radius = r

    if best_match is not None:
        # Calculate the center of mass of the best match
        com_position = best_location[0] + best_match.shape[1] / 2, best_location[1] + best_match.shape[0] / 2

        if plot_all:

            radius = best_radius

            plot_best = make_shape(shape, radius)

            # Rotate the best match
            plot_best = transform.rotate(plot_best, best_angle, resize=False)


            # Pad the best match to fit the image while getting it on position
            pad_x_0 = com_position[1] - plot_best.shape[1] // 2
            pad_x_1 = image.shape[0] - (com_position[1] + plot_best.shape[1] // 2)
            pad_y_0 = com_position[0] - plot_best.shape[0] // 2
            pad_y_1 = image.shape[1] - (com_position[0] + plot_best.shape[0] // 2)

            # Plot best match
            plt.figure(figsize=(12.80, 10.24))
            plt.imshow(best_match, cmap='gray')
            plt.title("Best Match")
            plt.show()

            # Pad the best match to fit the image while getting it on position
            plot_best = np.pad(plot_best, ((int(pad_x_0), int(pad_x_1)), (int(pad_y_0), int(pad_y_1)),), mode='constant')
            plot_best_outline = measure.find_contours(plot_best, 0.5)[0]

            plt.figure(figsize=(12.80, 10.24))
            plt.imshow(image, cmap='gray')
            plt.plot(plot_best_outline[:, 1], plot_best_outline[:, 0], 'r', linewidth=0.5)
            plt.title("Best Match")
            plt.show()

        return best_angle, com_position, best_radius

    else:
        print("No match found")


def build_mask(image:np.ndarray, radius:[int, tuple[int,int]], shape:str, mask_margin:int, position:tuple[int,int],
               angle:int, plot_mask:bool=False, save_mask:str="-1"):
    """
    Build mask with given parameters and shape
    Args:
        image: Input image as a NumPy array.
        radius: Radius of size of the mask.
        shape: Form of the shape. Must be either "circular", "rectangular" or "octagonal".
        mask_margin: Margin to add to the radius.
        position: Position of the shape in the image.
        angle: Angle of the shape in the image.
        plot_mask: If True, plot the mask.
        save_mask: If not -1, save the mask to the given path.

    Returns:
        np.ndarray: The mask of the detected shape.
        tuple: The center of mass of the detected
    """

    # Handle radius = size case for rectangular
    if shape == "rectangular":
        radius = [radius[0] + mask_margin, radius[1] + mask_margin]
    else:
        radius += mask_margin

    # Create the shape
    best_match = make_shape(shape, radius)

    # Rotate the best match
    best_match = transform.rotate(best_match, angle, resize=False)

    # Pad the best match to fit the image while getting it on position
    pad_x_0 = position[1] - best_match.shape[1] // 2
    pad_x_1 = image.shape[0] - (position[1] + best_match.shape[1] // 2)
    pad_y_0 = position[0] - best_match.shape[0] // 2
    pad_y_1 = image.shape[1] - (position[0] + best_match.shape[0] // 2)

    # Pad the best match to fit the image while getting it on position
    best_match = np.pad(best_match, ((int(pad_x_0), int(pad_x_1)), (int(pad_y_0), int(pad_y_1)),), mode='constant')

    # Convert the best match to uint8
    best_match = best_match.astype(np.uint8) * 255

    # Fill in the holes in the best match
    best_match_bin = best_match.astype(bool)
    filled_best_match = ndi.binary_fill_holes(best_match_bin)

    # Calculate the center of mass of the best match
    com_of_mask = com_of_spot(best_match)

    # Erode and dilate to ensure mask is big enough
    erode = morphology.binary_erosion(best_match)
    dilate = morphology.binary_dilation(best_match)

    # Create the mask using the eroded and dilated masks
    mask = np.logical_and(dilate, ~erode)

    # Fill in the holes in the mask
    filled_mask = ndi.binary_fill_holes(mask)

    # Cut images for display
    image_dis = cut_image_around_comk(image, com_of_mask, radius, 50)
    best_match_dis = cut_image_around_comk(best_match, com_of_mask, radius, 50)
    mask_dis = cut_image_around_comk(mask, com_of_mask, radius, 50)
    filled_mask_dis = cut_image_around_comk(filled_mask, com_of_mask, radius, 50)

    if plot_mask or save_mask != "-1":
        fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharey=True)

        ax[0, 0].imshow(image_dis, cmap="gray")
        #ax[0, 0].plot(com_of_mask[1], com_of_mask[0], 'ro', markersize=0.5)
        ax[0, 0].set_title('Original Image')

        best_match_outline = measure.find_contours(best_match_dis, 0.5)[0]
        ax[0, 1].imshow(image_dis, cmap="gray")
        ax[0, 1].plot(best_match_outline[:, 1], best_match_outline[:, 0], 'r', linewidth=0.5)
        ax[0, 1].set_title('Best Match')

        mask_outline = measure.find_contours(mask_dis, 0.5)[0]
        ax[1, 0].imshow(image_dis, cmap="gray")
        ax[1, 0].plot(mask_outline[:, 1], mask_outline[:, 0], 'r', linewidth=0.5)
        ax[1, 0].set_title('Mask')

        ax[1, 1].imshow(filled_mask_dis, cmap="gray")
        ax[1, 1].set_title('Filled Mask')

        fig.tight_layout()

        if save_mask != -1: # If not -1, save the mask as save_mask parameter
            plt.savefig(save_mask)

        if plot_mask:
            plt.show()

        plt.close()

    return filled_best_match, com_of_mask


def create_circular_mask(image:np.ndarray, center:tuple[int,int], radius:int, margin=0, plot_mask:bool=False):
    """
    Creates a circular mask for the given image.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        center (tuple): The (y, x) coordinates of the circle center.
        radius (int): The radius of the circular mask.
        margin (int): The margin to add to the radius. Unused for now.
        plot_mask (bool): If True, plot the mask.

    Returns:
        np.ndarray: The circular mask as a NumPy array.
    """
    from skimage.draw import disk

    mask = np.zeros(image.shape[:2], dtype=bool)
    rr, cc = disk(center, radius + margin, shape=image.shape)
    mask[rr, cc] = True

    # Erode and dilate to ensure mask is big enough
    erode = morphology.binary_erosion(mask)
    dilate = morphology.binary_dilation(erode)
    mask = np.logical_and(dilate, ~erode)

    # Fill in the holes in the mask
    filled_mask = ndi.binary_fill_holes(mask)

    if plot_mask:
        # Plot the outline of the mask
        mask_outline = measure.find_contours(filled_mask, 0.5)[0]
        plt.plot(mask_outline[:, 1], mask_outline[:, 0], 'r', linewidth=0.5)
        plt.imshow(image, cmap='gray')
        plt.title('Circular Mask')
        plt.show()

    return filled_mask


def get_sg_params(main_folder:str, fiber_diameter:int, fiber_shape:str, progress_signal=None,
                              save_mask:bool=True, plot_mask:bool=False, plot_all:bool=False):
    """
    Calculate the scrambling gain of the fiber using the given images.
    Args:
        main_folder: Path to the main folder containing the entrance and exit images.
        fiber_diameter: Diameter of the fiber in micrometers.
        fiber_shape: Shape of the fiber. Must be either "circular", "rectangular" or "octagonal".
        progress_signal: Signal to update the progress.
        save_mask: If True, save the mask as png.
        plot_mask: If True, plot the mask.
        plot_all: If True, plot all the steps.

    Returns:
    scrambling_gain: List of scrambling gains for each pair of entrance and exit images.
    """

    # Detect if positions.json file exists
    positions_file = os.path.join(main_folder, "positions.json")
    if not os.path.exists(positions_file):
        raise FileNotFoundError(f"Positions file {positions_file} not found. Please run new measurement first.")

    # Define the image folders
    entrance_image_folder_reduced = os.path.join(main_folder, "entrance/reduced")
    exit_image_folder_reduced = os.path.join(main_folder, "exit/reduced")
    entrance_mask_folder = os.path.join(main_folder, "entrance/mask")
    exit_mask_folder = os.path.join(main_folder, "exit/mask")

    # Create mask folders
    os.makedirs(entrance_mask_folder, exist_ok=True)
    os.makedirs(exit_mask_folder, exist_ok=True)

    # Calculate the radius of the fiber in pixels. Also handle rectangular case
    if fiber_shape == "rectangular":
        fiber_px_radius_entrance = (int(fiber_diameter[0] / 0.5169363821005045 / 2), int(fiber_diameter[1] / 0.5169363821005045 / 2))
        fiber_px_radius_exit = (int(fiber_diameter[0] / 0.439453125 / 2), int(fiber_diameter[1] / 0.439453125 / 2))
    else:
        fiber_px_radius_entrance = int(fiber_diameter / 0.5169363821005045 / 2)
        fiber_px_radius_exit = int(fiber_diameter / 0.439453125 / 2)

    # Get values of the entrance image
    entrance_image_files = sorted([f for f in os.listdir(entrance_image_folder_reduced) if f.endswith('reduced.png')])
    entrance_coms = [] # Center of mass of spot in the entrance image
    entrance_comk = [] # Center of mask and with that the center of the fiber
    entrance_radii = []

    # Send progress signal
    if progress_signal:
        progress_signal.emit("Processing entrance images")

    # Set best_params to None
    best_params = None

    # Process entrance images
    for image_file in entrance_image_files:
        image_path = os.path.join(entrance_image_folder_reduced, image_file)

        if progress_signal:
            progress_signal.emit(f"Processing entrance image {image_file}")

        print(image_path)

        # Read the image
        image = io.imread(image_path)

        # Get the center of mass of the spot. Threshold important to ensure correct spot detection
        com = com_of_spot(image, threshold=10, plot=False)

        # Append the center of mass to the list
        entrance_coms.append(com)

        # Create the mask path
        mask_path = os.path.join(entrance_mask_folder, image_file.replace(".png", "_mask.png"))

        # Find center of fiber depending on the shape
        if fiber_shape == "circular":
            # Detect the circle in the image
            center_y, center_x, radius = detect_circle(image, fiber_px_radius_entrance)

            # Create the circular mask
            mask = create_circular_mask(image, (center_y, center_x), radius, plot_mask=plot_mask)

            # Save the mask
            io.imsave(mask_path, mask.astype(np.uint8) * 255)

            # Plot the detected circle with outline of the mask
            plt.imshow(image, cmap='gray')
            circle_outline = plt.Circle((center_x, center_y), radius, color='r', fill=False)
            plt.gca().add_artist(circle_outline)

            if save_mask:
                mask_name = image_path.replace(".png", "_mask.png")
                plt.savefig(mask_name)

            if plot_mask:
                plt.show()

            plt.close()

            # Write the center of the mask
            comk = [center_y, center_x]

        elif fiber_shape == "octagonal" or fiber_shape == "rectangular":
            # Save mask if save_mask is True
            if save_mask:
                save_mask = image_path.replace(".png", "_mask.png")
            else:
                save_mask = -1

            # Match the shape of the fiber
            angle, position, radius = match_shape(image, fiber_px_radius_entrance, fiber_shape, plot_all=plot_all, best_params=best_params)

            # Save best_params for next iteration to accelerate shape matching
            best_params = [radius, angle]

            # Make the mask with the given parameters
            mask, comk = build_mask(image, radius, fiber_shape, 1, position, angle, plot_mask=plot_mask,
                                    save_mask=save_mask)

            # Save the mask
            io.imsave(mask_path, mask.astype(np.uint8) * 255)

        else:
            raise ValueError("Invalid fiber shape. Must be either 'circular', 'rectangular' or 'octagonal'.")

        # Append the center of the mask and the radius to the lists
        entrance_comk.append(comk)
        entrance_radii.append(radius)

    # Convert lists to NumPy arrays
    entrance_coms = np.array(entrance_coms)
    entrance_comk = np.array(entrance_comk)
    entrance_radii = np.array(entrance_radii)

    # Get values of the exit image
    exit_image_files = sorted([f for f in os.listdir(exit_image_folder_reduced) if f.endswith('reduced.png')])
    exit_comk = []
    exit_coms = []
    exit_radii = []
    max_flux_list = []

    print("Entrance center of mask:", entrance_comk)
    print("Entrance center of mass:", entrance_coms)
    print("Entrance radii:", entrance_radii)

    # Write progress signal
    if progress_signal:
        progress_signal.emit("Input images done. Processing exit images")

    # Set best_params to None
    best_params = None

    com = None
    mask = None
    comk = None
    radius = None

    # Process exit images
    for image_file in exit_image_files:
        image_path = os.path.join(exit_image_folder_reduced, image_file)

        # Write progress signal
        if progress_signal:
            progress_signal.emit(f"Processing exit image {image_file}")

        print(image_path)

        image = io.imread(image_path)

        mask_path = os.path.join(exit_mask_folder, image_file.replace(".png", "_mask.png"))

        if mask is None:
            # Find center of fiber
            if fiber_shape == "circular":
                center_y, center_x, radius = detect_circle(image, fiber_px_radius_exit, sigma=3)
                comk = [center_y, center_x]

                mask_margin_circ = grow_mask(image, comk, radius, "circular")

                mask = create_circular_mask(image, (center_y, center_x), radius + mask_margin_circ, plot_mask=plot_mask)

                io.imsave(mask_path, mask.astype(np.uint8) * 255)

            elif fiber_shape == "octagonal" or fiber_shape == "rectangular":
                if save_mask:
                    save_mask = image_path.replace(".png", "_mask.png")
                else:
                    save_mask = -1

                angle, position, radius = match_shape(image, fiber_px_radius_exit, fiber_shape, plot_all=plot_all)

                # Calculate the margin to add to the radius to ensure minimal flux outside of mask
                mask_margin = grow_mask(image, position, radius, fiber_shape, angle=angle)

                mask, comk = build_mask(image, radius, fiber_shape, mask_margin, position, angle, plot_mask=plot_mask,
                                        save_mask=save_mask)

                radius = fiber_px_radius_exit

            else:
                raise ValueError("Invalid fiber shape. Must be either 'circle' or 'octagon'.")

        io.imsave(mask_path, mask.astype(np.uint8) * 255)

        exit_comk.append(comk)
        exit_radii.append(radius)

        # Check flux outside mask
        max_flux = int(check_mask_flux_single(image, mask, plot=False))
        max_flux_list.append(max_flux)

        # Convert mask to bool
        mask.astype(bool)

        # Use exit fiber mask to set background to zero
        image[~mask] = 0

        # Find the center of mass of the image with reduced background
        com = com_of_spot(image)
        exit_coms.append(com)

    exit_comk = np.array(exit_comk)
    exit_coms = np.array(exit_coms)
    exit_radii = np.array(exit_radii)

    print("Exit center of mask:", exit_comk)
    print("Exit center of mass:", exit_coms)
    print("Exit radii:", exit_radii)

    if progress_signal:
        progress_signal.emit("Exit images done. Writing parameters to json file")

    # Todo: Most parameters outside of com and comk are probably obsolete
    # Calculate distance between entrance COMK and COM
    #entrance_distances = np.linalg.norm(entrance_coms - exit_comk, axis=1)
    entrance_distances_x = entrance_coms[:, 0] - entrance_comk[:, 0]
    entrance_distances_y = entrance_coms[:, 1] - entrance_comk[:, 1]
    print("Entrance distances x,y:", entrance_distances_x, entrance_distances_y)

    # Calculate distance between exit COMK and COM
    #exit_distances = np.linalg.norm(exit_coms - exit_comk, axis=1)
    exit_distances_x = exit_coms[:, 0] - exit_comk[:, 0]
    exit_distances_y = exit_coms[:, 1] - exit_comk[:, 1]

    print("Exit distances x,y:", exit_distances_x, exit_distances_y)

    # Calculate total distances
    entrance_distances = np.sqrt(entrance_distances_x**2 + entrance_distances_y**2)
    exit_distances = np.sqrt(exit_distances_x**2 + exit_distances_y**2)

    # Check for bad measurements
    distance_avg = np.mean(exit_distances)
    is_bad = []
    for i in range(len(exit_distances)):
        if exit_distances[i] > distance_avg * 2.5:
            flag = True
            is_bad.append(flag)
        else:
            flag = False
            is_bad.append(flag)
    print("Bad measurements:", is_bad)

    # Choose the exit/entrance pair with the smallest entrance distance from COMK as reference
    reference_index = np.argmin(entrance_distances)
    print("Reference index:", reference_index)

    # Save all parameters in json file
    parameters = {
        "entrance_coms": entrance_coms.tolist(),
        "entrance_comk": entrance_comk.tolist(),
        "entrance_radii": entrance_radii.tolist(),
        "entrance_distances_x": entrance_distances_x.tolist(),
        "entrance_distances_y": entrance_distances_y.tolist(),
        "entrance_distances": entrance_distances.tolist(),
        "exit_coms": exit_coms.tolist(),
        "exit_comk": exit_comk.tolist(),
        "exit_radii": exit_radii.tolist(),
        "exit_distances_x": exit_distances_x.tolist(),
        "exit_distances_y": exit_distances_y.tolist(),
        "exit_distances": exit_distances.tolist(),
        "reference_index": int(reference_index),
        "max_flux": max_flux_list,
        "bad_measurements": is_bad,
        "fiber_px_radius_entrance": fiber_px_radius_entrance,
        "fiber_px_radius_exit": fiber_px_radius_exit
    }

    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'w') as f:
        json.dump(parameters, f, indent=4)


def get_precise_entrance_positions(main_folder: str, progress_signal=None):
    from scipy.interpolate import interp1d
    parameter_file = os.path.join(main_folder, "scrambling_gain_parameters.json")
    positions_file = os.path.join(main_folder, "positions.json")

    if not os.path.exists(parameter_file):
        raise FileNotFoundError(f"Parameter file {parameter_file} not found. Please run get_sg_params first.")
    if not os.path.exists(positions_file):
        raise FileNotFoundError(f"Positions file {positions_file} not found. Please run get_positions first.")

    # Load comk values
    with open(parameter_file, 'r') as f:
        parameters = json.load(f)
    entrance_comk = np.array(parameters["entrance_comk"])
    comk_y = entrance_comk[:, 1]

    # Load and clean motor positions: from "1=4.98400" to float
    with open(positions_file, 'r') as f:
        raw_positions = json.load(f)
    positions = np.array([float(p.split('=')[1]) for p in raw_positions])

    if len(comk_y) != len(positions):
        raise ValueError(f"Length mismatch: {len(comk_y)} comks vs {len(positions)} positions")

    # Create interpolation: motor position -> comk_y
    # This gives a smooth fit of comk_y across motor positions
    interp = interp1d(positions, comk_y, kind='cubic', fill_value="extrapolate")

    # Evaluate the interpolation at each motor position (now with higher precision)
    refined_comk_y = interp(positions)

    if progress_signal:
        progress_signal.emit("Calculated precise entrance positions.")

    return refined_comk_y



def check_mask_flux_single(image:np.ndarray, mask:np.ndarray, plot:bool=False, print_text:bool=True):
    """
    Check the flux of the mask in the image.
    Args:
        image: The input image as a NumPy array.
        mask: The mask as a NumPy array.
        plot: If True, plot the image with the mask removed.
        print_text: If True, print the max flux value outside the mask.

    Returns:
        int: The max flux value outside the mask.
    """
    # Convert mask to bool
    mask = mask.astype(bool)

    # Set area of mask to zero
    image_wo_mask = image.copy()
    image_wo_mask[mask] = 0

    if plot:
        plt.imshow(image_wo_mask, cmap='gray')
        plt.title('Image with Mask Removed')
        plt.show()

    if print_text:
        print("Max flux value outside of mask:", np.max(image_wo_mask))

    return np.max(image_wo_mask)


def grow_mask(image:np.ndarray, position:(tuple[int,int],list), radius:[int, tuple[int,int]], shape:str, angle:int = 0):
    """
    Grow the mask until the flux outside the mask is less than 10 and doesn't change for 2 iterations.
    Args:
        image: Image as a NumPy array.
        position: Position of the mask as tuple (y, x).
        radius: Radius or size of the mask.
        shape: Shape of the mask/fiber (circular, octagonal, rectangular).
        angle: Angle at which the mask is rotated.

    Returns: The margin by which the mask was grown.

    """
    # Set initial flux
    min_flux = np.inf

    # Define the center of the mask
    center_y, center_x = position[0], position[1]

    # Create mask with differing margins
    for margin in range(0, 1000, 2):
        if shape == "circular":
            mask = create_circular_mask(image, (center_y, center_x), radius + margin, plot_mask=False)

        elif shape == "octagonal" or shape == "rectangular":
            mask, comk = build_mask(image, radius, shape, margin, position, angle)

        else:
            raise ValueError("Invalid fiber shape. Must be either 'circular', 'octagonal' or 'rectangular'.")

        # Check flux outside mask
        max_flux = int(check_mask_flux_single(image, mask, plot=False, print_text=False))

        if shape == "rectangular":  # Handle rectangular case
            radius_1 = radius[0]
        else:
            radius_1 = radius

        if max_flux < min_flux:
            min_flux = max_flux

        # If the flux is less than 10 and doesn't change for 2 iterations, return the margin
        elif max_flux == min_flux and max_flux < 10:
            print(f"Margin: {margin}, Flux: {max_flux}")
            return margin

        # If the flux is less than 10 and the mask is too big, return the margin
        elif radius_1 + margin > 1.2 * radius_1:
            print(f"Max margin reached. Margin: {margin}, Flux: {max_flux}")
            return margin



def capture_images_and_reduce(main_image_folder:str, fiber_diameter:[int, tuple[int,int]], progress_signal=None,
                              number_of_positions:int=11, exposure_times:dict[str, str]=None):
    """
    Capture images and reduce them for the scrambling gain calculation
    Args:
        main_image_folder: Path to the main image folder.
        fiber_diameter: Diameter of the fiber in micrometers.
        progress_signal: Signal to update the progress.
        number_of_positions: Number of positions to take images at.
        exposure_times: Dictionary containing the exposure times for the cameras.

    """
    if exposure_times is None:
        # Raise an error if the exposure times are not given
        raise ValueError("Exposure times must be given.")

    from core.hardware.cameras import thorlabs_cam_control as tcc
    import time
    from core.hardware import motor_control as smc

    # Define image folders
    os.makedirs(main_image_folder, exist_ok=False)
    entrance_image_folder = os.path.join(main_image_folder, "entrance")
    exit_image_folder = os.path.join(main_image_folder, "exit")
    entrance_dark_image_folder = os.path.join(entrance_image_folder, "dark")
    exit_dark_image_folder = os.path.join(exit_image_folder, "dark")
    reduced_entrance_image_folder = os.path.join(entrance_image_folder, "reduced")
    reduced_exit_image_folder = os.path.join(exit_image_folder, "reduced")
    plots_folder = os.path.join(main_image_folder, "plots")
    entrance_light_folder = os.path.join(entrance_image_folder, "light")
    exit_light_folder = os.path.join(exit_image_folder, "light")

    # Clear folders before adding new images
    core.file_management.clear_folder(main_image_folder)

    # Create folders if they don't exist
    os.makedirs(entrance_image_folder, exist_ok=True)
    os.makedirs(exit_image_folder, exist_ok=True)
    os.makedirs(entrance_dark_image_folder, exist_ok=True)
    os.makedirs(exit_dark_image_folder, exist_ok=True)
    os.makedirs(reduced_entrance_image_folder, exist_ok=True)
    os.makedirs(reduced_exit_image_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(entrance_light_folder, exist_ok=True)
    os.makedirs(exit_light_folder, exist_ok=True)

    # Move to filter "none" for no light
    mtf.move("Closed")
    print("Taking darks")

    # Send progress signal
    if progress_signal:
        progress_signal.emit("Taking darks")

    # Take darks
    for i in range(5):
        tcc.take_image("entrance_cam", entrance_dark_image_folder + f"/entrance_cam_dark{i:03d}.png",
                       exposure_time=exposure_times["entrance"])
        tcc.take_image("exit_cam", exit_dark_image_folder + f"/exit_cam_dark{i:03d}.png",
                       exposure_time=exposure_times["exit"])

    # Move to filter "0" for light
    mtf.move("Open")

    # Make reference move
    #smc.make_reference_move()

    print("Taking images")

    if progress_signal:
        progress_signal.emit("Darks Done! Taking lights.")

    # Calculate the step size and leftmost position. Also handle rectangular case
    if isinstance(fiber_diameter, tuple):
        max_size = max(fiber_diameter)
        step_size = max_size / 1000 * 0.8 / (number_of_positions - 1)  # Step size in mm
        pos_left = 5 - max_size / 1000 * 0.8 / 2  # Leftmost position in mm
    else:
        step_size = fiber_diameter / 1000 * 0.8 / (number_of_positions-1)  # Step size in mm
        pos_left = 5 - fiber_diameter / 1000 * 0.8 / 2  # Leftmost position in mm

    # Take images
    position_list = []  # List to save positions
    for i in range(number_of_positions):
        print(i, pos_left + i * step_size)

        # Move the motor to the next position
        position = smc.move_motor_to_position(pos_left + i * step_size)

        # Save position to list
        position_list.append(position)

        # Take images
        tcc.take_image("entrance_cam", entrance_light_folder + f"/entrance_cam_image{i:03d}.png",
                       exposure_time=exposure_times["entrance"])
        tcc.take_image("exit_cam", exit_light_folder + f"/exit_cam_image{i:03d}.png",
                       exposure_time=exposure_times["exit"])

        print(f"Image {i + 1} out of {number_of_positions} at Position {pos_left + i * step_size} done! "
              f"Move spot to next position")

        time.sleep(1)  # Probably not necessary
    print("All images taken!")

    # Save the positions to json file
    positions_file = os.path.join(main_image_folder, "positions.json")
    with open(positions_file, 'w') as f:
        json.dump(position_list, f, indent=4)

    # Send progress signal
    if progress_signal:
        progress_signal.emit("Positions saved to json.")

    # Reset the motor to the initial position
    smc.move_motor_to_position(5)

    if progress_signal:
        progress_signal.emit("Images taken! Reducing images.")

    # Reduce images
    reduce_images(main_image_folder, number_of_positions)

    if progress_signal:
        progress_signal.emit("Images reduced!")


def reduce_images(main_image_folder, number_of_positions:int):
    from astropy.io import fits

    os.makedirs(main_image_folder, exist_ok=True)
    entrance_image_folder = os.path.join(main_image_folder, "entrance")
    exit_image_folder = os.path.join(main_image_folder, "exit")
    entrance_dark_image_folder = os.path.join(entrance_image_folder, "dark")
    exit_dark_image_folder = os.path.join(exit_image_folder, "dark")
    reduced_entrance_image_folder = os.path.join(entrance_image_folder, "reduced")
    reduced_exit_image_folder = os.path.join(exit_image_folder, "reduced")
    entrance_light_folder = os.path.join(entrance_image_folder, "light")
    exit_light_folder = os.path.join(exit_image_folder, "light")

    # Clear folders before adding new images
    #core.file_management.clear_folder(main_image_folder)

    # Create folders if they don't exist
    os.makedirs(entrance_image_folder, exist_ok=True)
    os.makedirs(exit_image_folder, exist_ok=True)
    os.makedirs(entrance_dark_image_folder, exist_ok=True)
    os.makedirs(exit_dark_image_folder, exist_ok=True)
    os.makedirs(reduced_entrance_image_folder, exist_ok=True)
    os.makedirs(reduced_exit_image_folder, exist_ok=True)
    os.makedirs(entrance_light_folder, exist_ok=True)
    os.makedirs(exit_light_folder, exist_ok=True)

    # Create master darks
    entrance_master_dark = (core.data_processing.create_master_dark(entrance_dark_image_folder, plot=False))
    exit_master_dark = core.data_processing.create_master_dark(exit_dark_image_folder, plot=False)

    # Save the master darks as fits
    fits.writeto(os.path.join(entrance_dark_image_folder, "entrance_master_dark.fits"), entrance_master_dark, overwrite=True)
    fits.writeto(os.path.join(exit_dark_image_folder, "exit_master_dark.fits"), exit_master_dark, overwrite=True)

    # Reduce images
    for i in range(number_of_positions):
        image = png_to_numpy(entrance_light_folder + f"/entrance_cam_image{i:03d}.png")
        core.data_processing.reduce_image_with_dark(image, entrance_master_dark,
                                                    reduced_entrance_image_folder + f"/entrance_cam_image{i:03d}_reduced.png",
                                                    save=True)
        image = png_to_numpy(exit_light_folder + f"/exit_cam_image{i:03d}.png")
        core.data_processing.reduce_image_with_dark(image, exit_master_dark,
                                                    reduced_exit_image_folder + f"/exit_cam_image{i:03d}_reduced.png",
                                                    save=True)

    print("All images reduced!")


def cut_image_around_comk(image, comk, fiber_px_radius:[int, tuple[int,int]], margin):
    """
    Cut the image around the center of the mask.
    Args:
        image: The input image as a NumPy array.
        comk: The center of the mask.
        fiber_px_radius: The radius of the fiber in pixels.
        margin: The margin to add to the radius.

    Returns:
        np.ndarray: The cut image.
    """
    if isinstance(fiber_px_radius, (list, tuple)): # Handle rectangular case (radius is tuple or list in that case)
        bigger_side = max(fiber_px_radius)
        cut_image = image[int(comk[0]) - bigger_side - margin:int(comk[0]) + bigger_side + margin,
                    int(comk[1]) - bigger_side - margin:int(comk[1]) + bigger_side + margin]
    else:
        cut_image = image[int(comk[0]) - fiber_px_radius - margin:int(comk[0])  + fiber_px_radius + margin,
                    int(comk[1]) - fiber_px_radius- margin:int(comk[1]) + fiber_px_radius + margin]

    return cut_image


def sg_new(main_folder:str, progress_signal=None):
    """
    Calculate the scrambling gain with the new method.
    Args:
        main_folder: Main folder of the fiber.
        progress_signal: Progress signal to update the progress.

    """
    positions_file = os.path.join(main_folder, "positions.json")

    # Load motor positions (in mm or similar)
    with open(positions_file, 'r') as f:
        raw_positions = json.load(f)

    positions = np.array([float(item.split('=')[1]) for item in raw_positions])

    # Read parameters from json file
    if progress_signal:
        progress_signal.emit("Loading parameters form json")
    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    # Get the parameters. Currently only using com and comks, which is probably best.
    entrance_coms = parameters["entrance_coms"]
    entrance_comk = parameters["entrance_comk"]
    exit_coms = parameters["exit_coms"]
    exit_comk = parameters["exit_comk"]

    # Convert lists to NumPy arrays
    entrance_coms = np.array(entrance_coms)
    entrance_comk = np.array(entrance_comk)
    exit_coms = np.array(exit_coms)
    exit_comk = np.array(exit_comk)

    # Calculate median of entrance coms and exit comk, because fiber should always be in the same position
    entrance_coms_median = np.median(entrance_coms, axis=0)
    exit_comk_median = np.median(exit_comk, axis=0)

    # Calculate the distance of the com to the comk
    dist_to_comk = np.zeros(len(entrance_coms))
    for i in range(len(entrance_coms)):
        dist_to_comk[i] = np.sqrt((entrance_coms_median[0] - entrance_comk[i][0]) ** 2 + (entrance_coms_median[1] - entrance_comk[i][1]) ** 2)

    # Get reference index by finding the minimum distance to the comk
    reference_index = np.argmin(dist_to_comk)

    # Calculate the com position relative to the comk
    com_pos_on_mask_entrance = np.zeros((len(entrance_coms), 2))
    com_pos_on_mask_exit = np.zeros((len(entrance_coms), 2))
    for i in range(len(entrance_coms)):
        com_pos_on_mask_entrance[i] = (entrance_coms_median[0] - entrance_comk[i][0], entrance_coms_median[1] - entrance_comk[i][1])
        com_pos_on_mask_exit[i] = (exit_coms[i][0] - exit_comk_median[0], exit_coms[i][1] - exit_comk_median[1])

    # Rescale values with reference index as zero
    gauge_points_entrance = np.zeros((len(entrance_coms), 2))
    gauge_distance_entrance = np.zeros(len(entrance_coms))
    gauge_points_exit = np.zeros((len(entrance_coms), 2))
    gauge_distance_exit = np.zeros(len(entrance_coms))
    for i in range(len(entrance_coms)):
        gauge_points_entrance[i] = (com_pos_on_mask_entrance[i][0] - com_pos_on_mask_entrance[reference_index][0], com_pos_on_mask_entrance[i][1] - com_pos_on_mask_entrance[reference_index][1])
        gauge_points_exit[i] = (com_pos_on_mask_exit[i][0] - com_pos_on_mask_exit[reference_index][0], com_pos_on_mask_exit[i][1] - com_pos_on_mask_exit[reference_index][1])
        gauge_distance_entrance[i] = np.sqrt(gauge_points_entrance[i][0] ** 2 + gauge_points_entrance[i][1] ** 2)
        gauge_distance_exit[i] = np.sqrt(gauge_points_exit[i][0] ** 2 + gauge_points_exit[i][1] ** 2)

    # Calculate the motor positions relative to the reference position
    gauge_distance_entrance_motor = np.zeros(len(entrance_coms))
    print("Motor positions", positions)
    for i in range(len(entrance_coms)):
        gauge_distance_entrance_motor[i] = positions[i] - positions[reference_index]

    # Check for bad measurements
    distance_avg = np.mean(gauge_distance_exit)
    is_bad = []
    for i in range(len(gauge_distance_exit)):
        if gauge_distance_exit[i] > distance_avg * 2.0:
            flag = True
            is_bad.append(flag)
        else:
            flag = False
            is_bad.append(flag)
    print("Bad measurements:", is_bad, "Exit distances:", gauge_distance_exit)

    # Remove bad measurements
    gauge_distance_entrance = np.delete(gauge_distance_entrance, np.where(is_bad))
    gauge_distance_exit = np.delete(gauge_distance_exit, np.where(is_bad))
    gauge_points_entrance = np.delete(gauge_points_entrance, np.where(is_bad), axis=0)
    gauge_points_exit = np.delete(gauge_points_exit, np.where(is_bad), axis=0)
    gauge_distance_entrance_motor = np.delete(gauge_distance_entrance_motor, np.where(is_bad))

    # Move the reference index to new position after removing bad measurements
    reference_index = np.argmin(gauge_distance_entrance)

    # Convert everything to micrometers
    gauge_points_entrance = gauge_points_entrance * 0.5169363821005045
    gauge_points_exit = gauge_points_exit * 0.439453125
    gauge_distance_entrance = gauge_distance_entrance * 0.5169363821005045
    gauge_distance_exit = gauge_distance_exit * 0.439453125
    print("Gauged motor positions pre conversion", gauge_distance_entrance_motor)
    gauge_distance_entrance_motor = gauge_distance_entrance_motor * 1e3  # Convert to micrometers from mm
    print("Gauged motor positions post conversion", gauge_distance_entrance_motor)

    # Calculate the scrambling gain
    scrambling_gain = np.zeros(len(gauge_distance_entrance))
    for i in range(len(gauge_distance_entrance)):
        if i == reference_index:
            continue

        scrambling_gain[i] = gauge_distance_entrance_motor[i] / gauge_distance_exit[i]

    # Calculate sg_min
    sg_min = np.max(gauge_distance_entrance_motor)/np.max(gauge_distance_exit)
    print(f"Scrambling gain min: {sg_min}")

    if progress_signal:
        progress_signal.emit("Scrambling gain: " + str(sg_min))

    # Write SG values to new json file
    sg_parameters = {"scrambling_gain": scrambling_gain.tolist(), "sg_min": int(sg_min), "reference_index": int(reference_index)}
    with open(os.path.join(main_folder, "scrambling_gain_new.json"), 'w') as f:
        json.dump(sg_parameters, f)

    # Remove reference index from the array
    scrambling_gain = np.delete(scrambling_gain, reference_index)
    print(scrambling_gain)

    """
    # Plot the results
    plt.scatter(gauge_points_entrance[:, 1], gauge_points_entrance[:, 0], label='Entrance Gauged Points')
    plt.scatter(0, 0, label='Reference Point', color='r')
    plt.xlabel('X-distance [μm]')
    plt.ylabel('Y-distance [μm]')
    plt.title('Entrance Gauged Points')
    plt.legend()
    plt.savefig(os.path.join(main_folder, "plots/entrance_gauged_points.png"))
    plt.close()"""

    plt.scatter(gauge_points_exit[:, 1], gauge_points_exit[:, 0], label='Exit Gauged Points')
    plt.scatter(0, 0, label='Reference Point', color='r')
    plt.xlabel('X-distance [μm]')
    plt.ylabel('Y-distance [μm]')
    plt.title('Exit Gauged Points')
    plt.legend()
    plt.savefig(os.path.join(main_folder, "plots/exit_gauged_points.png"))
    plt.close()

    """
    plt.scatter(gauge_distance_entrance, gauge_distance_exit, label='Gauged Distances')
    plt.scatter(gauge_distance_entrance[reference_index], gauge_distance_exit[reference_index], label='Reference Point', color='r')
    plt.xlabel('Entrance COM distance [μm]')
    plt.ylabel('Exit COM distance [μm]')
    plt.title('Gauged Distances')
    plt.legend()
    plt.savefig(os.path.join(main_folder, "plots/gauged_distances.png"))
    plt.close()
    """

    # Linear function for plot visuals
    def func(x, a, b):
        return a * x + b

    # Plot from "plot_sg_cool_like" function
    plt.figure(figsize=(10, 5), dpi=100)
    plt.scatter(gauge_distance_entrance_motor, gauge_points_exit[:, 1], label='X movement')
    plt.scatter(gauge_distance_entrance_motor, gauge_points_exit[:, 0], label='Y movement')
    plt.scatter(gauge_distance_entrance_motor, gauge_distance_exit, label='Total movement')
    plt.plot(gauge_distance_entrance_motor, func(gauge_distance_entrance_motor, 5e-4, 0), 'g--', label='SG 2000', alpha=1, linewidth=0.5)
    plt.plot(gauge_distance_entrance_motor, func(gauge_distance_entrance_motor, -5e-4, 0), 'g--', alpha=1, linewidth=0.5)
    plt.fill_between(gauge_distance_entrance_motor, func(gauge_distance_entrance_motor, 5e-4, 0), func(gauge_distance_entrance_motor, -5e-4, 0),
                     color='g', alpha=0.3, hatch="//", zorder=2)
    plt.plot(gauge_distance_entrance_motor, func(gauge_distance_entrance_motor, 2e-3, 0), 'r--', label='SG 500', alpha=1, linewidth=0.5)
    plt.plot(gauge_distance_entrance_motor, func(gauge_distance_entrance_motor, -2e-3, 0), 'r--', alpha=1, linewidth=0.5)
    plt.fill_between(gauge_distance_entrance_motor, func(gauge_distance_entrance_motor, 2e-3, 0), func(gauge_distance_entrance_motor, -2e-3, 0),
                     color='r', alpha=0.2, hatch="/", zorder=1)
    plt.legend()
    plt.ylim(-0.08, 0.08)
    plt.ylabel('Exit COM distance [μm]')
    plt.xlabel('Entrance Spot displacement [μm]')
    plt.title('Scrambling Gain')
    plt.savefig(os.path.join(main_folder, "plots/scrambling_gain_plot.png"))
    plt.close()

if __name__ == "__main__":
    # Example usage
    main_folder = "/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/O_40_0000_0003_test/SG"
    print(get_precise_entrance_positions(main_folder))