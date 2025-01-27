from cProfile import label

import numpy as np
from PIL import Image
from astropy.io import fits
import os
import json
import image_analysation
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, filters, morphology, measure
from skimage.draw import disk, circle_perimeter, polygon
from scipy import ndimage as ndi
import cv2


def png_to_numpy(image_path):
    """
    Reads a PNG image and converts it to a NumPy array.

    Parameters:
        image_path (str): Path to the PNG image.

    Returns:
        np.ndarray: The image as a NumPy array.
    """
    with Image.open(image_path) as img:
        return np.array(img)

def detect_circle(image:np.ndarray, fiber_px_radius:int):
    """
    Detects a circle in the given image using Hough Circle Transform.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        fiber_px_radius (int): The radius of the fiber in pixels.

    Returns:
        tuple: (center_y, center_x, radius) of the detected circle.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image

    # Detect edges using Canny edge detector
    edges = feature.canny(image_gray, sigma=1.6, low_threshold=7, high_threshold=20)

    """plt.imshow(edges)
    plt.show()"""

    # Perform Hough Circle Transform
    hough_radii = np.arange(fiber_px_radius - 5, fiber_px_radius + 5, 1)
    hough_res = transform.hough_circle(edges, hough_radii)

    # Select the most prominent circle
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1, normalize=False)

    return cy[0], cx[0], radii[0]

def make_shape(shape:str, radius:int):
    """
    Create a shape for the fiber with the given radius.
    Args:
        shape: Shape form. Must be either "circle", "square" or "octagon".
        radius: Fiber radius in pixels.

    Returns:
        np.ndarray: The shape as a NumPy array.
    """
    if shape == "circle":
        mask = np.zeros((2 * radius, 2 * radius), dtype=bool)
        rr, cc = disk((radius, radius), radius)
        mask[rr, cc] = True
        """plt.imshow(mask, cmap='gray')
        plt.axis("off")
        plt.show()"""
        return mask

    if shape == "square":
        mask = np.zeros((2 * radius, 2 * radius), dtype=bool)
        mask[0:radius, 0:radius] = True

        """plt.imshow(mask, cmap='gray')
        plt.xlim(0, 2 * radius)
        plt.ylim(0, 2 * radius)
        plt.axis("off")
        plt.show()"""
        return mask

    if shape == "octagon":
        # Calculate side length from the radius
        size = 2 * radius * (np.sqrt(2) - 1)

        # Total size of the mask (to fit the octagon comfortably)
        total_size = int(radius * 4)  # Total size of the mask
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

def match_shape(image:np.ndarray, radius:int, shape:str, num_rotations:int=50, radius_threshold:int=5, plot_all:bool=False,
                plot_best:bool=False):
    """
    Match the given shape to the image using template matching.
    Args:
        image: Input image as a NumPy array.
        radius: Expected radius of the shape in pixels.
        shape: Form of the shape. Must be either "circle", "square" or "octagon".
        num_rotations: Number of rotations to try. Currently locked to 45°.
        radius_threshold: Threshold for the radius to try.
        plot_all: If True, plot all the steps.
        plot_best: If True, plot the best match.

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

    # Detect edges using Canny edge detector
    edges = feature.canny(image, sigma=2, low_threshold=1, high_threshold=15)

    # Ensure outline is closed (2 iterations of dilation and erosion)
    erode = morphology.binary_erosion(edges)
    dilate = morphology.binary_dilation(edges)
    mask_1 = np.logical_and(dilate, ~erode)

    erode = morphology.binary_erosion(mask_1)
    dilate = morphology.binary_dilation(mask_1)
    mask_2 = np.logical_and(dilate, ~erode)

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
        # Plot the filled mask
        plt.imshow(filled_mask)
        plt.title("Filled Mask")
        plt.show()

    # Set the stop value for the rotation depending on the shape
    if shape == "circle":
        stop_value = 0
    elif shape == "rectangle":
        stop_value = 180
    elif shape == "octagon":
        stop_value = 45
    else:
        raise ValueError("Invalid shape. Must be either 'circle', 'rectangle' or 'octagon")

    # Convert bool to uint8
    uint8_mask = filled_mask.astype(np.uint8)
    #uint8_mask = image

    """if image.dtype != np.uint8:
        edges = (edges * 255).astype(np.uint8)"""

    for r in range(radius - radius_threshold, radius + radius_threshold): # Iterate over different radii
        # Create the octagon
        full_oct = make_shape(shape,r)

        # Convert the full_oct to just the outline of the octagon
        erode = morphology.binary_erosion(full_oct)
        dilate = morphology.binary_dilation(full_oct)

        template = np.logical_and(dilate, ~erode)

        #print("Template shape:", template.shape)

        for angle in np.linspace(0, stop_value, num_rotations, endpoint=False): # Iterate over different rotations
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

def binary_filtering(image:np.ndarray, plot:bool=False): # Unused
    """
    Apply binary filtering to the given image.
    Args:
        image: Input image as a NumPy array.
        plot: If True, plot the filtering steps.

    Returns:

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

def build_mask(image, radius:int, shape:str, mask_margin:int, position:tuple[int,int],
               angle:int, plot_mask:bool=False, save_mask:str="-1"):
    """
    Detect the given shape in the image.
    Args:
        image: Input image as a NumPy array.
        radius: Expected radius of the shape in pixels.
        shape: Form of the shape. Must be either "circle", "square" or "octagon".
        plot_mask: If True, plot the mask.
        plot_all: If True, plot all the steps.
        plot_best: If True, plot the best match.

    Returns:
        np.ndarray: The mask of the detected shape.
        tuple: The center of mass of the detected
    """

    radius = radius + mask_margin

    best_match = make_shape(shape, radius)

    # Rotate the best match
    best_match = transform.rotate(best_match, angle, resize=False)

    # Pad the best match to fit the image while getting it on position
    pad_x_0 = position[1] - best_match.shape[1] // 2
    pad_x_1 = image.shape[0] - (position[1] + best_match.shape[1] // 2)
    pad_y_0 = position[0] - best_match.shape[0] // 2
    pad_y_1 = image.shape[1] - (position[0] + best_match.shape[0] // 2)

    best_match = np.pad(best_match, ((int(pad_x_0), int(pad_x_1)), (int(pad_y_0), int(pad_y_1)),), mode='constant')

    best_match = best_match.astype(np.uint8) * 255

    best_match_bin = best_match.astype(bool)
    filled_best_match = ndi.binary_fill_holes(best_match_bin)

    com_of_mask = com_of_spot(best_match)

    erode = morphology.binary_erosion(best_match)
    dilate = morphology.binary_dilation(best_match)

    mask = np.logical_and(dilate, ~erode)

    filled_mask = ndi.binary_fill_holes(mask)

    # Cut images for display
    image_dis = cut_image_around_comk(image, com_of_mask, radius, 20)
    best_match_dis = cut_image_around_comk(best_match, com_of_mask, radius, 20)
    mask_dis = cut_image_around_comk(mask, com_of_mask, radius, 20)
    filled_mask_dis = cut_image_around_comk(filled_mask, com_of_mask, radius, 20)

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

        if save_mask != -1:
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

def image_to_fits(image_path:str):
    """
    Converts an image array to a FITS file. Saves the image with the same name.

    Parameters:
        image_path: The path to the image file.
    """
    # Convert the image to a NumPy array
    image_array = png_to_numpy(image_path)

    # Create a PrimaryHDU object from the image array
    hdu = fits.PrimaryHDU(image_array)

    # Create an HDUList to contain the HDU
    hdul = fits.HDUList([hdu])

    fits_path = image_path.replace('.png', '.fits')

    # Write the HDUList to a FITS file
    hdul.writeto(fits_path, overwrite=True)

def plot_original_with_mask_unfilled(image:np.ndarray, center_y:int, center_x:int, radius:int):
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


def com_of_spot(image:np.ndarray, threshold=None, plot:bool=False):
    """
    Calculate the center of mass of a spot in the given image.
    Args:
        image: The input image as a NumPy array.
        plot: If True, plot the image with the center of mass marked.

    Returns:
        com (tuple): The (y, x) coordinates of the center of mass.
    """
    # Get Area of Interest
    aoi = image_analysation.Area_of_Interest(image, threshold=threshold)

    # Get the range of the AOI
    y_range, x_range = image_analysation.narrow_index(aoi)

    # Cut the image to the AOI
    cut_image = image_analysation.cut_image(image, aoi=aoi, margin=10)

    # Get the center of mass of the cut image
    com = image_analysation.LocateFocus(cut_image)

    dim = cut_image.shape
    size = [dim[1] / 100, dim[0] / 100]


    if plot:
        plt.figure(figsize=size)
        plt.imshow(cut_image, cmap='gray', origin='lower')
        plt.scatter(com[1], com[0], color='red', s=0.1, marker='.')  # Mark the COM with a red 'x'
        plt.title('Cut Image with Center of Mass')
        plt.axis('off')
        plt.show()

    # Adjust the center of mass to the original image, also adjust for the margin
    com = [com[0] + y_range[0] - 10, com[1] + x_range[0] - 10]

    if plot:
        plt.figure(figsize=(12.80, 10.24))
        plt.imshow(image, cmap='gray', origin='lower')
        plt.scatter(com[1], com[0], color='red', s=0.5, marker='.')  # Mark the COM with a red 'x'
        plt.title('Original Image with Center of Mass')
        plt.axis('off')
        plt.show()

    return com

def plot_circle_movement(image_folder:str, fiber_px_radius:int): # Todo: This function doesn't make sense for now,
    # due to precision issues with the circle detection also needs to be updated generally
    """
    Plots the movement of the center of circle of the fibers in the given image folder.
    Args:
        image_folder: Path to the folder containing the images.
        fiber_px_radius: Radius of the fiber in pixels.

    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

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

def get_sg_params(main_folder:str, fiber_diameter:int, fiber_shape:str,
                              save_mask:bool=True,
                              plot_result:bool=False, plot_mask:bool=False, plot_best:bool=False, plot_all:bool=False):
    """
    Calculate the scrambling gain of the fiber using the given images.
    Args:
        main_folder: Path to the main folder containing the entrance and exit images.
        fiber_diameter: Diameter of the fiber in micrometers.
        fiber_shape: Shape of the fiber. Must be either "circle" or "octagon".
        save_mask: If True, save the mask as png.
        plot_result: If True, plot the result.
        plot_mask: If True, plot the mask.
        plot_best: If True, plot the best match.
        plot_all: If True, plot all the steps.

    Returns:
    scrambling_gain: List of scrambling gains for each pair of entrance and exit images.
    """
    entrance_image_folder_reduced = os.path.join(main_folder, "entrance/reduced")
    exit_image_folder_reduced = os.path.join(main_folder, "exit/reduced")
    entrance_mask_folder = os.path.join(main_folder, "entrance/mask")
    exit_mask_folder = os.path.join(main_folder, "exit/mask")

    # Create mask folders
    os.makedirs(entrance_mask_folder, exist_ok=True)
    os.makedirs(exit_mask_folder, exist_ok=True)

    # Calculate the radius of the fiber in pixels
    fiber_px_radius_entrance = int(fiber_diameter / 0.526 / 2)
    fiber_px_radius_exit = int(fiber_diameter / 0.45 / 2)

    mask_scale_factor_circ = 1.05
    mask_scale_factor_oct = 1.08

    # Get values of the entrance image
    entrance_image_files = [f for f in os.listdir(entrance_image_folder_reduced) if f.endswith('reduced.png')]
    entrance_coms = [] # Center of mass of spot in the entrance image
    entrance_comk = [] # Center of mask and with that the center of the fiber
    entrance_radii = []

    # Process entrance images
    for image_file in entrance_image_files:
        image_path = os.path.join(entrance_image_folder_reduced, image_file)
        print(image_path)
        image = io.imread(image_path)
        com = com_of_spot(image, threshold=200)
        entrance_coms.append(com)
        mask_path = os.path.join(entrance_mask_folder, image_file.replace(".png", "_mask.png"))

        # Find center of fiber depending on the shape
        if fiber_shape == "circle":
            center_y, center_x, radius = detect_circle(image, fiber_px_radius_entrance)

            mask = create_circular_mask(image, (center_y, center_x), radius, plot_mask=plot_mask)
            io.imsave(mask_path, mask.astype(np.uint8) * 255)

            # Plot the detected circle
            plt.imshow(image, cmap='gray')
            circle_outline = plt.Circle((center_x, center_y), radius, color='r', fill=False)
            plt.gca().add_artist(circle_outline)

            if save_mask:
                mask_name = image_path.replace(".png", "_mask.png")
                plt.savefig(mask_name)

            if plot_mask:
                plt.show()

            plt.close()

            comk = [center_y, center_x]

        elif fiber_shape == "octagon":

            if save_mask:
                save_mask = image_path.replace(".png", "_mask.png")
            else:
                save_mask = -1

            angle, position, radius = match_shape(image, fiber_px_radius_entrance, fiber_shape, plot_all=plot_all, plot_best=plot_best)

            mask, comk = build_mask(image, radius, "octagon", 1, position, angle, plot_mask=plot_mask,
                                    save_mask=save_mask)

            io.imsave(mask_path, mask.astype(np.uint8) * 255)

        else:
            raise ValueError("Invalid fiber shape. Must be either 'circle' or 'octagon'.")

        entrance_comk.append(comk)
        entrance_radii.append(radius)

    # Convert lists to NumPy arrays
    entrance_coms = np.array(entrance_coms)
    entrance_comk = np.array(entrance_comk)
    entrance_radii = np.array(entrance_radii)

    # Get values of the exit image
    exit_image_files = [f for f in os.listdir(exit_image_folder_reduced) if f.endswith('reduced.png')]
    exit_comk = []
    exit_coms = []
    exit_radii = []
    max_flux_list = []

    print("Entrance center of mask:", entrance_comk)
    print("Entrance center of mass:", entrance_coms)
    print("Entrance radii:", entrance_radii)

    # Process exit images
    for image_file in exit_image_files:
        image_path = os.path.join(exit_image_folder_reduced, image_file)
        print(image_path)
        image = io.imread(image_path)

        mask_path = os.path.join(exit_mask_folder, image_file.replace(".png", "_mask.png"))

        # Find center of fiber
        if fiber_shape == "circle":
            center_y, center_x, radius = detect_circle(image, fiber_px_radius_exit)
            comk = [center_y, center_x]

            mask_margin_circ = grow_mask(image, comk, radius, "circle")

            mask = create_circular_mask(image, (center_y, center_x), radius + mask_margin_circ, plot_mask=plot_mask)

        elif fiber_shape == "octagon":

            if save_mask:
                save_mask = image_path.replace(".png", "_mask.png")
            else:
                save_mask = -1

            angle, position, radius = match_shape(image, fiber_px_radius_exit, fiber_shape, plot_all=plot_all,
                                                  plot_best=plot_best)

            mask_margin_oct = grow_mask(image, position, radius, "octagon", angle=angle)

            mask, comk = build_mask(image, radius, "octagon", mask_margin_oct, position, angle, plot_mask=plot_mask,
                                    save_mask=save_mask)

            radius = fiber_px_radius_exit

            io.imsave(mask_path, mask.astype(np.uint8) * 255)

        else:
            raise ValueError("Invalid fiber shape. Must be either 'circle' or 'octagon'.")

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

        """print(com)
        # Save image
        io.imsave(main_folder+"/test_image_com.png", image)


        plt.imshow(image, cmap='gray')
        plt.plot(com, 'ro', markersize=0.5)
        plt.title("Reduced exit image with mask")
        plt.show()"""

    exit_comk = np.array(exit_comk)
    exit_coms = np.array(exit_coms)
    exit_radii = np.array(exit_radii)

    print("Exit center of mask:", exit_comk)
    print("Exit center of mass:", exit_coms)
    print("Exit radii:", exit_radii)

    # Calculate distance between entrance COMK and COM
    #entrance_distances = np.linalg.norm(entrance_coms - exit_comk, axis=1)
    entrance_distances_x = entrance_coms[:, 0] - entrance_comk[:, 0]
    entrance_distances_y = entrance_coms[:, 1] - entrance_comk[:, 1]
    print("Entrance distances x,y:", entrance_distances_x, entrance_distances_y)

    # Calculate distance between exit COMK and COM
    #exit_distances = np.linalg.norm(exit_coms - exit_comk, axis=1)
    exit_distances_x = exit_coms[:, 0] - exit_comk[:, 0] #Todo: Exit shift better calculate relative to reference? Alternatively raise precision of COMK
    exit_distances_y = exit_coms[:, 1] - exit_comk[:, 1] #Todo: Also why are these values not partly negative? Negativity depends on the rel position

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

def calc_sg(main_folder:str, plot_result:bool=False):

    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    entrance_coms = parameters["entrance_coms"]
    entrance_comk = parameters["entrance_comk"]
    entrance_radii = parameters["entrance_radii"]
    entrance_distances_x = parameters["entrance_distances_x"]
    entrance_distances_y = parameters["entrance_distances_y"]
    entrance_distances = parameters["entrance_distances"]
    exit_coms = parameters["exit_coms"]
    exit_comk = parameters["exit_comk"]
    exit_radii = parameters["exit_radii"]
    exit_distances_x = parameters["exit_distances_x"]
    exit_distances_y = parameters["exit_distances_y"]
    exit_distances = parameters["exit_distances"]
    reference_index = parameters["reference_index"]
    max_flux = parameters["max_flux"]
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
        scrambling_gain.append(
            (entrance_distances[i] / 2 * fiber_px_radius_entrance) / (exit_distances[i] / 2 * fiber_px_radius_exit))

    sg_min = np.max(entrance_distances / fiber_px_radius_entrance) / np.max(exit_distances / fiber_px_radius_exit)
    print(f"SG_min value: {sg_min}")

    scrambling_gain = np.array(scrambling_gain)
    scrambling_gain_rounded = np.round(scrambling_gain, 2)
    print(f"Scrambling gain: ", scrambling_gain_rounded)

    # Delete the reference element from the arrays
    scrambling_gain = np.delete(scrambling_gain, reference_index)

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


def main(fiber_diameter:int, fiber_shape:str, number_of_positions:int=11):
    """
    Main function to run the scrambling gain calculation pipeline.

    Args:
        fiber_diameter: Diameter of the fiber in micrometers.
        fiber_shape: Shape of the fiber. Must be either "circle" or "octagon".
        number_of_positions: Number of positions to take images at.
    """

    main_image_folder = "D:/Vincent/thorlabs_cams_images_oct_89_other_way+camclean"

    capture_images_and_reduce(fiber_diameter, main_image_folder, number_of_positions)

    get_sg_params(main_image_folder, fiber_diameter, fiber_shape, plot_result=True)

    calc_sg(main_image_folder, plot_result=True)

def capture_images_and_reduce(fiber_diameter:int, main_image_folder:str, number_of_positions:int=11):
    """
    Capture images and reduce them for the scrambling gain calculation
    Args:
        fiber_diameter: Diameter of the fiber in micrometers.
        number_of_positions: Number of positions to take images at.

    Returns:
        tuple: Paths to the reduced entrance and exit image folders.
    """
    import image_reduction
    import thorlabs_cam_control as tcc
    import file_mover
    import time
    import move_to_filter
    import step_motor_control as smc

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
    file_mover.clear_folder(main_image_folder)

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

    move_to_filter.move("none")
    print("Taking darks")

    # Take darks
    for i in range(5):
        tcc.take_image("entrance_cam", entrance_dark_image_folder + f"/entrance_cam_dark{i:03d}.png")
        tcc.take_image("exit_cam", exit_dark_image_folder + f"/exit_cam_dark{i:03d}.png")

    entrance_master_dark = (image_reduction
                            .create_master_dark(entrance_dark_image_folder, plot=False))
    exit_master_dark = image_reduction.create_master_dark(exit_dark_image_folder, plot=False)

    move_to_filter.move("0")

    smc.make_reference_move()

    print("Taking images")

    step_size = fiber_diameter / 1000 * 0.8 / (number_of_positions-1)  # Step size in mm
    pos_left = 5 - fiber_diameter / 1000 * 0.8 / 2  # Leftmost position in mm

    # Take images
    for i in range(number_of_positions):
        print(i, pos_left + i * step_size)
        smc.move_motor_to_position(pos_left + i * step_size)
        tcc.take_image("entrance_cam", entrance_light_folder + f"/entrance_cam_image{i:03d}.png")
        tcc.take_image("exit_cam", exit_light_folder + f"/exit_cam_image{i:03d}.png")
        print(f"Image {i + 1} out of {number_of_positions} at Position {pos_left + i * step_size} done! "
              f"Move spot to next position")
        time.sleep(1)  # Probably not necessary
    print("All images taken!")

    # Reset the motor to the initial position
    smc.move_motor_to_position(5)

    # Reduce images
    for i in range(number_of_positions):
        image = png_to_numpy(entrance_light_folder + f"/entrance_cam_image{i:03d}.png")
        image_reduction.reduce_image_with_dark(image, entrance_master_dark,
                                               reduced_entrance_image_folder + f"/entrance_cam_image{i:03d}_reduced.png",
                                               save=True)
        image = png_to_numpy(exit_light_folder + f"/exit_cam_image{i:03d}.png")
        image_reduction.reduce_image_with_dark(image, exit_master_dark,
                                               reduced_exit_image_folder + f"/exit_cam_image{i:03d}_reduced.png",
                                               save=True)

    print("All images reduced!")

def cut_image_around_comk(image, comk, fiber_px_radius, margin):
    """
    Cut the image around the center of the mask.
    Args:
        image: The input image as a NumPy array.
        comk: The center of the mask.
        fiber_px_radius: The radius of the fiber in pixels.

    Returns:
        np.ndarray: The cut image.
    """
    cut_image = image[int(comk[0]) - fiber_px_radius - margin:int(comk[0])  + fiber_px_radius + margin,
                int(comk[1]) - fiber_px_radius- margin:int(comk[1]) + fiber_px_radius + margin]

    return cut_image

def make_comparison_video(main_folder:str, fiber_diameter):
    entrance_image_folder = os.path.join(main_folder, "entrance/reduced")
    exit_image_folder = os.path.join(main_folder, "exit/reduced")
    video_prep_exit_folder = os.path.join(main_folder, "video_prep/exit")
    video_prep_entrance_folder = os.path.join(main_folder, "video_prep/entrance")
    sg_parameters_file = os.path.join(main_folder, "scrambling_gain_parameters.json")

    entrance_image_files = [f for f in os.listdir(entrance_image_folder) if f.endswith('reduced.png')]
    exit_image_files = [f for f in os.listdir(exit_image_folder) if f.endswith('reduced.png')]

    fiber_input_radius = int(fiber_diameter / 0.526 / 2)
    fiber_exit_radius = int(fiber_diameter / 0.45 / 2)

    # Margin for better visuals
    margin = 20

    # Create folders if they don't exist
    os.makedirs(video_prep_entrance_folder, exist_ok=True)
    os.makedirs(video_prep_exit_folder, exist_ok=True)

    # Read entrance_comk and exit_comk from json file
    with open(sg_parameters_file, 'r') as f:
        parameters = json.load(f)

    entrance_comk = parameters["entrance_comk"]
    exit_comk = parameters["exit_comk"]

    # Cut the images to the same size around the fiber
    for i in range(len(entrance_image_files)):
        comk = entrance_comk[i]
        image_file = entrance_image_files[i]

        image_path = os.path.join(entrance_image_folder, image_file)
        image = io.imread(image_path)
        image = cut_image_around_comk(image, comk, fiber_input_radius, margin)
        io.imsave(os.path.join(video_prep_entrance_folder, image_file.replace(".png","_cut.png")), image)

    for i in range(len(exit_image_files)):
        comk = exit_comk[i]
        image_file = exit_image_files[i]

        image_path = os.path.join(exit_image_folder, image_file)
        image = io.imread(image_path)
        image = cut_image_around_comk(image, comk, fiber_exit_radius, margin)
        io.imsave(os.path.join(video_prep_exit_folder, image_file.replace(".png","_cut.png")), image)

    # No scaling needed, done in video creation

    # Create video
    from moviepy import VideoFileClip, clips_array
    import imageio

    video_name = os.path.join(main_folder, "comparison_video.mp4")

    video_entrance_files = [f for f in os.listdir(video_prep_entrance_folder) if f.endswith('cut.png')]
    video_exit_files = [f for f in os.listdir(video_prep_exit_folder) if f.endswith('cut.png')]

    # Create entrance video
    entrance_video_path = os.path.join(video_prep_entrance_folder, "entrance_video.mp4")
    imageio.mimsave(entrance_video_path,
                    [io.imread(os.path.join(video_prep_entrance_folder, f)) for f in video_entrance_files], fps=5)

    # Create exit video
    exit_video_path = os.path.join(video_prep_exit_folder, "exit_video.mp4")
    imageio.mimsave(exit_video_path,
                    [io.imread(os.path.join(video_prep_exit_folder, f)) for f in video_exit_files], fps=5)

    # Load videos
    entrance_clip = VideoFileClip(entrance_video_path)
    exit_clip = VideoFileClip(exit_video_path)

    # Combine videos side by side
    final_clip = clips_array([[entrance_clip, exit_clip]])
    final_clip.write_videofile(video_name, fps=5)

def plot_masks(main_folder:str, fiber_diameter:int):
    fiber_input_radius = int(fiber_diameter / 0.526 / 2)
    fiber_exit_radius = int(fiber_diameter / 0.45 / 2)

    entrance_mask_folder = os.path.join(main_folder, "entrance/mask")
    exit_mask_folder = os.path.join(main_folder, "exit/mask")
    plot_folder = os.path.join(main_folder, "plots")

    # Create plot folder if it doesn't exist
    os.makedirs(plot_folder, exist_ok=True)

    entrance_mask_files = [f for f in os.listdir(entrance_mask_folder) if f.endswith('mask.png')]
    exit_mask_files = [f for f in os.listdir(exit_mask_folder) if f.endswith('mask.png')]

    # Read entrance coms from json file
    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    entrance_coms = parameters["entrance_coms"]
    entrance_comk = parameters["entrance_comk"]
    exit_comk = parameters["exit_comk"]

    margin = 20

    # Plot Mask outline overlaid on the image
    for i in range(len(entrance_mask_files)):
        com = entrance_coms[i]
        comk = entrance_comk[i]

        entrance_mask = io.imread(os.path.join(entrance_mask_folder, entrance_mask_files[i]))
        entrance_image = io.imread(os.path.join(main_folder, "entrance/reduced", entrance_mask_files[i].replace("_mask", "")))

        # Cut image to the fiber size
        entrance_image_cut = cut_image_around_comk(entrance_image, comk, fiber_input_radius, margin)

        # Adjust com to the cut image
        com = [com[0] - comk[0] + fiber_input_radius + margin, com[1] - comk[1] + fiber_input_radius + margin]

        # Cut the mask to the fiber size
        entrance_mask_cut = cut_image_around_comk(entrance_mask, comk, fiber_input_radius, margin)

        entrance_mask_outline = measure.find_contours(entrance_mask_cut, 0.5)[0]

        dpi = 100
        size = [entrance_image_cut.shape[1] / dpi, float(entrance_image_cut.shape[0] / dpi)]

        # noinspection PyTypeChecker
        plt.figure(figsize=size, dpi = dpi)
        plt.imshow(entrance_image_cut, cmap='gray')
        plt.scatter(com[1], com[0], color='r', s=0.5)
        plt.plot(entrance_mask_outline[:, 1], entrance_mask_outline[:, 0], 'r', linewidth=0.5)
        plt.title('Entrance Mask Overlay')
        plt.axis('off')
        plt.savefig(os.path.join(plot_folder, entrance_mask_files[i].replace(".png", "_overlay.png")), dpi="figure")
        plt.close()
        #plt.show()

    for i in range(len(exit_mask_files)):
        comk = exit_comk[i]
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
        plt.savefig(os.path.join(plot_folder, exit_mask_files[i].replace(".png", "_overlay.png")), dpi="figure")
        plt.close()
        #plt.show()

def check_mask_flux_all(main_folder:str):
    exit_mask_folder = os.path.join(main_folder, "exit/mask")
    exit_image_folder = os.path.join(main_folder, "exit/reduced")

    exit_mask_files = [f for f in os.listdir(exit_mask_folder) if f.endswith('mask.png')]
    exit_image_files = [f for f in os.listdir(exit_image_folder) if f.endswith('reduced.png')]

    for i in range(len(exit_mask_files)):
        mask = io.imread(os.path.join(exit_mask_folder, exit_mask_files[i]))
        image = io.imread(os.path.join(exit_image_folder, exit_image_files[i]))

        check_mask_flux_single(image, mask)

def check_mask_flux_single(image:np.ndarray, mask:np.ndarray, plot:bool=False, print_text:bool=True):
    """
    Check the flux of the mask in the image.
    Args:
        image: The input image as a NumPy array.
        mask: The mask as a NumPy array.
        plot: If True, plot the image with the mask removed.
    """
    # Convert mask to bool
    mask = mask.astype(bool)

    image_wo_mask = image.copy()
    image_wo_mask[mask] = 0

    if plot:
        plt.imshow(image_wo_mask, cmap='gray')
        plt.title('Image with Mask Removed')
        plt.show()

    if print_text:
        print("Max flux value outside of mask:", np.max(image_wo_mask))

    return np.max(image_wo_mask)

def plot_sg_cool_like(main_folder:str, fiber_diameter:int):
    """
    Plot the scrambling gain in a cool way.
    Args:
        main_folder: Path to the main folder containing the images.
    """
    with open(os.path.join(main_folder, "scrambling_gain_parameters.json"), 'r') as f:
        parameters = json.load(f)

    entrance_coms = parameters["entrance_coms"]
    entrance_comk = parameters["entrance_comk"]
    entrance_distances_x = parameters["entrance_distances_x"]
    entrance_distances_y = parameters["entrance_distances_y"]
    exit_coms = parameters["exit_coms"]
    exit_comk = parameters["exit_comk"]
    exit_distances_x = parameters["exit_distances_x"]
    exit_distances_y = parameters["exit_distances_y"]
    reference_index = parameters["reference_index"]
    max_flux = parameters["max_flux"]
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
    entrance_distances = np.array(entrance_distances) * 0.45
    exit_distances = np.array(exit_distances) * 0.526
    exit_distances_x = np.array(exit_distances_x) * 0.526
    exit_distances_y = np.array(exit_distances_y) * 0.526

    def func(x, a, b):
        return a * x + b

    def sg_func(d_in, D_in, d_out, D_out):
        return (d_in / D_in) / (d_out / D_out)

    print(entrance_distances)

    for i in range(len(entrance_distances)):
        if i == reference_index:
            continue
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
    plt.savefig(os.path.join(main_folder, "plots/scrambling_gain_plot.png"))
    plt.close()
    #plt.show()

def grow_mask(image, position, radius, shape, angle = 0):
    min_flux = np.inf

    center_y, center_x = position[0], position[1]

    for margin in range(0, 1000, 2):
        if shape == "circle":
            mask = create_circular_mask(image, (center_y, center_x), radius + margin, plot_mask=False)

        elif shape == "octagon":
            mask, comk = build_mask(image, radius, "octagon", margin, position, angle)

        else:
            raise ValueError("Invalid fiber shape. Must be either 'circle' or 'octagon'.")


        # Check flux outside mask
        max_flux = int(check_mask_flux_single(image, mask, plot=False, print_text=False))

        if max_flux < min_flux:
            min_flux = max_flux

        elif max_flux == min_flux:
            print(f"Margin: {margin}, Flux: {max_flux}")
            return margin

        elif radius + margin > 1.2 * radius:
            print(f"Max margin reached. Margin: {margin}, Flux: {max_flux}")
            return margin

def plot_coms(main_folder):
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

    plt.scatter(entrance_coms[:, 1], entrance_coms[:, 0], label="COMs")
    for i, com in enumerate(entrance_coms):
        plt.text(com[1], com[0], str(i), fontsize=8, ha='right')
    plt.scatter(entrance_comk[:, 1], entrance_comk[:, 0], label="COMKs")
    plt.legend()
    plt.title("Entrance")
    plt.savefig(os.path.join(main_folder, "plots/Entrance_coms.png"))
    plt.close()
    #plt.show()

    plt.figure()
    plt.scatter(exit_coms[:, 1], exit_coms[:, 0], label="COMs")
    plt.scatter(exit_comk[:, 1], exit_comk[:, 0], label="COMKs")
    ref_com = exit_coms[reference_index]
    plt.text(ref_com[1], ref_com[0], "Ref", fontsize=8, ha='right')
    ref_comk = exit_comk[reference_index]
    plt.text(ref_comk[1], ref_comk[0], "Ref", fontsize=8, ha='right')
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.legend()
    plt.title("Exit")
    plt.savefig(os.path.join(main_folder, "plots/Exit_coms.png"))
    plt.close()

if __name__ == '__main__':

    #image_path = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/thorlabs_cams_images_oct_89_other_way+camclean/exit/reduced/exit_cam_image000_reduced.png'
    #image = io.imread(image_path)
    #print(com_of_spot(image, plot=True))

    #image_to_fits(image_path)

    entrance_folder = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/thorlabs_cams_images_test5/entrance/reduced'
    exit_folder = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/thorlabs_cams_images_test5/exit/reduced'

    main_folder = "E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/thorlabs_cams_images_oct_89_other_way+camclean"

    main_folder = "D:/Vincent/oct_89/SG"

    # entrance_folder = "entrance_images"
    # exit_folder = "exit_images"

    fiber_diameter = 89  # Value in micrometers
    """
    # Plot com movement for entrance images
    plot_circle_movement(entrance_folder, fiber_diameter, 'entrance')
    
    # Plot com movement for exit images
    plot_circle_movement(exit_folder, fiber_diameter, 'exit')"""

    #entrance_folder = "D:/Vincent/thorlabs_cams_images/entrance/reduced"
    #exit_folder = "D:/Vincent/thorlabs_cams_images/exit/reduced"

    #get_sg_params(main_folder, fiber_diameter, fiber_shape="octagon", plot_all=False, plot_result=True, plot_mask=True, save_mask=False)

    #calc_sg(main_folder, plot_result=True)

    plot_masks(main_folder, fiber_diameter)

    #_, _ = capture_images_and_reduce(fiber_diameter, 11)

    #main(fiber_diameter, "octagon", number_of_positions=11)     # Always check if main folder is empty or if
    # files are important before running

    #make_comparison_video(main_folder, fiber_diameter)
    #check_mask_flux_all(main_folder)
    #plot_sg_cool_like(main_folder, fiber_diameter)
    #make_shape("octagon", 100)

    #angle, position, radius = match_shape(image, 96, "octagon", plot_all=False, plot_best=False)
    #grow_mask(image, position, radius, "octagon", angle)

    #match_shape(image, 89, "octagon", plot_all=True, plot_best=True)

    #plot_coms(main_folder)
