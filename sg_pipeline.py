import numpy as np
from PIL import Image
from astropy.io import fits
import os

import image_analysation
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, filters, morphology, measure
from skimage.draw import disk, circle_perimeter
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
        plot_mask (bool): If True, plot the detected circle.

    Returns:
        tuple: (center_y, center_x, radius) of the detected circle.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image

    # Detect edges using Canny edge detector
    edges = feature.canny(image_gray, sigma=1.6, low_threshold=0.1, high_threshold=2)

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
        total_size = int(radius * 3)  # Diagonal length of the octagon
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
        from skimage.draw import polygon
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
        best_match (np.ndarray): Image of the best match.
    """
    best_match = None
    best_score = -np.inf
    best_angle = 0
    best_location = None
    best_radius = radius

    # Detect edges using Canny edge detector
    edges = feature.canny(image, sigma=2, low_threshold=1, high_threshold=7)

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

    # Convert bool to uint8
    uint8_mask = filled_mask.astype(np.uint8)

    """if image.dtype != np.uint8:
        edges = (edges * 255).astype(np.uint8)"""

    for r in range(radius - radius_threshold, radius + radius_threshold): # Iterate over different radii
        # Create the shape template
        template = make_shape(shape,r)

        for angle in np.linspace(0, 45, num_rotations, endpoint=False): # Iterate over different rotations #Todo: Make stop value dependent on shape
            # Rotate the template
            rotated_template = transform.rotate(template, angle, resize=True)
            rotated_template = (rotated_template * 255).astype(np.uint8)

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

        # Change shape of best match to fit the image
        best_match = np.pad(best_match, ((best_location[1], image.shape[0] - best_location[1] - best_match.shape[0]),
                                         (best_location[0], image.shape[1] - best_location[0] - best_match.shape[1])),
                            mode='constant', constant_values=0)

        # Convert the best match to a binary mask
        best_match_binary = best_match > 0

        if plot_best or plot_all:
            # Plot the best match
            plt.imshow(best_match_binary)


        # Create an image with the best match removed
        image_with_mask = np.copy(image)
        image_with_mask[best_match_binary] = 0

        # Create outline of the best match
        best_match_outline = measure.find_contours(best_match, 0.5)[0]

        if plot_best or plot_all:
            # Plot filled_mask with outline
            plt.plot(best_match_outline[:, 1], best_match_outline[:, 0], 'r', linewidth=0.5)
            plt.imshow(filled_mask)
            plt.show()

            # Plot the image with the best match removed
            plt.plot(best_match_outline[:, 1], best_match_outline[:, 0], 'r', linewidth=0.1)
            plt.plot(com_position[0], com_position[1], 'ro', markersize=0.5)
            plt.imshow(image, cmap='gray')
            plt.title(f'Best match. Angle: {best_angle:.2f}°, Position: {com_position}, Radius: {best_radius}px')
            plt.show()

        return best_match
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

def detect_shape(image:np.ndarray, radius:int, shape:str, plot_mask:bool=False, plot_all:bool=False,
                 plot_best:bool=False, save_mask:str="-1"):
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
    best_match = match_shape(image, radius, shape, plot_all=plot_all, plot_best=plot_best)

    com_of_mask = com_of_spot(best_match)

    erode = morphology.binary_erosion(best_match)
    dilate = morphology.binary_dilation(best_match)

    mask = np.logical_and(dilate, ~erode)

    filled_mask = ndi.binary_fill_holes(mask)

    if plot_mask or plot_all or save_mask != "-1":
        fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharey=True)

        ax[0, 0].imshow(image, cmap="gray")
        ax[0, 0].plot(com_of_mask[1], com_of_mask[0], 'ro', markersize=0.5)
        ax[0, 0].set_title('Original Image')

        best_match_outline = measure.find_contours(best_match, 0.5)[0]
        ax[0, 1].imshow(image, cmap="gray")
        ax[0, 1].plot(best_match_outline[:, 1], best_match_outline[:, 0], 'r', linewidth=0.5)
        ax[0, 1].set_title('Best Match')

        maks_outline = measure.find_contours(mask, 0.5)[0]
        ax[1, 0].imshow(image, cmap="gray")
        ax[1, 0].plot(maks_outline[:, 1], maks_outline[:, 0], 'r', linewidth=0.5)
        ax[1, 0].set_title('Mask')

        ax[1, 1].imshow(filled_mask, cmap="gray")
        ax[1, 1].set_title('Filled Mask')

        fig.tight_layout()

        if save_mask != "-1":
            plt.savefig(save_mask)

        if plot_mask or plot_all:
            plt.show()

        plt.close()

    return filled_mask, com_of_mask

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


def com_of_spot(image:np.ndarray, plot:bool=False):
    """
    Calculate the center of mass of a spot in the given image.
    Args:
        image: The input image as a NumPy array.
        plot: If True, plot the image with the center of mass marked.

    Returns:
        com (tuple): The (y, x) coordinates of the center of mass.
    """
    # Get Area of Interest
    aoi = image_analysation.Area_of_Interest(image)

    # Get the range of the AOI
    y_range, x_range = image_analysation.narrow_index(aoi)

    # Cut the image to the AOI
    cut_image = image_analysation.cut_image(image, margin=10)

    # Get the center of mass of the cut image
    com = image_analysation.LocateFocus(cut_image)

    if plot:
        plt.figure(figsize=(.48, .50))
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
        plt.scatter(com[1], com[0], color='red', s=0.1, marker='.')  # Mark the COM with a red 'x'
        plt.title('Original Image with Center of Mass')
        plt.axis('off')
        plt.show()

    return com

def plot_circle_movement(image_folder:str, fiber_px_radius:int): # Todo: This function doesn't make sense for now,
    # due to precision issues with the circle detection also needs to be updated
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

def calculate_scrambling_gain(entrance_image_folder:str, exit_image_folder:str, fiber_diameter:int, fiber_shape:str,
                              save_mask:bool=True,
                              plot_result:bool=False, plot_mask:bool=False, plot_best:bool=False, plot_all:bool=False):
    """
    Calculate the scrambling gain of the fiber using the given images.
    Args:
        entrance_image_folder: Path to the folder containing the entrance images.
        exit_image_folder: Path to the folder containing the exit images.
        fiber_diameter: Diameter of the fiber in micrometers.
        fiber_shape: Shape of the fiber. Must be either "circle" or "octagon".
        plot_result: If True, plot the result.
        plot_mask: If True, plot the mask.
        plot_best: If True, plot the best match.
        plot_all: If True, plot all the steps.

    Returns:
    scrambling_gain: List of scrambling gains for each pair of entrance and exit images.
    """
    fiber_px_radius_entrance = int(fiber_diameter / 0.526 / 2)
    fiber_px_radius_exit = int(fiber_diameter / 0.45 / 2)

    # Get values of the entrance image
    entrance_image_files = [f for f in os.listdir(entrance_image_folder) if f.endswith('reduced.png')]
    entrance_coms = [] # Center of mass of spot in the entrance image
    entrance_comk = [] # Center of mask and with that the center of the fiber
    entrance_radii = []

    # Process entrance images
    for image_file in entrance_image_files:
        image_path = os.path.join(entrance_image_folder, image_file)
        print(image_path)
        image = io.imread(image_path)
        com = com_of_spot(image)
        entrance_coms.append(com)

        # Find center of fiber depending on the shape
        if fiber_shape == "circle":
            center_y, center_x, radius = detect_circle(image, fiber_px_radius_entrance)

            # Plot the detected circle
            plt.imshow(image, cmap='gray')
            circle_outline = plt.Circle((center_x, center_y), radius, color='r', fill=False)
            plt.gca().add_artist(circle_outline)

            if save_mask:
                mask_name = image_path.replace(".png", "_mask.png")
                plt.savefig(mask_name)
                plt.close()

            if plot_mask:
                plt.show()

            comk = [center_y, center_x]

        elif fiber_shape == "octagon":

            if save_mask:
                save_mask = image_path.replace(".png", "_mask.png")
            else:
                save_mask = -1

            mask, comk = detect_shape(image, fiber_px_radius_entrance, "octagon", plot_mask=plot_mask,
                                      plot_all=plot_all, plot_best=plot_best, save_mask=save_mask)
            radius = fiber_px_radius_entrance

        else:
            raise ValueError("Invalid fiber shape. Must be either 'circle' or 'octagon'.")

        entrance_comk.append(comk)
        entrance_radii.append(radius)

    # Convert lists to NumPy arrays
    entrance_coms = np.array(entrance_coms)
    entrance_comk = np.array(entrance_comk)
    entrance_radii = np.array(entrance_radii)

    # Get values of the exit image
    exit_image_files = [f for f in os.listdir(exit_image_folder) if f.endswith('reduced.png')]
    exit_comk = []
    exit_coms = []
    exit_radii = []

    print("Entrance center of mask:", entrance_comk)
    print("Entrance center of mass:", entrance_coms)
    print("Entrance radii:", entrance_radii)

    # Process exit images
    for image_file in exit_image_files:
        image_path = os.path.join(exit_image_folder, image_file)
        print(image_path)
        image = io.imread(image_path)

        # Find center of fiber
        if fiber_shape == "circle":
            center_y, center_x, radius = detect_circle(image, fiber_px_radius_exit)
            comk = [center_y, center_x]
            mask = create_circular_mask(image, (center_y, center_x), radius, plot_mask=plot_mask)

            # Plot the detected circle
            plt.imshow(image, cmap='gray')
            circle_outline = plt.Circle((center_x, center_y), radius, color='r', fill=False)
            plt.gca().add_artist(circle_outline)

            if save_mask:
                mask_name = image_path.replace(".png", "_mask.png")     # Todo: Also plot com of spot
                plt.savefig(mask_name)
                plt.close()

            if plot_mask:
                plt.show()

        elif fiber_shape == "octagon":

            if save_mask:
                save_mask = image_path.replace(".png", "_mask.png")
            else:
                save_mask = -1

            mask, comk = detect_shape(image, fiber_px_radius_exit, "octagon", plot_mask=plot_mask,
                                      plot_all=plot_all, plot_best=plot_best, save_mask=save_mask)
            radius = fiber_px_radius_exit

        else:
            raise ValueError("Invalid fiber shape. Must be either 'circle' or 'octagon'.")

        exit_comk.append(comk)
        exit_radii.append(radius)

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

    # Calculate distance between entrance COMK and COM
    #entrance_distances = np.linalg.norm(entrance_coms - exit_comk, axis=1)
    entrance_distances_x = entrance_coms[:, 0] - entrance_comk[:, 0]
    entrance_distances_y = entrance_coms[:, 1] - entrance_comk[:, 1]
    print("Entrance distances x,y:", entrance_distances_x, entrance_distances_y)

    # Calculate distance between exit COMK and COM
    #exit_distances = np.linalg.norm(exit_coms - exit_comk, axis=1)
    exit_distances_x = exit_coms[:, 0] - exit_comk[:, 0] #Todo: Exit shift better calculate relative to reference? Alternatively raise precision of COMK
    exit_distances_y = exit_coms[:, 1] - exit_comk[:, 1] #Todo: Also why are these values not partly negative?

    print("Exit distances x,y:", exit_distances_x, exit_distances_y)

    # Calculate total distances with sign differences depending on the direction of the movement
    entrance_distances = np.sqrt(entrance_distances_x**2 + entrance_distances_y**2)
    exit_distances = np.sqrt(exit_distances_x**2 + exit_distances_y**2)

    # Choose the exit/entrance pair with the smallest entrance distance from COMK as reference
    reference_index = np.argmin(entrance_distances)
    print("Reference index:", reference_index)

    # Change sign of entrance distance depending on the direction of the movement
    for i in range(len(entrance_distances)):
        if entrance_distances_y[i] < 0:
            entrance_distances[i] = -entrance_distances[i]


    # Calculate the scrambling gain for each pair
    scrambling_gain = []
    for i in range(len(entrance_distances)):
        scrambling_gain.append(entrance_distances[i] - entrance_distances[reference_index] / 2 * entrance_radii[i] / exit_distances[i] - exit_distances[reference_index] / 2 * exit_radii[i])
    # Todo: Maybe better use only the x or y distance for the scrambling gain calculation?

    scrambling_gain = np.array(scrambling_gain)
    scrambling_gain_rounded = np.round(scrambling_gain, 2)
    print(f"Scrambling gain: ", scrambling_gain_rounded)

    # Delete the reference element from the arrays
    scrambling_gain = np.delete(scrambling_gain, reference_index)


    if plot_result:
        entrance_distances = np.delete(entrance_distances, reference_index)
        exit_distances_y = np.delete(exit_distances_y, reference_index)
        exit_distances_x = np.delete(exit_distances_x, reference_index)

        # Plot COM x and y movement of the fiber output with different spot positions as colorbar
        scatter = plt.scatter(exit_distances_x, exit_distances_y, c=entrance_distances, cmap='plasma')
        plt.colorbar(scatter, label='Relative Spot Displacement [px]')
        plt.xlabel('Exit COM x-distance [px]')
        plt.ylabel('Exit COM y-distance [px]')
        plt.title('COM Movement of Fiber Output')
        plt.grid(True)
        result_name = os.path.join(entrance_image_folder, "scrambling_gain_result.png")
        plt.savefig(result_name)
        plt.show()


    return scrambling_gain

def main(fiber_diameter:int, fiber_shape:str, number_of_positions:int=11):
    """
    Main function to run the scrambling gain calculation pipeline.

    Args:
        fiber_diameter: Diameter of the fiber in micrometers.
        number_of_positions: Number of positions to take images at.
    """

    reduced_entrance_image_folder, reduced_exit_image_folder = capture_images_and_reduce(fiber_diameter, number_of_positions)

    sg = calculate_scrambling_gain(reduced_entrance_image_folder, reduced_exit_image_folder, fiber_diameter, fiber_shape, plot_result=True)
    print("Scrambling gain:", sg)

def capture_images_and_reduce(fiber_diameter:int, number_of_positions:int=11):
    import image_reduction
    import thorlabs_cam_control as tcc
    import file_mover
    import time
    import move_to_filter
    import step_motor_control as smc

    # Define image folders
    main_image_folder = ("D:/Vincent/thorlabs_cams_images_oct_89_other_way+camclean")
    os.makedirs(main_image_folder, exist_ok=False)
    entrance_image_folder = os.path.join(main_image_folder, "entrance")
    exit_image_folder = os.path.join(main_image_folder, "exit")
    entrance_dark_image_folder = os.path.join(entrance_image_folder, "dark")
    exit_dark_image_folder = os.path.join(exit_image_folder, "dark")
    reduced_entrance_image_folder = os.path.join(entrance_image_folder, "reduced")
    reduced_exit_image_folder = os.path.join(exit_image_folder, "reduced")
    plots_folder = os.path.join(main_image_folder, "plots")

    # Clear folders before adding new images
    file_mover.clear_folder(main_image_folder)

    # Create folders if they don't exist
    os.makedirs(entrance_image_folder, exist_ok=True)
    os.makedirs(exit_image_folder, exist_ok=True)
    os.makedirs(entrance_dark_image_folder, exist_ok=True)
    os.makedirs(exit_dark_image_folder, exist_ok=True)
    os.makedirs(reduced_entrance_image_folder, exist_ok=True)
    os.makedirs(reduced_exit_image_folder, exist_ok=True)

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
        tcc.take_image("entrance_cam", entrance_image_folder + f"/entrance_cam_image{i:03d}.png")
        tcc.take_image("exit_cam", exit_image_folder + f"/exit_cam_image{i:03d}.png")
        print(f"Image {i + 1} out of {number_of_positions} at Position {pos_left + i * step_size} done! "
              f"Move spot to next position")
        time.sleep(1)  # Probably not necessary
    print("All images taken!")

    # Reset the motor to the initial position
    smc.move_motor_to_position(5)

    # Reduce images
    for i in range(number_of_positions):
        image = png_to_numpy(entrance_image_folder + f"/entrance_cam_image{i:03d}.png")
        image_reduction.reduce_image_with_dark(image, entrance_master_dark,
                                               reduced_entrance_image_folder + f"/entrance_cam_image{i:03d}_reduced.png",
                                               save=True)
        image = png_to_numpy(exit_image_folder + f"/exit_cam_image{i:03d}.png")
        image_reduction.reduce_image_with_dark(image, exit_master_dark,
                                               reduced_exit_image_folder + f"/exit_cam_image{i:03d}_reduced.png",
                                               save=True)

    print("All images reduced!")

    return reduced_entrance_image_folder, reduced_exit_image_folder

if __name__ == '__main__':
    """
    image_path = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/entrance/entrance_cam_image1.png'
    image = io.imread(image_path)
    print(com_of_spot(image, plot=True))
    """
    entrance_folder = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/thorlabs_cams_images_test5/entrance/reduced'
    exit_folder = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/thorlabs_cams_images_test5/exit/reduced'

    #entrance_folder = "entrance_images"
    #exit_folder = "exit_images"


    fiber_diameter = 89 # Value in micrometers
    """
    # Plot com movement for entrance images
    plot_circle_movement(entrance_folder, fiber_diameter, 'entrance')
    
    # Plot com movement for exit images
    plot_circle_movement(exit_folder, fiber_diameter, 'exit')"""

    #entrance_folder = "D:/Vincent/thorlabs_cams_images/entrance/reduced"
    #exit_folder = "D:/Vincent/thorlabs_cams_images/exit/reduced"

    #sg = calculate_scrambling_gain(entrance_folder, exit_folder, fiber_diameter, fiber_shape="octagon",
    #                              plot_result=True, plot_mask=True, save_mask=True)
    #print(sg)

    #_, _ = capture_images_and_reduce(fiber_diameter, 11)

    main(fiber_diameter, "octagon", number_of_positions=11)     # Always check if main folder is empty or if
    # files are important before running

    """image = exit_folder + "/exit_cam_image000_reduced.png"
    image = entrance_folder + "/entrance_cam_image000_reduced.png"
    image = png_to_numpy(image)"""
    #binary_filtering(image, plot=True)

    #filled_mask, com_of_mask = detect_shape(image, 92, "octagon", plot_best=True)