import numpy as np
from PIL import Image
from astropy.io import fits
import os

from numpy import ndarray, dtype
from numpy.f2py.symbolic import number_types
from numpy.lib._function_base_impl import _SCT

import image_analysation
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform, filters
from skimage.draw import disk, circle_perimeter


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

def detect_circle(image:np.ndarray, fibre_diameter:int, cam_type:str, threshold_value=20):
    """
    Detects a circle in the given image using Hough Circle Transform.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        fibre_diameter (int): The diameter of the fibre in micrometers.
        cam_type (str): The type of camera used. Must be either 'exit' or 'entrance'.
        threshold_value (float): The threshold value to filter out the brighter spot.

    Returns:
        tuple: (center_y, center_x, radius) of the detected circle.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image

    """
    # Apply a threshold to filter out the brighter spot
    image_gray = filters.threshold_local(image_gray, block_size=11, offset=0)

    plt.imshow(image_gray, cmap='gray')
    plt.show()
    print(image_gray)
    image_gray[image_gray > threshold_value] = 0
    print(image_gray)

    plt.imshow(image_gray, cmap='gray')
    plt.show()
    """

    # Detect edges using Canny edge detector
    edges = feature.canny(image_gray, sigma=1.6, low_threshold=0.1, high_threshold=2)

    """plt.imshow(edges)
    plt.show()"""

    if cam_type == 'exit':
        px_radius = int(fibre_diameter / 0.45 / 2) # 0.45 um/px #Todo: Are these values correct?
    elif cam_type == 'entrance':
        px_radius = int(fibre_diameter / 0.526 / 2) # 0.526 um/px
    else:
        raise ValueError("Invalid camera type. Must be either 'exit' or 'entrance'.")

    print(px_radius)

    # Perform Hough Circle Transform
    hough_radii = np.arange(px_radius - 5, px_radius + 5, 1)
    hough_res = transform.hough_circle(edges, hough_radii)

    # Select the most prominent circle
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1, normalize=False)

    return cy[0], cx[0], radii[0]

def create_circular_mask(image:np.ndarray, center:tuple[int,int], radius:int, margin=0):
    """
    Creates a circular mask for the given image.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        center (tuple): The (y, x) coordinates of the circle center.
        radius (int): The radius of the circular mask.
        margin (int): The margin to add to the radius.

    Returns:
        np.ndarray: The circular mask as a NumPy array.
    """
    from skimage.draw import disk

    mask = np.zeros(image.shape[:2], dtype=bool)
    rr, cc = disk(center, radius + margin, shape=image.shape)
    mask[rr, cc] = True
    return mask

def plot_histogram(image:np.ndarray):
    """
    Plots a histogram of the given image.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
    """
    plt.hist(image.ravel(), bins=256, density=True, color='gray', alpha=0.75)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')
    plt.show()


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

def plot_circle_movement(image_folder:str, fibre_diameter:int, cam_type:str): # Todo: This function doesn't make sense for now, due to precision issues with the circle detection
    """
    Plots the movement of the center of circle of the fibers in the given image folder.
    Args:
        image_folder: Path to the folder containing the images.
        fibre_diameter: Diameter of the fiber in micrometers.
        cam_type: Which camera type (position) was used. Must be either 'exit' or 'entrance'.

    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    cpos = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = io.imread(image_path)
        center_y, center_x, radius = detect_circle(image, fibre_diameter, cam_type)
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

def calculate_scrambling_gain(entrance_image_folder:str, exit_image_folder:str, fibre_diameter:int) -> ndarray[
    tuple[int, ...], dtype[_SCT]]:
    """
    Calculate the scrambling gain of the fiber using the given images.
    Args:
        entrance_image_folder: Path to the folder containing the entrance images.
        exit_image_folder: Path to the folder containing the exit images.
        fibre_diameter: Diameter of the fiber in micrometers.

    Returns:
    scrambling_gain: List of scrambling gains for each pair of entrance and exit images.
    """
    # Get the COM of the entrance image spots
    entrance_image_files = [f for f in os.listdir(entrance_image_folder) if f.endswith('.png')]
    entrance_coms = []
    entrance_cocs = []
    entrance_radii = []

    for image_file in entrance_image_files:
        image_path = os.path.join(entrance_image_folder, image_file)
        print(image_path)
        image = io.imread(image_path)
        com = com_of_spot(image)
        entrance_coms.append(com)

        # Find center of circle
        center_y, center_x, radius = detect_circle(image, fibre_diameter, cam_type='entrance')
        coc = [center_y, center_x]
        entrance_cocs.append(coc)
        entrance_radii.append(radius)

    entrance_coms = np.array(entrance_coms)
    entrance_cocs = np.array(entrance_cocs)
    entrance_radii = np.array(entrance_radii)

    # Get the C.O.C (center of circle) of the exit image
    exit_image_files = [f for f in os.listdir(exit_image_folder) if f.endswith('.png')]
    exit_cocs = []
    exit_coms = []
    exit_radii = []

    print("entrance:", entrance_cocs, entrance_coms, entrance_radii)

    for image_file in exit_image_files:
        image_path = os.path.join(exit_image_folder, image_file)
        print(image_path)
        image = io.imread(image_path)
        center_y, center_x, radius = detect_circle(image, fibre_diameter, cam_type='exit')
        coc = [center_y, center_x]
        exit_cocs.append(coc)
        exit_radii.append(radius)

        # Use exit circle mask to set background to zero
        mask = create_circular_mask(image, (center_y, center_x), radius, margin=0)
        image[~mask] = 0

        # Find the center of mass of the image with reduced background
        com = com_of_spot(image)
        exit_coms.append(com)

    exit_cocs = np.array(exit_cocs)
    exit_coms = np.array(exit_coms)
    exit_radii = np.array(exit_radii)

    print("exit:", exit_cocs, exit_coms, exit_radii)

    # Calculate distance between entrance COC and COM
    entrance_distances = np.linalg.norm(entrance_coms - entrance_cocs, axis=1)
    print("entrance distances:", entrance_distances)

    # Calculate distance between exit COC and COM
    exit_distances = np.linalg.norm(exit_coms - exit_cocs, axis=1)
    print("exit distances:", exit_distances)

    # Choose the exit/entrance pair with the smallest distance entrance distance from coc as reference
    reference_index = np.argmin(entrance_distances)
    print("Reference index:", reference_index)

    # Calculate the scrambling gain for each pair
    scrambling_gain = []
    for i in range(len(entrance_distances)):
        scrambling_gain.append(entrance_distances[i] - entrance_distances[reference_index] / 2 * entrance_radii[i] / exit_distances[i] - exit_distances[reference_index] / 2 * exit_radii[i])

    scrambling_gain = np.array(scrambling_gain)
    print("Scrambling gain:", scrambling_gain)

    # Delete the element which is zero
    scrambling_gain = np.delete(scrambling_gain, reference_index)

    return scrambling_gain

def main(fiber_diameter:int):
    """
    Main function to run the scrambling gain calculation pipeline.
    """
    import image_reduction
    import thorlabs_cam_control as tcc
    import time

    # Clear the image folders

    # Todo: define clear save folders

    # Filter wheel to 0

    # Take darks #Todo background subtraction, use filter wheel for no dark
    for i in range(4):
        tcc.take_image("entrance_cam",f"entrance_images/darks/entrance_cam_dark{i}.png")
        tcc.take_image("exit_cam",f"exit_images/darks/dark{i}.png")

    master_dark = image_reduction.create_master_dark("entrance_images/darks", plot=False)

    # Take images
    number_of_images = 2
    for i in range(number_of_images):
        tcc.take_image("entrance_cam",f"entrance_images/entrance_cam_image{i}.png")
        tcc.take_image("exit_cam",f"exit_images/exit_cam_image{i}.png")
        time.sleep(30) # Time to move spot manually Todo: Automate spot movement

    # Reduce images
    for i in range(number_of_images):
        image = png_to_numpy(f"exit_images/entrance_cam_image{i}.png")
        image_reduction.reduce_image_with_dark(image, master_dark, f"entrance_images/reduced/entrance_cam_image{i}_reduced.png", save=True)
        image = png_to_numpy(f"entrance_images/exit_cam_image{i}.png")
        image_reduction.reduce_image_with_dark(image, master_dark, f"exit_images/reduced/exit_cam_image{i}_reduced.png", save=True)


    sg = calculate_scrambling_gain("entrance_images/reduced", "exit_images/reduced", fiber_diameter) # Todo: get fiber diameter from input GUI
    print("Scrambling gain:", sg)

# Test functions
# home path: E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images
def process_these_images(): # Just for testing purposes
    image_folder = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/exit'
    fibre_diameter = 100
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = io.imread(image_path)
        center_y, center_x, radius = detect_circle(image, fibre_diameter, cam_type='exit')
        plot_original_with_mask_unfilled(image, center_y, center_x, radius)

        print(f"Detected circle: Center: ({center_y:.2f}, {center_x:.2f}), Radius: {radius:.2f}")

    # Same procedure for entrance images
    image_folder = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/entrance'

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = io.imread(image_path)
        center_y, center_x, radius = detect_circle(image, fibre_diameter, cam_type='entrance')
        plot_original_with_mask_unfilled(image, center_y, center_x, radius)

        print(f"Detected circle: Center: ({center_y:.2f}, {center_x:.2f}), Radius: {radius:.2f}")

    exit()

def process_image():
    image_path = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/exit/exit_cam_image.png'
    image = io.imread(image_path)

    # Detect the circle
    center_y, center_x, radius = detect_circle(image, fibre_diameter=100, cam_type='exit')
    print(f"Detected circle: Center: ({center_y:.2f}, {center_x:.2f}), Pixel Radius: {radius:.2f}")

    # Plot the original image with the detected circle
    plot_original_with_mask_unfilled(image, center_y, center_x, radius)

#image_to_fits(image_path)


"""
image_path = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/entrance/entrance_cam_image1.png'
image = io.imread(image_path)
print(com_of_spot(image, plot=True))
"""
entrance_folder = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/entrance'
exit_folder = 'E:/Important_Data/Education/Uni/Master/S4/Lab Stuff/SG_images/exit'
fiber_diameter = 100
"""
# Plot com movement for entrance images
plot_circle_movement(entrance_folder, fiber_diameter, 'entrance')

# Plot com movement for exit images
plot_circle_movement(exit_folder, fiber_diameter, 'exit')"""

sg = calculate_scrambling_gain(entrance_folder, exit_folder, fiber_diameter)
print(sg)
