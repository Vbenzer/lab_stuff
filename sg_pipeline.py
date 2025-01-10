import numpy as np
from PIL import Image
from astropy.io import fits
import os
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

def detect_circle(image, fibre_diameter, cam_type, threshold_value=20):
    """
    Detects a circle in the given image using Hough Circle Transform.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
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

    plt.imshow(edges)
    plt.show()

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

def create_circular_mask(image, center, radius, margin=0):
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

def plot_histogramm(image):
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


def image_to_fits(image_path):
    """
    Converts an image array to a FITS file.

    Parameters:
        image_array (np.ndarray): The input image as a NumPy array.
        fits_path (str): The path where the FITS file will be saved.
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

def plot_original_with_mask_unfilled(image, center_y, center_x, radius):
    """
    Plots the original image with an unfilled mask overlay.

    Parameters:
        image (np.ndarray): The original image as a NumPy array.
        mask (np.ndarray): The mask as a NumPy array.
    """
    overlay_image = image.copy()
    rr, cc = circle_perimeter(center_y, center_x, radius)
    overlay_image[rr, cc] = 255

    plt.imshow(overlay_image, cmap='gray')
    plt.imshow(overlay_image, cmap='gray')
    plt.title('Original Image with Unfilled Mask Overlay')
    plt.axis('off')
    plt.show()


def com_of_spot(image_array, plot=False):
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

# Test functions
def process_these_images(): # Just for testing purposes
    image_folder = 'E:/Important_Data/Lehranstalten/Uni/Master/S4/Lab Stuff/SG_images/exit'
    fibre_diameter = 100
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = io.imread(image_path)
        center_y, center_x, radius = detect_circle(image, fibre_diameter, cam_type='exit')
        plot_original_with_mask_unfilled(image, center_y, center_x, radius)

        print(f"Detected circle: Center: ({center_y:.2f}, {center_x:.2f}), Radius: {radius:.2f}")

    # Same procedure for entrance images
    image_folder = 'E:/Important_Data/Lehranstalten/Uni/Master/S4/Lab Stuff/SG_images/entrance'

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = io.imread(image_path)
        center_y, center_x, radius = detect_circle(image, fibre_diameter, cam_type='entrance')
        plot_original_with_mask_unfilled(image, center_y, center_x, radius)

        print(f"Detected circle: Center: ({center_y:.2f}, {center_x:.2f}), Radius: {radius:.2f}")

    exit()

def process_image():
    image_path = 'E:/Important_Data/Lehranstalten/Uni/Master/S4/Lab Stuff/SG_images/exit/exit_cam_image.png'
    image = io.imread(image_path)

    # Detect the circle
    center_y, center_x, radius = detect_circle(image, fibre_diameter=100, cam_type='exit')
    print(f"Detected circle: Center: ({center_y:.2f}, {center_x:.2f}), Pixel Radius: {radius:.2f}")

    # Plot the original image with the detected circle
    plot_original_with_mask_unfilled(image, center_y, center_x, radius)

#image_to_fits(image_path)



image_path = 'E:/Important_Data/Lehranstalten/Uni/Master/S4/Lab Stuff/SG_images/entrance/entrance_cam_image1.png'
image = io.imread(image_path)
print(com_of_spot(image, plot=True))
