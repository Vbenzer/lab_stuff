import json
import os
from tkinter.constants import RAISED

import cv2
import numpy as np
from PIL import Image
from astropy.io import fits
from matplotlib import pyplot as plt
from skimage import measure, color, feature, transform
from skimage.filters import threshold_otsu


def fits_to_arr(filename: str) -> np.ndarray:
    """
    Convert fits file to numpy array
    Args:
        filename: Path of the fits file

    Returns:
        Data of fits file as numpy array

    """
    with fits.open(filename) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data


def locate_focus(array, **kwargs):
    """
    Locates the input focus point on the fiber cross-section by identifying the
    brightest spot in the image and performing circle fitting.

    Args:
        array (numpy.ndarray): Input image as a NumPy array.
        **kwargs:
            threshold (float, optional): Manually specified threshold for detection.

    Returns:
        tuple:
            location_y (int): Y-coordinate of the focus point's contour.
            location_x (int): X-coordinate of the focus point's contour.
            radius (float): Radius of the detected focus point.
    """


    from skimage.morphology import convex_hull_image, disk
    from skimage.filters import threshold_otsu
    from skimage.feature import canny
    from skimage.transform import hough_circle
    from skimage.filters import median
    from scipy.ndimage import center_of_mass

    threshold = kwargs.get("threshold", None)
    if threshold is None:
        threshold = threshold_otsu(array)
        #threshold = max(threshold, 50)     # Not sure if this is necessary
    binary = array > threshold

    filter = median(binary, disk(10))
    com = center_of_mass(binary)
    # temp1   = convex_hull_image(filter)
    # rmin = 1
    # rmax = 100
    # temp2   = hough_circle(canny(temp1),np.arange(rmin,rmax,1))
    # rid,center_y,center_x = np.unravel_index(np.argmax(temp2), temp2.shape)
    # radius = rmin+rid

    return com


def fit_circular(binary, **kwargs):
    """
    Fits the shape of a circular fiber using the Hough Transform.

    Args:
        binary (numpy.ndarray): Binary area of interest (AOI) array.
        **kwargs:
            radi_modification (float, optional): Value to adjust the fitted radius.
            full_out (bool, optional): Whether to return detailed output information.
            rmin (int, optional): Minimum radius to consider during fitting.
            rmax (int, optional): Maximum radius to consider during fitting.

    Returns:
        numpy.ndarray: Fitted circular binary array used as AOI.
    """

    from skimage.feature import canny
    from skimage.transform import hough_circle
    from skimage.draw import disk

    radi_modi = kwargs.get("radi_modification", 0)

    edges = canny(binary)
    rmin = kwargs.get("rmin", 10)
    rmax = kwargs.get("rmax", 500)
    hough = hough_circle(edges, np.arange(rmin, rmax, 1))
    rid, y, x = np.unravel_index(np.argmax(hough), hough.shape)
    radius = rmin + rid + radi_modi
    rr, cc = disk((y, x), radius)
    aoi = np.zeros_like(binary)
    rr = rr * (rr < aoi.shape[0])
    cc = cc * (cc < aoi.shape[1])
    aoi[rr, cc] = 1.

    full_out = kwargs.get("full_out", False)
    if full_out:
        return aoi, y, x, radius
    else:
        return aoi


def area_of_interest(array, **kwargs):
    """
    Finds the area of interest (AOI) in the input array using thresholding and a convex hull method.

    Args:
        array (numpy.ndarray): Input array to analyze.
        **kwargs:
            threshold (float, optional): Custom threshold value. If not provided, Otsu's method is used.
            filtering (bool, optional): Whether to apply median filtering before the convex hull. Defaults to True.

    Returns:
        numpy.ndarray: Boolean array indicating the area of interest.
    """

    from skimage.filters import threshold_otsu
    from skimage.filters import median
    from skimage.morphology import convex_hull_image, disk

    threshold = kwargs.get("threshold", None)
    if threshold is None:   threshold = threshold_otsu(array)
    binary = array > threshold
    if kwargs.get("filtering", True):
        filter = median(binary, disk(5))
    else:
        filter = binary
    aoi = convex_hull_image(filter)

    return aoi


def find_circle_radius(image_data, com: tuple[float] | None=None, ee_value:float=0.98, plot:bool=False, save_data:bool=True, save_file:str=None):
    """
    Finding the radius of circle around the center of mass of input image data
    Args:
        image_data: Numpy array of the input image
        com: center of mass of input image
        ee_value: encircled energy value to fit circle to. Float between 0 and 1.
        plot: If True, plot the circle on the image
        save_data: If True, save the radius data to a json file
        save_file: Path to save the radius data

    Returns:
        Radius of encircled energy of input image array
    """
    # Ensure image data is finite (Probably better not use this)
    #image_data = np.nan_to_num(image_data)

    # Read the center of mass

    center_x, center_y = com


    # Create a radial profile centered on the center of mass
    y, x = np.indices(image_data.shape)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)

    # Compute the radial profile
    radial_profile = np.bincount(r.ravel(), weights=image_data.ravel())
    radial_counts = np.bincount(r.ravel())
    radial_profile = radial_profile / radial_counts

    # Calculate encircled energy
    cumulative_energy = np.cumsum(radial_profile)
    total_energy = cumulative_energy[-1]
    encircled_energy_fraction = cumulative_energy / total_energy

    # Find the radius where xx% of the encircled energy is reached
    radius = int(np.argmax(encircled_energy_fraction >= ee_value))

    if plot or save_data:
        # Plot for visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(image_data, cmap='gray', origin='lower')
        plt.scatter(center_x, center_y, color='red', label='Center')
        circle = plt.Circle((center_x, center_y), radius, color='blue', fill=False, label=f'{ee_value*100}% Encircled Energy')
        plt.gca().add_artist(circle)
        plt.legend()
        plt.title(f"Circle Detection with {ee_value*100}% Encircled Energy")

        if save_data:
            if save_file is None:
                raise ValueError("'save_file' must be given")
            plt.savefig(save_file+"plot.png")

        if plot:
            pause = True
            if pause:
                plt.show(block=False)
                plt.pause(0.5)
            else:
                plt.show()
    plt.close()

    if save_data:
        with open(save_file+"radius.json", "w") as f:
            json.dump({"radius": radius, "center of mass": com}, f)

    return radius


def trimming(array, yrange, xrange, margin=0, square=False):
    """
    Finding the index range of the interested area to narrow down the image.

    Parameters:
    [binary]     input numpy.array to be trimmed
    [y_range]    y coordinate range
    [x_range]    x coordinate range
    [margin]     margin for the trimmed range

    Returns:
    [out_array]  Trimmed image
    """
    y_range = yrange.copy()
    x_range = xrange.copy()


    x_range[0] = x_range[0] - margin
    x_range[1] = x_range[1] + margin
    y_range[0] = y_range[0] - margin
    y_range[1] = y_range[1] + margin

    if x_range[0] < 0:  x_range[0] = 0
    if y_range[0] < 0:  y_range[0] = 0
    if x_range[1] > array.shape[1]: x_range[1] = array.shape[1]
    if y_range[1] > array.shape[0]: y_range[1] = array.shape[0]

    if square:
        ylength = y_range[1] - y_range[0]
        xlength = x_range[1] - x_range[0]
        if (ylength < xlength):
            x_range[0] += int((xlength - ylength) / 2.)
            x_range[1] -= int((xlength - ylength) / 2.) + int((xlength - ylength) % 2)
        elif (ylength > xlength):
            y_range[0] += int((ylength - xlength) / 2.)
            y_range[1] -= int((ylength - xlength) / 2.) + int((ylength - xlength) % 2)

    out_array = array[y_range[0]:y_range[1], x_range[0]:x_range[1]]

    return out_array


def cut_image(array, margin=20, **kwargs):
    """
    Finds the index range of the area of interest (AOI) and trims the image accordingly.

    Args:
        array (numpy.ndarray): Input array to be trimmed.
        margin (int, optional): Margin to include around the trimmed region. Defaults to 20.
        **kwargs:
            aoi (numpy.ndarray, optional): Manually provided boolean AOI array. If not given, it will be computed automatically.

    Returns:
        numpy.ndarray: Trimmed image array.
    """
    aoi = kwargs.get("aoi", None)
    if aoi is None: aoi = area_of_interest(array)
    y_range, x_range = narrow_index(aoi)
    out_array = trimming(array, y_range, x_range, margin)

    return out_array


def narrow_index(binary: np.ndarray[bool]) -> tuple[list[int], list[int]]:
    """
    Finds the index range of the area of interest to narrow down the image.

    Args:
        binary (numpy.ndarray): Boolean array representing the area of interest.

    Returns:
        tuple:
            y_range (list[int]): Start and end indices in the vertical (Y) direction.
            x_range (list[int]): Start and end indices in the horizontal (X) direction.
    """
    from skimage.feature import canny

    edges = canny(binary)
    perimeter = np.nonzero(edges)
    y_range = [min(perimeter[0]), max(perimeter[0])]
    x_range = [min(perimeter[1]), max(perimeter[1])]

    return y_range, x_range


def calculate_multiple_radii_ex(reduced_data: list[np.ndarray], measurements_folder: str) -> list[int]:
    """
    Calculate the radii of multiple reduced data files.
    Args:
        reduced_data: List of reduced data files
        measurements_folder: Path to measurements folder

    Returns:
        List of radii

    """
    radii = []
    for n, red in enumerate(reduced_data):
        # Trim data to area of interest (perhaps not necessary with better background reduction)
        trimmed_data = cut_image(red, margin=500)  # Margin at 500 good for now

        # Locate center of mass within trimmed image (array)
        com = locate_focus(trimmed_data)

        # Find aperture with encircled energy
        os.makedirs(measurements_folder + f"/Radius", exist_ok=True)
        radius = find_circle_radius(trimmed_data, com, ee_value=0.98, plot=False,
                                                      save_file=measurements_folder + f"/Radius/datapoint{n}")
        radii.append(radius)
    return radii

def calculate_multiple_radii(reduced_data: list[np.ndarray], measurements_folder: str, debug: bool = False) -> list[int]:
    radii = []
    for n, red in enumerate(reduced_data):
        # Trim data to area of interest (perhaps not necessary with better background reduction)
        trimmed_data = cut_image(red, margin=500)

        if debug:
            # Show image
            plt.imshow(trimmed_data, cmap='gray')
            plt.title("Trimmed Image")
            plt.show()

        """threshold = threshold_otsu(trimmed_data)
        thresholded_image = trimmed_data > 4000
        # Show thresholded image
        plt.imshow(thresholded_image, cmap='gray')
        plt.show()"""

        # Detect edges using Canny edge detector
        thresh = threshold_otsu(trimmed_data)
        thresh = thresh * 0.6
        thresholded_image = trimmed_data > thresh

        if debug:
            plt.imshow(thresholded_image, cmap='gray')
            plt.title("Thresholded Image")
            plt.show()

        """print("Threshold:", thresh / np.max(trimmed_data))
        quant_thresh = thresh / np.max(trimmed_data)
        quant_thresh_low = quant_thresh * 0.5
        edges = feature.canny(trimmed_data, sigma=1, low_threshold=quant_thresh_low, high_threshold=quant_thresh, use_quantiles=True)
"""
        """if debug:
            # Show edges image
            plt.figure(figsize=(trimmed_data.shape[1] / 100, trimmed_data.shape[0] / 100)) # Adjust figure size for details
            plt.title("Trimmed Image")
            plt.imshow(edges)
            plt.show()

        # Fill in within the edges
        from scipy import ndimage as ndi
        edges_filled = ndi.binary_fill_holes(edges)

        if debug:
            # Show filled image
            plt.imshow(edges_filled, cmap='gray')
            plt.title("Filled Edges")
            plt.show()"""

        # Label connected regions
        labeled_image, num = measure.label(thresholded_image, return_num=True)

        if debug:
            print("Number of regions labeled", num)

        if debug:
            # Plot the labeled image
            plt.imshow(labeled_image, cmap='gray')
            plt.title("Labeled Image")
            plt.show()


        # Measure properties of labeled regions
        properties = measure.regionprops(labeled_image)

        properties = [prop for prop in properties if prop.area >= 300]

        # Print the centroid
        if len(properties) == 1:
            prop = properties[0]
            if debug:
                print(f"Centroid: {prop.centroid}")
                print(f"Eccentricity: {prop.eccentricity}")
                print(f"Area: {prop.area}")
                print(f"Equivalent Diameter: {prop.equivalent_diameter}")
                print(f"Major Axis Length: {prop.major_axis_length}")
                print(f"Minor Axis Length: {prop.minor_axis_length}")

            # Get the center of mass
            coc = [prop.centroid[1], prop.centroid[0]]  # Swap x and y

            if prop.eccentricity > 0.2:
                raise Warning("Eccentricity is too high (>0.2), please check the image.")
        else:
            # Raise an error
            raise ValueError("Multiple regions found, please check the image.")


        # Find aperture with encircled energy
        os.makedirs(measurements_folder + f"/Radius", exist_ok=True)
        radius = find_circle_radius(trimmed_data, coc, ee_value=0.98, plot=False,
                                    save_file=measurements_folder + f"/Radius/datapoint{n}")
        radii.append(radius)

        with open(measurements_folder + f"/Radius/datapoint{n}" + "props.json", "w") as f:
            json.dump({"Centroid": prop.centroid, "Eccentricity": prop.eccentricity, "Area": prop.area,
                       "Equivalent Diameter": prop.equivalent_diameter,
                       "Major Axis Length": prop.major_axis_length, "Minor Axis Length": prop.minor_axis_length,
                       }, f)

    return radii

def measure_eccentricity(data:np.ndarray, plot: bool = False):
    """
    Measure the eccentricity of a fiber in an image.
    Args:
        data: Numpy array of the image data
        plot: Boolean to plot the image and the labeled regions

    Returns: Eccentricity of the fiber, if only one region is found.

    """
    from skimage.filters import threshold_otsu
    from skimage.feature import canny
    from skimage import io, measure, color

    # Trim data to area of interest (perhaps not necessary with better background reduction)
    trimmed_data = cut_image(data, margin=500)  # Margin at 500 good for now

    if plot:
        plt.figure()
        plt.imshow(trimmed_data, cmap='gray')
        plt.show()

    # Find otsu threshold
    threshold = threshold_otsu(trimmed_data)
    thresholded_image = trimmed_data > threshold

    if plot:
        plt.figure()
        plt.imshow(thresholded_image, cmap='gray')
        plt.show()

    # Label connected regions
    labeled_image = measure.label(thresholded_image)

    # Measure properties of labeled regions
    properties = measure.regionprops(labeled_image)

    properties = [prop for prop in properties if prop.area >= 300]

    #print(properties)

    if plot:
        # Plot the labeled image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(labeled_image, cmap='nipy_spectral')

        # Annotate labels on their centroid
        for prop in properties:
            y, x = prop.centroid  # Get centroid of the region
            ax.text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

        plt.title("Labeled Regions")
        plt.axis("off")
        plt.show()

    if len(properties) == 1:
        return properties[0].eccentricity
    else:
        print("Multiple regions found, returning None")
        for prop in properties:
            print(f'Label: {prop.label}, Eccentricity: {prop.eccentricity}')
        return None


def measure_fiber_dimensions(data:np.ndarray, px_to_mu=0.439453125, plot=False):
    """
    Measure the dimensions of a fiber in an image.
    Args:
        data: Numpy array of the image data
        px_to_mu: Pixels to microns conversion factor
        plot: Boolean to plot the image and the labeled regions

    Returns: The dimensions of the fiber in microns, if only one region is found.

    """
    # Step 1: Trim and threshold image
    trimmed_data = cut_image(data, margin=500)
    threshold = threshold_otsu(trimmed_data)
    binary = trimmed_data > threshold

    # Step 2: Label regions and filter by area
    labeled = measure.label(binary)
    props = [prop for prop in measure.regionprops(labeled) if prop.area >= 300]

    results = []

    # Step 3: Analyze each region
    for prop in props:
        coords = prop.coords

        # Switch x, y for cv2
        coords_yx = np.array(coords)
        coords_yx[:, 0], coords_yx[:, 1] = coords[:, 1], coords[:, 0]
        coords = coords_yx.astype(np.int32)

        rect = cv2.minAreaRect(coords)
        (cx, cy), (w, h), angle = rect

        width_mu = min(w, h) * px_to_mu
        height_mu = max(w, h) * px_to_mu
        diameter_mu = prop.equivalent_diameter * px_to_mu
        eccentricity = prop.eccentricity
        aspect_ratio = height_mu / width_mu if width_mu > 0 else 0

        # Step 4: Shape classification logic
        if eccentricity < 0.1:
            shape = 'circular'
            dimensions = {'diameter': diameter_mu}
        elif eccentricity < 0.5:
            shape = 'octagonal'
            dimensions = {'diameter': diameter_mu}
        else:
            shape = 'rectangular'
            dimensions = {'width': width_mu, 'height': height_mu}

        results.append({
            'label': prop.label,
            'shape': shape,
            'eccentricity': eccentricity,
            'aspect_ratio': aspect_ratio,
            'dimensions_mu': dimensions,
        })

        # Step 5: Optional plot
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(labeled, cmap='gray')

            # Draw the box around the region
            box = np.intp(cv2.boxPoints(rect))
            ax.plot(*zip(*np.append(box, [box[0]], axis=0)), 'k-', lw=2)

            # Mark the label at the center
            ax.text(cx, cy, f"{prop.label}", color='red', ha='center', va='center', fontsize=12)

            # Determine which is major/minor based on width and height
            major_len = max(w, h) / 2
            minor_len = min(w, h) / 2

            # OpenCV angle logic fix
            angle_rad = np.deg2rad(angle)
            if w < h:
                angle_rad += np.pi / 2

            # Direction vectors
            major_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            minor_vec = np.array([-major_vec[1], major_vec[0]])

            center = np.array([cx, cy])
            major_end = center + major_vec * major_len
            minor_end = center + minor_vec * minor_len

            ax.plot([center[0], major_end[0]], [center[1], major_end[1]], 'r-', lw=2.5, label='Major Axis')
            ax.plot([center[0], minor_end[0]], [center[1], minor_end[1]], 'b-', lw=2.5, label='Minor Axis')

            ax.legend()
            ax.set_title(f"Label {prop.label}: {shape.capitalize()} Fiber")
            ax.axis("off")
            plt.show()

    return results if results else None


def create_master_dark(dark_folder:str, img_path:str=None, plot:bool=False, save:bool=False) -> np.ndarray:
    """
    Create a master dark frame by averaging all dark frames in the folder.

    Parameters:
        dark_folder (str): Path to the folder containing dark frames.
        img_path (str): Path to save the plot.
        plot (bool, optional): If True, plot the master dark frame.
        save (bool, optional): If True, save the master dark frame as a FITS file.

    Returns:
        np.ndarray: The master dark frame.
    """
    dark_data = []  # Initialize list to store all dark frame data

    # Iterate over all FITS files in the folder
    for file_name in os.listdir(dark_folder):
        if file_name.endswith(".fits"):  # Only process FITS files
            file_path = os.path.join(dark_folder, file_name)
            with fits.open(file_path) as hdul:
                dark_frame = hdul[0].data.astype(np.float32)  # Convert to float for precision
                dark_data.append(dark_frame)  # Append to the list
        if file_name.endswith(".png"):
            file_path = os.path.join(dark_folder, file_name)
            dark_frame = plt.imread(file_path)
            dark_data.append(dark_frame)

    # Calculate the master dark frame as the mean of all dark frames
    master_dark = np.mean(dark_data, axis=0)
    if save or plot:
        plt.figure()
        plt.imshow(master_dark, cmap='gray', origin='lower')
        if save:
            if img_path is None:
                raise ValueError("'img_path' must be provided if 'save' is True")
            plt.savefig(img_path)
        if plot:
            plt.show()


    return master_dark


def reduce_image_with_dark(science_data:np.ndarray, dark_data:np.ndarray, output_file:str, save:bool=False,
                           plot:bool=False, save_plot:bool=False, img_path:str=None) -> np.ndarray:
    """
    Reduces a science image by subtracting a dark frame.

    Parameters:
        science_data (np.ndarray): Science image data.
        dark_data (np.ndarray): Dark image data.
        output_file (str): Path to save the reduced FITS file.
        save (bool): Save reduce image to file?
        plot (bool, optional): If True, plot the reduced image.
        save_plot: Save plot as image if True.
        img_path: Required if save_plot is set to True. Path to save plot to.

    Returns:
          np.ndarray: The reduced image data.
    """

    # Ensure data are of the same shape
    if science_data.shape != dark_data.shape:
        raise ValueError("Science image and dark frame must have the same dimensions.")

    # Subtract the dark frame from the science image
    reduced_data = science_data - dark_data

    # Clip negative values to zero (or other minimum threshold, if applicable)
    #reduced_data = np.clip(reduced_data, 0, None) #TODO: Why was this here?

    if save:
        # Check if the output file is .fits or .png
        if output_file.endswith(".fits"):
            # Save the reduced image to a new FITS file
            hdu = fits.PrimaryHDU(data=reduced_data)
            hdu.writeto(output_file, overwrite=True)
            print(f"Reduced image saved to: {output_file}")

        elif output_file.endswith(".png"):
            # Save the reduced image to a new PNG file
            cv2.imwrite(output_file, reduced_data)
            print(f"Reduced image saved to: {output_file}")

        else:
            raise ValueError("Output file must be a .fits or .png file.")

    if plot or save_plot:
        # Plotting
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Science Image")
        plt.imshow(science_data, cmap='gray', origin='lower')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Dark Frame")
        plt.imshow(dark_data, cmap='gray', origin='lower')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Reduced Image")
        plt.imshow(reduced_data, cmap='gray', origin='lower')
        plt.colorbar()

        plt.tight_layout()
        if plot:
            plt.show()
        if save_plot:
            if img_path is None:
                raise ValueError("'img_path' must be provided if 'save' is True")
            plt.savefig(img_path)
        plt.close()

    return reduced_data


def plot_images(science_file, dark_file, reduced_file):
    """
    Plots the science, reduced and dark image.
    Args:
        science_file: Path of the science image FITS file.
        dark_file: Path of the dark frame FITS file.
        reduced_file: Path of the reduced FITS file.

    Returns: Plots

    """
    #Load images
    with fits.open(science_file) as hdul:
        science_data = hdul[0].data.astype(np.float32)

    with fits.open(dark_file) as hdul:
        dark_data = hdul[0].data.astype(np.float32)

    with fits.open(reduced_file) as hdul:
        reduced_data = hdul[0].data.astype(np.float32)

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Science Image")
    plt.imshow(science_data, cmap='gray', origin='lower')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Dark Frame")
    plt.imshow(dark_data, cmap='gray', origin='lower')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Reduced Image")
    plt.imshow(reduced_data, cmap='gray', origin='lower')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


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


def com_of_spot(image:np.ndarray, threshold=None, plot:bool=False):
    """
    Calculate the center of mass of a spot in the given image.
    Args:
        image: The input image as a NumPy array.
        threshold: Threshold for the spot detection.
        plot: If True, plot the image with the center of mass marked.

    Returns:
        com (tuple): The (y, x) coordinates of the center of mass.
    """
    # Get Area of Interest
    aoi = area_of_interest(image, threshold=threshold)

    # Get the range of the AOI
    y_range, x_range = narrow_index(aoi)

    # Cut the image to the AOI
    cut_image_data = cut_image(image, aoi=aoi, margin=10)

    # Get the center of mass of the cut image
    com = locate_focus(cut_image_data)

    dim = cut_image_data.shape
    size = [dim[1] / 100, dim[0] / 100]


    if plot:
        # noinspection PyTypeChecker
        plt.figure(figsize=size)
        plt.imshow(cut_image_data, cmap='gray', origin='lower')
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


def analyse_f_number(image:np.ndarray, measurements_folder:str):
    """
    Analyse the f-number of the image and save it to a json
    Args:
        image: Image to analyse
        measurements_folder: Path to the folder where the measurements are saved

    Returns:

    """
    # Calculate radius
    trimmed_data = core.data_processing.cut_image(image, margin=500)  # Margin at 500 good for now

    # Locate center of mass within trimmed image (array)
    com = core.data_processing.locate_focus(trimmed_data)

    # Swap x and y
    com = (com[1], com[0])

    print("Calculating radius...")

    # Find aperture with encircled energy
    os.makedirs(measurements_folder, exist_ok=True)
    radius = core.data_processing.find_circle_radius(trimmed_data, com, ee_value=0.98, plot=True, save_data=False)

    print(f"Radius: {radius}")

    # Add the radius to existing json file
    # Path to the JSON file
    json_file_path = measurements_folder + "radii.json"

    # Check if the file exists
    if os.path.exists(json_file_path):
        # Load existing data
        with open(json_file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        # Initialize an empty list if the file does not exist
        existing_data = {"radius": []}

    # Append new data
    radii = existing_data["radius"]

    # Delete old measurements if too many
    if len(radii) > 200:
        radii = radii[-200:]

    # Don't append if radius deviates too much from the mean
    if len(radii) > 0:
        mean_radius = np.mean(radii)
        if abs(radius - mean_radius) > 200:
            print("Radius deviates too much from mean. Not appending.")
        else:
            radii.append(radius)
    else:
        radii.append(radius)

    new_data = {"radius": radii}

    # Write the updated data back to the file
    with open(json_file_path, 'w') as file:
        json.dump(new_data, file, indent=4)

    # Plot radius vs number of measurements
    plt.plot(radii)
    plt.xlabel("Number of measurements")
    plt.ylabel("Radius")
    plt.title("Radius vs Number of Measurements")
    plt.savefig(measurements_folder + "radius_vs_measurements.png")
    plt.close()


def detect_circle(image:np.ndarray, fiber_px_radius:int, fiber_px_radius_max:int=None) -> tuple[int, int, int]:
    """
    Detects a circle in the given image using Hough Circle Transform.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        fiber_px_radius (int): The radius of the fiber in pixels.
        fiber_px_radius_max (int, optional): The maximum radius of the fiber in pixels. If set fiber_px_radius will
        be used as min.

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

    plt.imshow(edges)
    plt.show()

    if fiber_px_radius_max is not None:
        # Define the range of radii to search for
        hough_radii = np.arange(fiber_px_radius, fiber_px_radius_max, 10)
        print("hough_radii", hough_radii)
    else:
        hough_radii = np.arange(fiber_px_radius - 5, fiber_px_radius + 5, 1)

    # Perform Hough Circle Transform
    hough_res = transform.hough_circle(edges, hough_radii)

    # Select the most prominent circle
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1, normalize=False)

    return cy[0], cx[0], radii[0]


if __name__ == "__main__":
    # Example usage
    filter = "3.5"
    main_folder = rf"D:\Vincent\O_50_0000_0000\FRD\filter_{filter}\REDUCED"
    reduced_data_1 = main_folder + rf"\filter_{filter}_pos_0_light_reduced.fits"
    reduced_data_2 = main_folder + rf"\filter_{filter}_pos_5_light_reduced.fits"
    reduced_data_3 = main_folder + rf"\filter_{filter}_pos_9.9_light_reduced.fits"
    data1 = fits_to_arr(reduced_data_1)
    data2 = fits_to_arr(reduced_data_2)
    data3 = fits_to_arr(reduced_data_3)
    data = [data1, data2, data3]
    measurements_folder = r"D:\Vincent\test"
    print(calculate_multiple_radii(data, measurements_folder, debug=True))