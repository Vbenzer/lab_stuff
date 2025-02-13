import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import json

def fits_to_arr(filename: str) -> np.ndarray:
    """
    Convert fits file to numpy array
    Args:
        filename: Path of the fits file

    Returns:
        Data of fits file as numpy array

    """
    with fits.open(filename) as hdul:
        data = hdul[0].data
    return data

def LocateFocus(array, **kwargs):
    '''
    Locating the input focus point on the fiber cross-section.
    Methodology is to locate the brightest spot in the image and
    perform a circle fitting.

    Parameters:
    [array]     input numpy.array for the image\n
    kwargs:\n
    [threshold] manual input of the threshold

    Returns:
    [location_y]    y coordinate for the contour of the focus point\n
    [location_x]    x coordinate for the contour of the focus point\n
    [radius]        radius of the focus point
    '''
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


def FitCircular(binary, **kwargs):
    '''
    Fitting the shape of circular fiber with hough transform.

    Parameters:
    -----------
    [binary]                 binary aoi array \n
    [radi_modification]      modify radii for the fitted circular area\n
    [full_out]\t             whether to output the detailed information\n
    [rmin,rmax]\t            min and max of fitted radius

    Returns:
    -----------
    [fitted]    fitted circular binary array used as aoi
    '''
    from skimage.feature import canny
    from skimage.transform import hough_circle
    from skimage.draw import disk

    radi_modi = kwargs.get("radi_modification", 0)

    edges = canny(binary)
    rmin = kwargs.get("rmin", 10);
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

def Area_of_Interest(array, **kwargs):
    '''
    Finding the area of interest for input array with threshold and convex hull method

    Parameters:
    [array]     input numpy.array

    Returns:
    [aoi]       boolean array indicating the area of interest
    '''
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

def find_circle_radius(image_data, com: tuple[float] | None=None, ee_value:float=0.95, plot:bool=False, save_data:bool=True, save_file:str=None):
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
    if com:
        center_x, center_y = com
    else:
        center_y, center_x = int(com[0]), int(com[1]) #Todo: potentially change to float, pixel grid to center of pixel

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

    # Find the radius where 95% (or other if changed) of the encircled energy is reached
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
    Finding the index range of the area of interest to narrow down the image.

    Parameters:
    ------------
    [array]         input numpy.array to be trimmed\n
    [margin]        margin for the range to be trimmed\n
    [aoi (opt)]     manually given area of interest as boolean array\n

    Returns:
    ------------
    [out_array]  Trimmed image
    """
    aoi = kwargs.get("aoi", None)
    if aoi is None: aoi = Area_of_Interest(array)
    y_range, x_range = narrow_index(aoi)
    out_array = trimming(array, y_range, x_range, margin)

    return out_array


def narrow_index(binary: np.ndarray[bool]) -> tuple[list[int], list[int]]:
    """
    Finding the index range of the interested area to narrow down the image.

    Parameters:
    [binary]     input numpy.array for the interested area (binary needed) (boolean array)

    Returns:
    [y_range]    y coordinate range
    [x_range]    x coordinate range
    """
    from skimage.feature import canny

    edges = canny(binary)
    perimeter = np.nonzero(edges)
    y_range = [min(perimeter[0]), max(perimeter[0])]
    x_range = [min(perimeter[1]), max(perimeter[1])]

    return y_range, x_range


def calculate_multiple_radii(reduced_data: list[np.ndarray], measurements_folder: str) -> list[int]:
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
        com = LocateFocus(trimmed_data)

        # Find aperture with encircled energy
        os.makedirs(measurements_folder + f"/Radius", exist_ok=True)
        radius = find_circle_radius(trimmed_data, com, ee_value=0.98, plot=False,
                                                      save_file=measurements_folder + f"/Radius/datapoint{n}")
        radii.append(radius)
    return radii


# Usage
if __name__ == "__main__":
    #Path of fits file to analyse
    fits_file = 'D:/stepper_motor/test_images/sequence_stepper_filter_fstop_analysis/Filter2/REDUCED/LIGHT_0000_0.00s.fits_reduced.fits'

    #Turn data to numpy array
    data = fits_to_arr(fits_file)

    #Trim data to area of interest (perhaps not necessary with better background reduction)
    trimmed_data = cut_image(data, margin=500) #Margin at 500 good for now

    #Locate center of mass within trimmed image (array)
    com = LocateFocus(trimmed_data) #Todo: Does com makes sense with trimmed data?

    #Find aperture with 95% (or other) encircled energy
    radius = find_circle_radius(trimmed_data, com, ee_value=0.9,plot=True)

    print(f"Radius at encircled energy: {radius}")
