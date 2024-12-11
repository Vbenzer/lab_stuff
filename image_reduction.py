import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os


def create_master_dark(dark_folder, plot=False):
    """
    Create a master dark frame by averaging all dark frames in the folder.

    Parameters:
        dark_folder (str): Path to the folder containing dark frames.
        plot (bool, optional): If True, plot the master dark frame.

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

    # Calculate the master dark frame as the mean of all dark frames
    master_dark = np.mean(dark_data, axis=0)

    if plot:
        plt.figure()
        plt.imshow(master_dark, cmap='gray', origin='lower')
        plt.show()

    return master_dark

def reduce_image_with_dark(science_data, dark_data, output_file, save=False, plot=False):
    """
    Reduces a science image by subtracting a dark frame.

    Parameters:
        science_data (np.ndarray): Science image data.
        dark_data (np.ndarray): Dark image data.
        output_file (str): Path to save the reduced FITS file.
        save (bool): Save reduce image to file?
        plot (bool, optional): If True, plot the reduced image.

    Returns:
          np.ndarray: The reduced image data.
    """

    # Ensure data are of the same shape
    if science_data.shape != dark_data.shape:
        raise ValueError("Science image and dark frame must have the same dimensions.")

    # Subtract the dark frame from the science image
    reduced_data = science_data - dark_data

    # Clip negative values to zero (or other minimum threshold, if applicable)
    reduced_data = np.clip(reduced_data, 0, None)

    if save:
        # Save the reduced image to a new FITS file
        hdu = fits.PrimaryHDU(data=reduced_data)
        hdu.writeto(output_file, overwrite=True)
        print(f"Reduced image saved to: {output_file}")

    if plot:
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


if __name__ == "__main__":

    """# Usage example
    science_file = 'test_short.fits.fits'
    dark_file = 'DARK_0.02s_2024-12-04_14-25-11.fits'
    output_file = 'reduced_image_close.fits'
    reduce_image_with_dark(science_file, dark_file, output_file)

    plot_images(science_file, dark_file, output_file)"""

    create_master_dark("Darks", plot=True)
