import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import cv2


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


if __name__ == "__main__":



    create_master_dark("Darks", plot=True)
