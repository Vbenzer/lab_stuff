"""Utilities to analyse modal noise via Fourier transforms.

This module provides helper functions to load images, compute their
Fourier transform and visualise the resulting power spectra.  It is
mainly intended for quick explorations of fiber images.
"""
import numpy as np


def fast_fourier_transform(image):
    """
    Perform a fast Fourier transform on the input image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Fourier transformed image.
    """
    image_transformed = np.fft.fft2(image)
    return np.fft.fftshift(image_transformed)

def create_test_image(plot:bool=False):
    """
    Create a test image with the following properties:
    - Size: 100x100 pixels
    - Black background 2px at each edge
    - Vertical lines every 4 pixels, 2px wide
    Returns:
        numpy.ndarray: Test image with vertical lines.
    """
    # Create a 100x100 black image
    image = np.zeros((100, 100), dtype=np.uint8)

    # Add a 2px black border (already black by default)
    # Draw vertical lines every 4 pixels, 2px wide, within the border
    for x in range(3, 97, 4):
        image[3:97, x:x + 2] = 255  # Set the vertical lines to white


    if plot:
        from matplotlib import pyplot as plt
        # Figure size fitting the image
        size = (image.shape[1] / 10, image.shape[0] / 10)
        plt.figure(figsize=size)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        plt.title("Test Image with Vertical Lines")
        plt.colorbar()
        plt.show()

        # Print image values for debugging
        print("Test Image Values:", np.std(image), np.mean(image))

    return image


def _load_image(image: str | np.ndarray, save: bool) -> tuple[np.ndarray, str | None]:
    """Load ``image`` from disk if necessary.

    Parameters
    ----------
    image:
        Either a path to an image or an array.
    save:
        If ``True`` plots will be written next to the image and a base
        directory is returned.

    Returns
    -------
    numpy.ndarray
        The loaded image as array.
    str | None
        The directory where plots should be stored or ``None`` when
        ``save`` is ``False``.
    """

    from core.data_processing import png_to_numpy
    import os

    basepath = None
    if isinstance(image, str):
        if save:
            basepath = os.path.abspath(os.path.join(os.path.dirname(image), "..", "..")) + "/results"
            os.makedirs(basepath, exist_ok=True)

        if image.endswith(".png"):
            image = png_to_numpy(image)
        elif image.endswith(".gif"):
            from PIL import Image
            image = np.array(Image.open(image))
        else:
            raise ValueError("Unsupported image format. Please provide a .png or .gif file.")
    else:
        if save:
            raise ValueError("If 'image' is not a string, 'save' must be False.")

    return image, basepath


def _plot_image(img: np.ndarray, title: str, save_path: str | None = None) -> None:
    """Display ``img`` and optionally write it to ``save_path``."""
    from matplotlib import pyplot as plt

    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def _radial_average(power_spectrum: np.ndarray, freq_x: np.ndarray, freq_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial average of ``power_spectrum``."""
    u, v = np.meshgrid(freq_x, freq_y)
    r = np.sqrt(u ** 2 + v ** 2)
    r_flat = r.ravel()
    p_flat = power_spectrum.ravel()
    r_max = r_flat.max()
    nbins = min(power_spectrum.shape)
    bin_edges = np.linspace(0, r_max, nbins + 1)
    sum_p, _ = np.histogram(r_flat, bins=bin_edges, weights=p_flat)
    counts, _ = np.histogram(r_flat, bins=bin_edges)
    radial_prof = sum_p / np.maximum(counts, 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, radial_prof


def pipeline(image, cut_image_par: bool = False, save: bool = False, debug: bool = False) -> np.ndarray:
    """Run the modal noise analysis pipeline on ``image``.

    Parameters
    ----------
    image:
        Path to an image or an image array.
    cut_image_par:
        If ``True`` the image is cropped using :func:`core.data_processing.cut_image`.
    save:
        Save the generated plots next to the input image.
    debug:
        Print additional debug output.

    Returns
    -------
    numpy.ndarray
        The computed power spectrum.
    """

    from core.data_processing import cut_image
    from matplotlib import pyplot as plt
    import os

    img_array, basepath = _load_image(image, save)

    if cut_image_par:
        img_array = cut_image(img_array)

    _plot_image(img_array, "Cut Image", os.path.join(basepath, "cut_image.png") if save else None)

    transformed = fast_fourier_transform(img_array)
    _plot_image(np.log1p(np.abs(transformed)), "Fourier Transform Magnitude",
                os.path.join(basepath, "fourier_trans_2d.png") if save else None)

    power_spectrum = np.abs(transformed) ** 2

    pixel_size = 5.2e-6  # pixel size in meters
    frequency_x = np.fft.fftshift(np.fft.fftfreq(img_array.shape[1], d=pixel_size))
    frequency_y = np.fft.fftshift(np.fft.fftfreq(img_array.shape[0], d=pixel_size))

    if debug:
        print("Frequency X:", frequency_x)
        print("Frequency Y:", frequency_y)

    targets = np.arange(0, round(max(abs(frequency_x)), -5), 10000)
    targets = np.concatenate((-targets[::-1], targets[1:]))
    indices_x = [np.argmin(np.abs(frequency_x - t)) for t in targets]
    indices_y = [np.argmin(np.abs(frequency_y - t)) for t in targets]

    plt.figure(figsize=(8, 8))
    plt.imshow(np.log1p(power_spectrum), cmap="jet")
    plt.title("Power Spectrum")
    plt.colorbar()
    plt.grid(True)
    plt.xlabel("Frequency X (1/mm)")
    plt.ylabel("Frequency Y (1/mm)")
    plt.xticks(ticks=indices_x, labels=[int(t/1000) for t in targets], rotation=45)
    plt.yticks(ticks=indices_y, labels=[int(t/1000) for t in targets])
    if save:
        plt.savefig(os.path.join(basepath, "fourier_trans_2d_physical_units.png"))
        plt.close()
    else:
        plt.show()

    bin_centers, radial_prof = _radial_average(power_spectrum, frequency_x, frequency_y)
    mask = bin_centers <= min(frequency_x.max(), frequency_y.max())

    plt.plot(bin_centers[mask], radial_prof[mask])
    plt.xlabel("Spatial frequency (cycles/µm)")
    plt.ylabel("Radial-averaged power")
    plt.yscale("log")
    plt.title("1D Power Spectrum")
    if save:
        plt.savefig(os.path.join(basepath, "fourier_plot.png"))
        plt.close()
    else:
        plt.show()

    return power_spectrum





if __name__ == "__main__":
    #image = r"D:\Vincent\C_100_0000_0003\SG1\exit\reduced\exit_cam_image000_reduced.png"
    image = r"/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/C_100_0000_0003/SG3/exit/reduced/exit_cam_image000_reduced.png"

    power_spectrum = pipeline(image, cut_image_par=True, debug=False, save=True)

    # Plot the power spectrum
    #from matplotlib import pyplot as plt
    #plt.imshow(np.log1p(power_spectrum), cmap='gray')
    #plt.title("Power Spectrum")
    #plt.colorbar()
    #plt.show()

    #image_path = r"D:\Vincent\C_100_0000_0003\test\stp2.gif"

    #image = create_test_image(plot=False)

    #image = np.abs(pipeline(image, cut_image_par=True))

    # Convert the test image to FITS format
    #from core.data_processing import image_to_fits
    #fits_image = image_to_fits(image, image_path=r"D:\Vincent\C_100_0000_0003\test\test1.fits")


