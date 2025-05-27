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


def pipeline(image, cut_image_par:bool=False, debug:bool=False):
    from core.data_processing import png_to_numpy
    from matplotlib import pyplot as plt
    from core.data_processing import cut_image

    if isinstance(image, str):
        image_path = image
        if image_path.endswith('.png'):
            # Load the png image and convert it to a numpy array
            image = png_to_numpy(image_path)
        elif image_path.endswith('.gif'):
            from PIL import Image
            image = Image.open(image_path)
            image = np.array(image)  # Convert to grayscale
        else:
            raise ValueError("Unsupported image format. Please provide a .png or .gif file.")

    if cut_image_par:
        cut_image = cut_image(image)
    else:
        cut_image = image

    # Show cut image
    plt.imshow(cut_image, cmap='gray')
    plt.title("Cut Image")
    plt.colorbar()
    plt.show()

    transformed_image = fast_fourier_transform(cut_image)

    plt.imshow(np.log1p(np.abs(transformed_image)), cmap='gray')
    plt.title("Fourier Transform Magnitude")
    plt.colorbar()
    plt.show()

    power_spectrum = np.abs(transformed_image) ** 2

    # Use the pixel size to calculate frequency axes
    pixel_size = 5.2e-6  # Pixel size in meters
    frequency_x = np.fft.fftfreq(cut_image.shape[1], d=pixel_size)
    frequency_y = np.fft.fftfreq(cut_image.shape[0], d=pixel_size)
    frequency_x = np.fft.fftshift(frequency_x)
    frequency_y = np.fft.fftshift(frequency_y)

    if debug:
        print("Frequency X:", frequency_x)
        print("Frequency Y:", frequency_y)

    # Define the target values: 0 and increments of ±10000
    targets = np.arange(0, round(max(abs(frequency_x)), -5), 10000)  # Positive direction
    targets = np.concatenate((-targets[::-1], targets[1:]))  # Add negative direction

    # Find the indices of the closest values
    indices_x = [np.argmin(np.abs(frequency_x - target)) for target in targets]
    indices_y = [np.argmin(np.abs(frequency_y - target)) for target in targets]

    # Print the results
    if debug:
        for target, index in zip(targets, indices_x):
            print(f"Target: {target}, Closest Value: {frequency_x[index]}, Index: {index}")

        for target, index in zip(targets, indices_y):
            print(f"Target: {target}, Closest Value: {frequency_y[index]}, Index: {index}")

    # Plot the power spectrum with frequency axes
    plt.figure(figsize=(8, 8))
    plt.imshow(np.log1p(power_spectrum), cmap='jet')
    plt.title("Power Spectrum")
    plt.colorbar()
    plt.grid(True)
    plt.xlabel("Frequency X (1/mm)")
    plt.ylabel("Frequency Y (1/mm)")

    plt.xticks(ticks=[i for i in indices_x], labels=[int(targets[i]/1000) for i in range(len(targets))], rotation=45)
    plt.yticks(ticks=[i for i in indices_y], labels=[int(targets[i]/1000) for i in range(len(targets))])
    plt.show()

    # Radial average of the power spectrum
    u, v = np.meshgrid(frequency_x, frequency_y)
    r = np.sqrt(u**2 + v**2)

    r_flat = r.ravel()
    P_flat = np.abs(transformed_image) ** 2
    P_flat = P_flat.ravel()

    # choose your radial bins
    r_max = r_flat.max()
    nbins = min(transformed_image.shape)
    bin_edges = np.linspace(0, r_max, nbins + 1)

    # histogram: sum of power in each bin
    sum_P, _ = np.histogram(r_flat, bins=bin_edges, weights=P_flat)

    # histogram: count of pixels in each bin
    counts, _ = np.histogram(r_flat, bins=bin_edges)

    # avoid division by zero
    radial_prof = sum_P / np.maximum(counts, 1)

    # bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])




    print("Maximum frequency:", min(np.max(frequency_x), np.max(frequency_y)))
    mask = (bin_centers <= min(np.max(frequency_x), np.max(frequency_y)))




    plt.plot(bin_centers[mask], radial_prof[mask])
    plt.xlabel("Spatial frequency (cycles/µm)")
    plt.ylabel("Radial‐averaged power")
    plt.yscale('log')
    plt.title("1D Power Spectrum")
    plt.show()

    return power_spectrum



if __name__ == "__main__":
    image = r"D:\Vincent\C_100_0000_0003\SG1\exit\reduced\exit_cam_image000_reduced.png"

    power_spectrum = pipeline(image, cut_image_par=True, debug=True)

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


