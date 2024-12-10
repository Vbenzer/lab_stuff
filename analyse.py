import image_analysation
import image_reduction
import find_f_number
from astropy.io import fits
import os
import numpy as np

run_description = "Test"
dark_folder = "Darks"
light_folder = "Lights"
reduce_images_folder = "ReducedImages"
measurements_folder = "Measurements"

m_dark = image_reduction.create_master_dark(dark_folder)

for file_name in os.listdir(light_folder):
    if file_name.endswith(".fits"):  # Only process FITS files
        file_path = os.path.join(light_folder, file_name)
        with fits.open(file_path) as hdul:
            light_frame = hdul[0].data.astype(np.float32)  # Convert to float for precision
            pos = 1 # Todo: read from header the pos_value

        output_file = os.path.join(reduce_images_folder, file_name+"_reduced.fits")
        reduced_data = []
        red_file = image_reduction.reduce_image_with_dark(light_frame, m_dark, output_file)
        reduced_data.append((red_file, pos))


rad_pos_data = []

for red, pos in reduced_data:
    # Trim data to area of interest (perhaps not necessary with better background reduction)
    trimmed_data = image_analysation.CutImage(red, margin=500)  # Margin at 500 good for now

    # Locate center of mass within trimmed image (array)
    com = image_analysation.LocateFocus(trimmed_data)

    # Find aperture with 95% (or other) encircled energy
    radius = image_analysation.find_circle_radius(trimmed_data, com, ee_value=0.9, plot=True)

    rad_pos_data.append((radius, pos))

f_number,f_number_err = find_f_number.calculate_f_number(rad_pos_data, plot_regression=True)
print(f"Calculated F-number (f/#): {f_number:.2f} ± {f_number_err:.2f}")
