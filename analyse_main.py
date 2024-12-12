import image_analysation
import image_reduction
import find_f_number
from astropy.io import fits
import os
import numpy as np

project_folder = "D:/stepper_motor/test_images/sequence_stepper_filter_fstop_analysis/Filter2/"
run_description = "Test"
dark_folder = project_folder+"DARK"
light_folder = project_folder+"LIGHT"
reduce_images_folder = project_folder+"REDUCED/"
measurements_folder = project_folder+"Measurements/"
pos_values=[0,5,9.9]

m_dark = image_reduction.create_master_dark(dark_folder, plot=False)

reduced_data = []
for file_name in os.listdir(light_folder):
    if file_name.endswith(".fits"):  # Only process FITS files
        file_path = os.path.join(light_folder, file_name)
        with fits.open(file_path) as hdul:
            light_frame = hdul[0].data.astype(np.float32)  # Convert to float for precision

        os.makedirs(reduce_images_folder, exist_ok=True)
        output_file = os.path.join(reduce_images_folder, file_name+"_reduced.fits")
        red_file = image_reduction.reduce_image_with_dark(light_frame, m_dark, output_file, save=True)
        reduced_data.append(red_file)






radii = []


for n,red in enumerate(reduced_data):
    # Trim data to area of interest (perhaps not necessary with better background reduction)
    trimmed_data = image_analysation.CutImage(red, margin=500)  # Margin at 500 good for now

    # Locate center of mass within trimmed image (array)
    com = image_analysation.LocateFocus(trimmed_data)

    # Find aperture with 95% (or other) encircled energy
    radius = image_analysation.find_circle_radius(trimmed_data, com, ee_value=0.98, plot=True, save_file=measurements_folder+f"Radius_{n}")
    radii.append(radius)

radii.sort()
pos_values.sort()

print('radii:',radii,'pos:',pos_values)

f_number,f_number_err = find_f_number.calculate_f_number(radii,pos_values, plot_regression=True)
print(f"Calculated F-number (f/#): {f_number:.2f} ± {f_number_err:.2f}")
