"""Module collimation_test_obs.py.

Auto-generated docstring for better readability.
"""
import core.data_processing
import image_analysation_obs
import image_reduction_obs
import find_f_number_obs
from astropy.io import fits
import os
import numpy as np
import subprocess
import json


def main(project_folder:str): # Obsolete


    dark_folder = project_folder+"/DARK"
    light_folder = project_folder+"/LIGHT"
    reduce_images_folder = project_folder+"/REDUCED/"
    measurements_folder = project_folder+"/Measurements/"
    os.makedirs(measurements_folder, exist_ok=True)



    m_dark = core.data_processing.create_master_dark(dark_folder, plot=False)

    reduced_data = []
    for file_name in os.listdir(light_folder):
        if file_name.endswith(".fits"):  # Only process FITS files
            file_path = os.path.join(light_folder, file_name)
            with fits.open(file_path) as hdul:
                light_frame = hdul[0].data.astype(np.float32)  # Convert to float for precision

            os.makedirs(reduce_images_folder, exist_ok=True)
            output_file = os.path.join(reduce_images_folder, file_name+"_reduced.fits")
            red_file = core.data_processing.reduce_image_with_dark(light_frame, m_dark, output_file, save=True)
            reduced_data.append(red_file)






    radii = []


    for n,red in enumerate(reduced_data):
        # Trim data to area of interest (perhaps not necessary with better background reduction)
        trimmed_data = core.data_processing.cut_image(red, margin=500)  # Margin at 500 good for now

        # Locate center of mass within trimmed image (array)
        com = core.data_processing.locate_focus(trimmed_data)

        # Find aperture with 95% (or other) encircled energy
        os.makedirs(measurements_folder+f"/Radius", exist_ok=True)
        radius = core.data_processing.find_circle_radius(trimmed_data, com, ee_value=0.98, plot=False, save_file=measurements_folder + f"/Radius/datapoint{n}")
        radii.append(radius)

    radius=radii[0]*7.52e-3
    with open(measurements_folder+"radius.json","w") as f:
        json.dump({"Radius":radius}, f)
    print(radius)