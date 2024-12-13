import image_analysation
import image_reduction
import find_f_number
import file_mover
from astropy.io import fits
import os
import numpy as np
import subprocess
import time

def run_batch_file(batch_file_path):
    """
    Run a batch file using subprocess.
    """
    try:
        # Use subprocess to run the batch file
        result = subprocess.run(batch_file_path, shell=True, check=True, text=True)
        print(f"Batch file executed successfully with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the batch file: {e}")
def main(project_folder:str):

    run_batch_file("D:\stepper_motor\start_nina_with_fstop.bat")

    # Wait for the flag file to appear
    flag_file = "D:/stepper_motor/nina_flag.txt"
    print("Waiting for N.I.N.A. to complete...")
    while not os.path.exists(flag_file):
        time.sleep(10)  # Check every 5 seconds

    print("N.I.N.A. completed!")
    # Clean up the flag file
    os.remove(flag_file)

    #Move files to project folder
    file_mover.move_files_and_folders("D:/Vincent/nina_output", project_folder)

    time.sleep(1)
    file_mover.clear_folder("D:/Vincent/nina_output")

    #Close Nina
    run_batch_file("D:\stepper_motor\close_nina.bat")

    dark_folder = project_folder+"/DARK"
    light_folder = project_folder+"/LIGHT"
    reduce_images_folder = project_folder+"/REDUCED/"
    measurements_folder = project_folder+"/Measurements/"
    os.makedirs(measurements_folder, exist_ok=True)

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
        os.makedirs(measurements_folder+f"/Radius", exist_ok=True)
        radius = image_analysation.find_circle_radius(trimmed_data, com, ee_value=0.98, plot=False, save_file=measurements_folder+f"/Radius/datapoint{n}")
        radii.append(radius)

    radii.sort()
    pos_values.sort()

    radii = np.array(radii)
    pos_values = np.array(pos_values)


    print('radii:',radii,'pos:',pos_values)

    f_number,f_number_err = find_f_number.calculate_f_number(radii, pos_values, plot_regression=True, save_path=measurements_folder)
    print(f"Calculated F-number (f/#): {f_number:.3f} ± {f_number_err:.3f}")
