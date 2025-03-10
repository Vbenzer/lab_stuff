import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
import qhy_ccd_take_image
import image_reduction as ir

def get_ff_with_all_filters(working_directory):
    """
    Takes and reduces one far field image for each color filter
    Args:
        working_directory: Working directory where the images will be saved

    Returns:

    """
    import move_to_filter
    # Define Folders
    dark_folder = os.path.join(working_directory, "darks")
    science_folder = os.path.join(working_directory, "science")
    reduced_folder = os.path.join(working_directory, "reduced")

    # Create folders if they do not exist
    os.makedirs(dark_folder, exist_ok=False)
    os.makedirs(science_folder, exist_ok=False)
    os.makedirs(reduced_folder, exist_ok=False)

    # Define filters
    filters = ["400", "450", "500", "600", "700", "800"]

    # Initialize camera
    camera = qhy_ccd_take_image.Camera(exp_time=2000000)

    # Go to block filter and take darks
    move_to_filter.move("none")
    camera.take_multiple_frames(dark_folder, "dark", num_frames=5)

    for filter_name in filters:
        move_to_filter.move(filter_name)
        camera.take_single_frame(science_folder, filter_name)

    # Close camera
    camera.close()

    # Return to no filter
    move_to_filter.move("0")

    # Create master dark
    mdark = ir.create_master_dark(dark_folder)

    # Reduce all science images
    for image in os.listdir(science_folder):
        if image.endswith(".fits"):
            image_path = os.path.join(science_folder, image)
            with fits.open(image_path) as hdul:
                science_data = hdul[0].data.astype(np.float32)
                reduced_path = os.path.join(reduced_folder, image)
                ir.reduce_image_with_dark(science_data, mdark, output_file=reduced_path, save=True)

def inf_fiber_original_tp_plot(csv_file:str, show=False):
    import pandas as pd
    # Read data from csv file
    print(csv_file)
    data = pd.read_csv(csv_file, header=None)
    list1 = data[0].tolist()
    print(list1)
    #list1 = [float(s.strip(',')) for s in list1]
    list2 = data[1].tolist()
    print(list2)

    if show:
        plt.figure()
        plt.plot(list1, list2, label="Original")
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Throughput")
        plt.show()

    # Convert attenuation to throughput
    fiber_length = 0.005  # km
    attenuation = np.array(list2) * fiber_length
    throughput = 10**(-attenuation/10)


    plt.figure()
    plt.plot(list1, throughput, label="Original in Throughput")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Throughput")
    plt.savefig("D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/Throughput/throughput_original.png")
    if show:
        plt.show()
    plt.close()



    return list1, throughput

def overplot_original_and_data(data, original_data, show=False):
    import json
    wavelength_or, tp_or = inf_fiber_original_tp_plot(original_data)

    # get other data from json file
    with open(data) as f:
        data = json.load(f)

    filters = ["400", "450", "500", "600", "700", "800"]
    wavelength = []
    throughput = []
    for filter in filters:
        wavelength.append(int(filter)*1e-3)
        throughput.append(data[filter])

    plt.figure()
    plt.plot(wavelength_or, tp_or, label="Original")

    # Column plot of measured data
    plt.bar(wavelength, throughput,width=0.05, label="Data")

    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Throughput")
    plt.legend()
    plt.savefig("D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/Throughput/throughput_comparison.png")
    plt.close()




if __name__ == "__main__":
    #working_directory = "D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/FF_with_all_filters"
    #get_ff_with_all_filters(working_directory)
    csv_file = "D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/Throughput/Default Dataset.csv"
    measured_data = "D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/Throughput/throughput.json"
    #inf_fiber_original_tp_plot(csv_file, show=True)
    overplot_original_and_data(measured_data, csv_file, show=True)