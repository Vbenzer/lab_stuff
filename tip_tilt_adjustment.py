import image_analysation as ia
import os
from skimage import io
import json
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import file_mover

"""# Plot image
plt.imshow(image)
plt.show()"""
def analyse_f_number(image, measurements_folder):
    # Calculate radius
    trimmed_data = ia.cut_image(image, margin=500)  # Margin at 500 good for now

    # Locate center of mass within trimmed image (array)
    com = ia.LocateFocus(trimmed_data)

    # Swap x and y
    com = (com[1], com[0])

    print("Calculating radius...")

    # Find aperture with encircled energy
    os.makedirs(measurements_folder, exist_ok=True)
    radius = ia.find_circle_radius(trimmed_data, com, ee_value=0.98, plot=True, save_data=False)

    # Close external matplotlib window



    print(f"Radius: {radius}")

    # Add the radius to existing json file
    # Path to the JSON file
    json_file_path = measurements_folder + "radii.json"

    # Check if the file exists
    if os.path.exists(json_file_path):
        # Load existing data
        with open(json_file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        # Initialize an empty list if the file does not exist
        existing_data = {"radius": []}

    # Append new data
    radii = existing_data["radius"]

    # Delete old measurements if too many
    if len(radii) > 200:
        radii = radii[-200:]

    # Don't append if radius deviates too much from the mean
    if len(radii) > 0:
        mean_radius = np.mean(radii)
        if abs(radius - mean_radius) > 200:
            print("Radius deviates too much from mean. Not appending.")
        else:
            radii.append(radius)
    else:
        radii.append(radius)

    new_data = {"radius": radii}

    # Write the updated data back to the file
    with open(json_file_path, 'w') as file:
        json.dump(new_data, file, indent=4)

    # Plot radius vs number of measurements]
    plt.plot(radii)
    plt.xlabel("Number of measurements")
    plt.ylabel("Radius")
    plt.title("Radius vs Number of Measurements")
    plt.savefig(measurements_folder + "radius_vs_measurements.png")
    plt.close()


if __name__ == "__main__":
    # Nina output folder
    nina_output = "D:/Vincent/nina_output/SNAPSHOT/"
    measurements_folder = "D:/Vincent/tip_tilt/"

    image_name = os.listdir(nina_output)[0]

    with fits.open(nina_output + image_name) as hdul:
        image = hdul[0].data.astype(np.float32)  # Convert to float for precision

    analyse_f_number(image, measurements_folder)

    # Clear Nina folder
    file_mover.clear_folder(nina_output)

