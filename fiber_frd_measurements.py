import numpy as np
import json
import matplotlib.pyplot as plt
import file_save_managment
import analyse_main
from astropy.io import fits

def main_measure_all_filters(project_folder:str, progress_signal=None):
    for i in range(2, 7):
        # Create project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        progress_signal.emit(f"Starting measurement for filter {i}")

        # Run the main measuring pipeline for each filter
        analyse_main.main_measure(filter_folder, progress_signal,
                                  batch_file_path=f"D:\stepper_motor\start_nina_with_fstop_filter{i}.bat")

        progress_signal.emit(f"Measurement for filter {i} complete!")

    print("All filters complete!")
    progress_signal.emit("All filters complete!")

def main_analyse_all_filters(project_folder:str, progress_signal=None):
    f_num = np.zeros(5)
    f_num_err = np.zeros(5)

    for i in range(2, 7):
        # Create project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        progress_signal.emit(f"Starting analysis for filter {i}")

        analyse_main.run_from_existing_files(filter_folder, progress_signal)

        progress_signal.emit(f"Analysis for filter {i} complete!")

        # Load the f-number and its error from the JSON file
        with open(filter_folder + "/Measurements/f_number.json") as f:
            data = json.load(f)
            f_num[i - 2] = data["f_number"]
            f_num_err[i - 2] = data["f_number_err"]

        progress_signal.emit(f"Filter {i} f-number: {f_num[i - 2]}")

    progress_signal.emit("All filters complete! Starting final plot.")

    # Input f-numbers
    input_f_num = np.array([6.21, 5.103, 4.571, 4.063, 3.597]) # These are from the setup_F#_EE_98 file, 18.2.25
    input_f_num_err = np.array([0.04, 0.007, 0.01, 0.005, 0.013])

    # Sort the f-numbers in descending order
    f_num = np.sort(f_num)[::-1]
    f_num_err = np.sort(f_num_err)[::-1]

    # Plot the output f-numbers vs input f-numbers
    plt.errorbar(input_f_num, f_num, xerr=input_f_num_err, yerr=f_num_err, fmt="o", color="blue", label="Data points", capsize=5)
    plt.plot(input_f_num, f_num, linestyle='--', color="blue")
    plt.plot([2.5, 6], [2.5, 6], color="red", label="y=x")
    plt.xlabel("Input f/#")
    plt.ylabel("Output f/#")
    plt.title("Output f/# vs. Input f/#")
    plt.grid(True)
    plt.legend()
    plt.savefig(project_folder + "/f_number_vs_input.png")
    plt.close()
    #plt.show()

    # Set measurement name to last folder name of project folder
    measurement_name = project_folder.split("/")[-1]

    file_save_managment.save_measurement_hdf5("D:/Vincent/frd_measurements.h5", measurement_name, f_num, f_num_err)

def sutherland_plot(project_folder:str):
    # Input f-numbers
    input_f_num = np.array([6.21, 5.103, 4.571, 4.063, 3.597]) # These are from the setup_F#_EE_98 file, 18.2.25
    input_f_num_err = np.array([0.04, 0.007, 0.01, 0.005, 0.013])

    # Load the distance to chip from the JSON file
    dist = np.zeros(5)
    for i in range(2, 7):
        with open(project_folder + f"/filter_{i}/Measurements/f_number.json") as f:
            data = json.load(f)
        dist[i-2] = data["distance_to_chip"]

    distance_to_chip = np.mean(dist)

    ee_list = []

    for i in range(2, 7):
        print(f"Processing filter {i}")
        # Access project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        # Load the LIGHT_0014_0.08s_reduced.fits file for each filter
        reduced_image_path = filter_folder + "/REDUCED/LIGHT_0014_0.08s_reduced.fits"
        with fits.open(reduced_image_path) as hdul:
            reduced_image = hdul[0].data.astype(np.float32)

        # Find center of mass
        import image_analysation as ia
        # Trim data to area of interest (perhaps not necessary with better background reduction)
        trimmed_data = ia.cut_image(reduced_image, margin=500)  # Margin at 500 good for now

        # Locate center of mass within trimmed image (array)
        com = ia.LocateFocus(trimmed_data)

        ee_sublist = []

        for fnum in input_f_num:
            # Calculate the radius of a circle with input f-ratios
            aperture_radius = (distance_to_chip + 9.9) / (2 * fnum) # 9.9: Distance to chip at 0 position

            # Convert to pixels
            aperture_radius = aperture_radius // 7.52e-3

            # Create a circle mask
            import sg_pipeline
            mask = sg_pipeline.create_circular_mask(trimmed_data, (com[0], com[1]), aperture_radius
                                                    , plot_mask=False)

            # Calculate the flux within the mask
            flux = np.sum(mask * trimmed_data)

            # Calculate the flux outside the mask
            mask_outside = np.invert(mask)
            flux_outside = np.sum(mask_outside * trimmed_data)

            # Calculate the encircled energy of the mask
            ee = flux / (flux + flux_outside)
            ee_sublist.append(ee)

        ee_list.append(ee_sublist)

    # Fit function
    def practical_funct_log(x, a, b):
        return 1 / (1 + np.exp(a * (x - b)))

    def practical_funct(x, a, b, c):
        return 1 / (1+a*(x+b)**c)

    def theoretical_funct(x,t):
        return t**2/x**2

    # Fit the data
    from scipy.optimize import curve_fit
    popt_list = []
    p0_list = [[0.005, -3.6, 3.6],[0.02, -3.5, 3.1],[0.073, -3.5, 2.3],[0.2, -3.5,  1.6],[0.24, -3.2,  1.5]]
    for i,ee in enumerate(ee_list):
        popt = curve_fit(practical_funct, input_f_num, ee, p0=p0_list[i])
        popt_list.append(popt)
        print(popt)


    # Plot the encircled energy vs input f-numbers
    labels = ["F/6.21", "F/5.103", "F/4.571", "F/4.063", "F/3.597"]
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    x_range = np.linspace(3.5, 6.5, 1000)
    for idx, ee in enumerate(ee_list):
        print(ee)
        plt.errorbar(input_f_num, ee, xerr=input_f_num_err, fmt="o", color=colors[idx % len(colors)],
                     label=f"Input {labels[idx % len(labels)]}", capsize=5)
        plt.plot(x_range, practical_funct(x_range, *popt_list[idx][0]), linestyle='-', color=colors[idx % len(colors)])
        dynamic_range = np.linspace(input_f_num[idx], 6.5, 1000)
        #plt.plot(dynamic_range, theoretical_funct(dynamic_range, input_f_num[idx]), linestyle='--', color=colors[idx % len(colors)], linewidth=0.5)
    plt.xlabel("Output Aperture f/#")
    plt.ylabel("Encircled Energy")
    plt.title("Encircled Energy vs. Output Aperture f/#")
    plt.grid(True)
    plt.legend()
    plt.savefig(project_folder + "/encircled_energy_vs_output.png")
    plt.close()
    # plt.show()



if __name__ == "__main__":
    project_folder = "D:/Vincent/OptranWF_100_187_P_measurement_3/FRD"
    sutherland_plot(project_folder)