import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constant input f-number (the same for all measurements)
input_f_num = np.array([6.152, 5.103, 4.57, 4.089, 3.578]) # These are from the setup_F#_EE_98_Measurement_2 file, 31.3.25
input_f_num_err = np.array([0.003, 0.007, 0.03, 0.01, 0.021])


# Function to read measurements from HDF5 file and plot the data
def plot_measurements(filename):
    # Read data from the HDF5 file
    with h5py.File(filename, 'r') as file:
        # Initialize empty lists to store data for plotting
        f_numbers = []
        f_numbers_err = []
        measurement_names = []

        # Loop through all groups (each measurement) in the HDF5 file
        for measurement_name in file.keys():
            measurement_group = file[measurement_name]
            f_number = np.array(measurement_group['f_number'])
            f_number_err = np.array(measurement_group['f_number_err'])

            # Append the data for this measurement
            f_numbers.append(f_number)
            f_numbers_err.append(f_number_err)
            measurement_names.append(measurement_name)

        # Convert lists to numpy arrays for easier handling
        f_numbers = np.array(f_numbers)
        f_numbers_err = np.array(f_numbers_err)

    # Plot the measurements
    plt.figure(figsize=(10, 6))

    # Create a colormap to use for different measurements
    colors = cm.plasma(np.linspace(0, 1, len(measurement_names)))

    # If exists remove "Vincent" from the measurement names
    measurement_names = [name.replace("Vincent\\", "") for name in measurement_names]
    measurement_names = [name.replace("\\FRD", "") for name in measurement_names]

    # Plot each measurement with error bars and dashed lines
    for i, measurement_name in enumerate(measurement_names):
        color = colors[i]  # Get the color for this measurement
        plt.errorbar(input_f_num, f_numbers[i],xerr=input_f_num_err, yerr=f_numbers_err[i], fmt='o', label=measurement_name, color=color)
        plt.plot(input_f_num, f_numbers[i], linestyle='--', color=color)  # Dashed line with the same color

    # Add reference line y = x (no FRD)
    plt.plot([min(input_f_num), max(input_f_num)], [min(input_f_num), max(input_f_num)], 'k--', label='y = x (No FRD)')

    # Labels and title
    plt.xlabel('Input f/#')
    plt.ylabel('Output f/#')
    plt.title('Output f/# vs. Input f/#')

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig("D:/Vincent/FRD_main_plot.png")
    plt.show()


# Example usage
filename = 'D:/Vincent/frd_measurements.h5'
plot_measurements(filename)
