import numpy as np
import os
import json

def main(main_folder, calibration_file):
    """
    Main function for the throughput analysis.
    Args:
        main_folder: Path to the main folder.
        calibration_file: Path to the calibration file.
        number_of_measurements: Number of data points to take.
    """
    # Calculate the throughput
    calculate_throughput(main_folder, calibration_file)

    # Plot the throughput
    plot_throughput(main_folder, save=True)

def measure_single_filter(main_folder, filter_name, number_of_measurements):
    """
    Measure the transmission of a single filter.
    Args:
        main_folder: Path to the main folder.
        filter_name: Name of the filter.
        number_of_measurements: Number of data points to take.
    """

    import power_meter_control as pmc
    # Make the measurement
    pmc.make_measurement(main_folder, number_of_measurements, filter_name)

def measure_all_filters(main_folder, number_of_measurements=100, progress_signal=None):
    """
    Measure the transmission of all filters.
    Args:
        main_folder: Path to the main folder.
        number_of_measurements: Number of data points to take.
    """

    import move_to_filter

    # Make the measurement for all filters
    filter_list = ["350", "400", "450", "500", "600", "700", "800"]

    for filter_name in filter_list:
        move_to_filter.move(filter_name)
        measure_single_filter(main_folder, filter_name, number_of_measurements)
        if progress_signal:
            progress_signal.emit(f"Measurement for filter {filter_name} completed.")

def plot_throughput(main_folder:str, save:bool=False):
    """
    Plot the throughput of the filters.
    Args:
        main_folder: Path to the main folder.
    """
    # Load the data
    filter_list = ["350", "400", "450", "500", "600", "700", "800"]
    data = {}
    with open(os.path.join(main_folder, "throughput.json"), 'r') as f:
        data = json.load(f)

    # Plot the throughput
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values())
    ax.set_xlabel("Filter")
    ax.set_ylabel("Throughput")
    ax.set_title("Filter Throughput")
    if save:
        plt.savefig(os.path.join(main_folder, "throughput.png"))
        plt.close()
    else:
        plt.show()

def calculate_throughput(main_folder, calibration_file):
    """
    Calculate the throughput of the filters.
    Args:
        main_folder: Path to the main folder.

    """
    # Load the calibration data
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    # Calculate calibration quotient
    calibration_quotient = np.mean(calibration_data["channel_1"]) / np.mean(calibration_data["channel_2"])

    # Load the data
    filter_list = ["350", "400", "450", "500", "600", "700", "800"]
    data = {}

    for filter_name in filter_list:
        file_name = os.path.join(main_folder, filter_name + ".json")
        with open(file_name, 'r') as f:
            data[filter_name] = json.load(f)

    # Calculate the throughput
    throughput = {}
    for filter_name in filter_list:
        throughput[filter_name] = np.mean(data[filter_name]["channel_1"]) / (calibration_quotient * np.mean(data[filter_name]["channel_2"]))

    # Write throughput to json
    throughput_file = os.path.join(main_folder, "throughput.json")
    with open(throughput_file, 'w') as f:
        json.dump(throughput, f, indent=4)


def create_test_data(main_folder):
    """
    Create test data for the throughput analysis.
    Args:
        main_folder: Path to the main folder.
    """
    import numpy as np
    import os
    import json

    # Create test data
    for filter_name in ["350", "400", "450", "500", "600", "700", "800"]:
        data = {"channel_1": np.random.rand(100).tolist(),
                "channel_2": np.random.rand(100).tolist()}

        file_name = os.path.join(main_folder, filter_name + ".json")
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    main_folder = "D:/Vincent/Test3/"
    calibration_file = "D:/Vincent/Calibration/calibration.json"
    #measure_all_filters(main_folder)
    #calculate_throughput(main_folder, calibration_file)
    #plot_throughput(main_folder)
    #create_test_data(main_folder)
