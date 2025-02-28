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
    pmc.make_measurement(main_folder, number_of_measurements, filter_name + ".json")

def measure_all_filters(main_folder, number_of_measurements=100, progress_signal=None, calibration:str=None):
    """
    Measure the transmission of all filters.
    Args:
        main_folder: Path to the main folder.
        number_of_measurements: Number of data points to take.
    """
    import datetime
    import move_to_filter

    if calibration:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        main_folder = "D:/Vincent/Calibration/" + date + "_" + calibration + "/"
        os.makedirs(main_folder, exist_ok=False)

    else:
        os.makedirs(main_folder, exist_ok=False)

    # Make the measurement for all filters
    filter_list = ["400", "450", "500", "600", "700", "800"]

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
    filter_list = ["400", "450", "500", "600", "700", "800"]
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
    # Get calibration quotient
    calibration_quotient = calc_cal_quotient(calibration_file)

    # Load the data
    filter_list = ["400", "450", "500", "600", "700", "800"]
    data = {}

    for filter_name in filter_list:
        file_name = os.path.join(main_folder, filter_name + ".json")
        with open(file_name, 'r') as f:
            data[filter_name] = json.load(f)

    print(data)

    # Filter out infinities and zeros
    data = data[np.isfinite(test) & (test != 0)]

    # Calculate the throughput
    throughput = {}
    for filter_name in filter_list:
        throughput[filter_name] = np.mean(data[filter_name]["channel_1"]) / (calibration_quotient * np.mean(data[filter_name]["channel_2"]))

    # Write throughput to json
    throughput_file = os.path.join(main_folder, "throughput.json")
    with open(throughput_file, 'w') as f:
        json.dump(throughput, f, indent=4)

def calc_cal_quotient_folder(calibration_folder:str):
    calibration_file_list = os.listdir(calibration_folder)

    cal_qou_list = []

    for file in calibration_file_list:
        calibration_quotient = calc_cal_quotient(os.path.join(calibration_folder, file))
        cal_qou_list.append(calibration_quotient)

    # Write calibration quotient to json
    cal_qou_file = os.path.join(calibration_folder, "calibration_quotient.json")
    with open(cal_qou_file, 'w') as f:
        json.dump(cal_qou_list, f, indent=4)

def calc_cal_quotient(calibration_file:str, folder:bool=False):
    # Load the calibration data

    if folder:
        calibration_file_list = os.listdir(calibration_file)
        for file in calibration_file_list:
            with open(file, 'r') as f:
                calibration_data = json.load(f)

    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    # Remove infinities from the data
    calibration_data["channel_1"] = [x for x in calibration_data["channel_1"] if x != float("inf")]
    calibration_data["channel_2"] = [x for x in calibration_data["channel_2"] if x != float("inf")]

    # Calculate calibration quotient
    calibration_quotient = np.mean(calibration_data["channel_1"]) / np.mean(calibration_data["channel_2"])

    return calibration_quotient
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
    for filter_name in ["400", "450", "500", "600", "700", "800"]:
        data = {"channel_1": np.random.rand(100).tolist(),
                "channel_2": np.random.rand(100).tolist()}

        file_name = os.path.join(main_folder, filter_name + ".json")
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    main_folder = "D:/Vincent/Test3/"
    calibration_file = "D:/Vincent/Calibration/2025-01-27_calibration_1_good.json"
    calibration_file = "D:/Vincent/Calibration/2025-01-30_calibration_450_6.json"
    measure_all_filters(main_folder, calibration="calibration_1")
    calc_cal_quotient_folder("D:/Vincent/Calibration/2025-01-30_calibration_1")
    #print(calc_cal_qoutient(calibration_file))
    #measure_all_filters(main_folder)
    #calculate_throughput(main_folder, calibration_file)
    #plot_throughput(main_folder)
    #create_test_data(main_folder)
