"""Utility functions for analysing filter throughput.

The routines in this module are used to measure the transmission
of different colour filters and to compute the throughput relative
to calibration data.  The functions were originally written as part
of a measurement GUI and therefore rely on measurement files stored
in ``JSON`` format.  Only a very small subset of error handling is
performed here as the data is assumed to be well formed.
"""
import json
import os
from typing import Dict, List

import numpy as np


def main(main_folder, calibration_folder):
    """
    Main function for the throughput analysis.
    Args:
        main_folder: Path to the main folder.
        calibration_folder: Path to the calibration file.
    """
    # Calculate the throughput
    calculate_throughput(main_folder, calibration_folder)

    # Plot the throughput
    plot_throughput(main_folder, save=True)


def _load_measurement_files(main_folder: str, filter_list: List[str]) -> Dict[str, dict]:
    """Return measurement data for each filter.

    Parameters
    ----------
    main_folder : str
        Directory containing ``*.json`` measurement files.
    filter_list : List[str]
        Names of the filters that should be loaded.

    Returns
    -------
    Dict[str, dict]
        Mapping of filter name to loaded JSON content.
    """

    data: Dict[str, dict] = {}
    for filter_name in filter_list:
        file_name = os.path.join(main_folder, f"{filter_name}.json")
        with open(file_name, "r") as f:
            data[filter_name] = json.load(f)
    return data


def _sanitize_measurements(data: Dict[str, dict], min_points: int = 20) -> None:
    """Clean measurement data in-place.

    Removes ``inf`` and zero entries from the measurement lists and ensures a
    minimum number of data points is available.  If too few values remain the
    channel is replaced with ``[0]`` and ``[1]`` to signal an invalid
    measurement.

    Parameters
    ----------
    data : Dict[str, dict]
        Dictionary returned by :func:`_load_measurement_files`.
    min_points : int, optional
        Minimum number of valid measurements required.
    """

    for values in data.values():
        values["channel_1"] = [x for x in values["channel_1"] if x not in (np.inf, 0.0)]
        values["channel_2"] = [x for x in values["channel_2"] if x not in (np.inf, 0.0)]

        if len(values["channel_1"]) < min_points or len(values["channel_2"]) < min_points:
            values["channel_1"] = [0]
            values["channel_2"] = [1]

def measure_single_filter(main_folder, filter_name, number_of_measurements):
    """
    Measure the transmission of a single filter.
    Args:
        main_folder: Path to the main folder.
        filter_name: Name of the filter.
        number_of_measurements: Number of data points to take.
    """

    from core.hardware import power_meter_control as pmc
    # Make the measurement
    pmc.make_measurement(main_folder, number_of_measurements, filter_name + ".json")

def measure_all_filters(main_folder, number_of_measurements=100, progress_signal=None, calibration:str=None, base_directory:str=None):
    """
    Measure the transmission of all filters.
    Args:
        main_folder: Path to the main folder.
        number_of_measurements: Number of data points to take.
        progress_signal: Signal to emit progress.
        calibration: Name of the calibration file.
        base_directory: Base directory of the project.
    """
    import datetime

    if calibration:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        main_folder = base_directory + r"/Calibration/" + date + "_" + calibration + "/"
        os.makedirs(main_folder, exist_ok=False)

    else:
        os.makedirs(main_folder, exist_ok=False)

    # Make the measurement for all filters
    filter_list = ["400", "450", "500", "600", "700", "800"]

    for filter_name in filter_list:
        if progress_signal:
            progress_signal.emit(f"Measuring filter {filter_name}")

        move_to_filter.move(filter_name)

        measure_single_filter(main_folder, filter_name, number_of_measurements)

    # When done reset filter to none
    if progress_signal:
        progress_signal.emit("Measuring done, resetting filter to none")
    move_to_filter.move("0")

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
    ax.set_xlabel("Filter [nm]")
    ax.set_ylabel("Throughput [%]")
    ax.set_title("Filter Throughput")
    if save:
        plt.savefig(os.path.join(main_folder, "throughput.png"))
        plt.close()
    else:
        plt.show()

def calculate_throughput(main_folder, calibration_folder):
    """
    Calculate the throughput of the filters.
    Args:
        main_folder: Path to the main folder.

    """
    # Obtain the calibration factors used to scale the measurements
    calibration_quotient_list = calc_cal_quotient(calibration_folder)

    # Persist calibration data for later inspection
    cal_qou_file = os.path.join(main_folder, "calibration_quotient.json")
    with open(cal_qou_file, "w") as f:
        json.dump(calibration_quotient_list, f, indent=4)

    filter_list = ["400", "450", "500", "600", "700", "800"]

    # Load all filter measurement files
    data = _load_measurement_files(main_folder, filter_list)

    # Clean up measurement lists (remove infinities/zeros)
    _sanitize_measurements(data)

    # Calculate the throughput for each filter
    throughput = {}
    for filter_name, calibration_quotient in zip(filter_list, calibration_quotient_list):
        avg_ch1 = np.mean(data[filter_name]["channel_1"])
        avg_ch2 = np.mean(data[filter_name]["channel_2"])
        throughput[filter_name] = avg_ch1 / (calibration_quotient * avg_ch2)

    # Write throughput to json
    throughput_file = os.path.join(main_folder, "throughput.json")
    with open(throughput_file, 'w') as f:
        json.dump(throughput, f, indent=4)

def calc_cal_quotient_folder(calibration_folder: str) -> None:  # deprecated
    """Create ``calibration_quotient.json`` from all files in ``calibration_folder``.

    The function iterates over all JSON files in ``calibration_folder`` and
    computes the calibration quotient for each using
    :func:`calc_cal_quotient`.  The resulting list is written back to the same
    folder for later use.  This helper remains for backwards compatibility and
    might be removed in the future.

    Parameters
    ----------
    calibration_folder : str
        Folder containing the calibration measurements.
    """

    calibration_file_list = os.listdir(calibration_folder)

    cal_qou_list = []

    for file in calibration_file_list:
        calibration_quotient = calc_cal_quotient(os.path.join(calibration_folder, file))
        cal_qou_list.append(calibration_quotient)

    # Write calibration quotient list back to disk
    cal_qou_file = os.path.join(calibration_folder, "calibration_quotient.json")
    with open(cal_qou_file, "w") as f:
        json.dump(cal_qou_list, f, indent=4)

def calc_cal_quotient(calibration_folder: str) -> List[float]:
    """Return calibration quotients for all JSON files in *calibration_folder*.

    The calibration quotient is defined as the ratio between the mean of
    ``channel_1`` and ``channel_2`` values stored in each calibration file.

    Parameters
    ----------
    calibration_folder : str
        Path to the folder containing the calibration measurement files.

    Returns
    -------
    List[float]
        One quotient for every file found in the folder.
    """
    # Load the calibration data
    data_list = []
    calibration_file_list = os.listdir(calibration_folder)
    for file in calibration_file_list:
        print("Reading file:", file)
        with open(calibration_folder + "/" + file, 'r') as f:
            calibration_data = json.load(f)
        data_list.append(calibration_data)


    # Remove infinities and zeros from the data
    for calibration_data in data_list:
        #print([x for x in calibration_data["channel_1"] if x != np.inf and x != 0.0])
        calibration_data["channel_1"] = [x for x in calibration_data["channel_1"] if x != np.inf and x != 0.0]
        calibration_data["channel_2"] = [x for x in calibration_data["channel_2"] if x != np.inf and x != 0.0]

        if len(calibration_data["channel_1"]) < 20 or len(calibration_data["channel_2"]) < 20:
            print("Warning: Data amount small")

    calibration_quotient_list = []
    for calibration_data in data_list:
        # Calculate calibration quotient
        calibration_quotient = np.mean(calibration_data["channel_1"]) / np.mean(calibration_data["channel_2"])
        calibration_quotient_list.append(calibration_quotient)

    return calibration_quotient_list

def create_test_data(main_folder):
    """Generate random measurement files for manual testing.

    Parameters
    ----------
    main_folder : str
        Directory where the JSON files will be created.
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
    main_folder = "D:/Vincent/OptranWF_100_187_P_measurement_6/Throughput"
    calibration_file = "D:/Vincent/Calibration/2025-01-27_calibration_1_good.json"
    calibration_folder = "D:/Vincent/Calibration/2025-02-28_calibration_new_ls"
    #measure_all_filters(main_folder, calibration="calibration_1")
    #calc_cal_quotient_folder("D:/Vincent/Calibration/2025-01-30_calibration_1")
    #print(calc_cal_qoutient(calibration_file))
    #measure_all_filters(main_folder)
    #calculate_throughput(main_folder, calibration_file)
    #plot_throughput(main_folder)
    #create_test_data(main_folder)
    main(main_folder, calibration_folder)
