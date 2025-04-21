import pyvisa
import json
import datetime
import os

rm = pyvisa.ResourceManager()

power_meter = rm.open_resource('USB0::0x1313::0x8022::M00299553::0::INSTR')  # Name of power meter

power_meter.read_termination = '\n'
power_meter.write_termination = '\n'

def make_calibration_measurement(file_name):
    """
    Make a calibration measurement and save the data to a json file.
    Args:
        file_name: Name of the file to save the data to.

    """
    calibration_folder = "D:/Vincent/Calibration/"
    channel_1 = []
    channel_2 = []
    for i in range(0, 100):
        channel_1.append(float(power_meter.query(":POW1:VAL?")))
        channel_2.append(float(power_meter.query(":POW2:VAL?")))

    # Write data to json
    file_path = calibration_folder + file_name
    parameters = {'channel_1': channel_1,
                  'channel_2': channel_2
                  }
    with open(file_path, 'w') as f:
        json.dump(parameters, f, indent=4)


def make_measurement(main_folder, number_of_measurements, file_name):
    """
    Make a measurement and save the data to a json file.
    Args:
        main_folder: Path to the main folder.
        number_of_measurements: Number of measurements to make.
        file_name: Name of the file to save the data to.

    """

    channel_1 = [float(power_meter.query(":POW1:VAL?")) for _ in range(number_of_measurements)]
    channel_2 = [float(power_meter.query(":POW2:VAL?")) for _ in range(number_of_measurements)]

    # Write data to json
    file_path = os.path.join(main_folder, file_name)
    parameters = {'channel_1': channel_1,
                  'channel_2': channel_2
                  }
    with open(file_path, 'w') as f:
        json.dump(parameters, f, indent=4)



if __name__ == '__main__':
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    make_calibration_measurement(date + '_calibration_450.json')


    """"print(power_meter.query(":POW1:VAL?"))
    print(power_meter.query(":POW2:VAL?"))"""