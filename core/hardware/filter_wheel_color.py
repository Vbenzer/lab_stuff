"""Module filter_wheel_color.py.

Auto-generated docstring for better readability.
"""
import sys
import elliptec
from time import sleep

filter_to_angle_dict = {'400':240, '450':180, '500':150, '600':90, '700':60, '800':0, 'Open':30, 'Closed':300}

if len(sys.argv) > 1:
    controller = elliptec.Controller('COM4')
    ro = elliptec.Rotator(controller)
    try:
        input = sys.argv[1]
        angle = filter_to_angle_dict[input]
        ro.set_angle(angle)
        sleep(2)
        controller.close_connection()
    except:
        print(sys.argv[1], "is not a valid input")
        print("Please provide a valid integer value of the filters wavelength.\n")
        print("The available filters are: 400, 450, 500, 600, 700, 800, Open for no filter or 'Closed' for no light.\n")
else:
    print("No input provided")


def move(filter: str, progress_signal=None):
    controller = elliptec.Controller('COM4')
    ro = elliptec.Rotator(controller)
    try:
        angle = filter_to_angle_dict[filter]
        if progress_signal:
            progress_signal.emit(f"Moving filter to {filter}")
        ro.set_angle(angle)
        sleep(2)
    except Exception as e:
        print(e)
        print(filter, "is not a valid input")
        print("Please provide a valid integer value of the filters wavelength.\n")
        print("The available filters are: 400, 450, 500, 600, 700, 800, Open for no filter or 'Closed' for no light.\n")
    controller.close_connection()

if __name__ == "__main__":
    move("Closed")