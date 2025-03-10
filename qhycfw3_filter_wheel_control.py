import serial
import time

class FilterWheel():
    def __init__(self, port: str):
        super().__init__()
        self.ser = serial.Serial(port, 9600, timeout=3)

        # Disable DTR & RTS to prevent reboot
        self.ser.dtr = False
        self.ser.rts = False

        print("Waiting for filter wheel to be ready...")
        while True:
            self.ser.write(b'NOW\r')  # Query current position
            response = self.ser.readline()
            if response.strip():
                break
            time.sleep(1)
        print("Filter wheel ready!")

    def move_to_filter(self, filter: str):
        name_to_filter_dict = {'2.5': b"0", '6.0': b"1", '5.0': b"2", '4.5': b"3", '4.0': b"4", '3.5': b"5"}
        reverse_dict = {v: k for k, v in name_to_filter_dict.items()}

        try:
            filter = name_to_filter_dict[filter]
            self.ser.write(filter)
            time.sleep(1)
            data = self.ser.readline()
            current_filter = reverse_dict[data]
            print("Current Filter:", current_filter)
        except:
            print(filter, "is not a valid input")
            print("Please provide a valid filter name.\n")
            print("The available filters are: 2.5, 3.5, 4.0, 4.5, 5.0, 6.0\n")

if __name__ == "__main__":
    fw = FilterWheel('COM5')

    while True:
        cmd = input("Enter filter position (or 'exit'): ")
        if cmd.lower() == "exit":
            break
        fw.move_to_filter(cmd)

    fw.ser.close()  # Only close when exiting