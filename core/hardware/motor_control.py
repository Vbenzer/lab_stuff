import serial
import time

# Set up serial connection parameters
port = "/dev/ttyUSB0"  # Replace with the correct port for your system (e.g., "COM3" for Windows or "/dev/ttyUSB0" for Linux)
port = 'COM7'  #this for windows

baud_rate = 9600  # Check manual for correct baud rate, default is usually 115200
timeout = 1  # 1 second timeout for responses

# Initialize the serial connection
ser = serial.Serial(port, baud_rate, timeout=timeout)
print("Port opened:", ser.name)


def open_connection():
    if ser.is_open:
        print("Port already open.")
    else:
        ser.open()
        print("Port opened:", ser.name)

# Function to send a command to the motor controller
def send_command(command:str):
    """
    Send a command to the motor controller and return the response
    Args:
        command: Command to send

    Returns: Response from the motor controller

    """
    # Ensure command ends with carriage return (\r) as required by the C-663 controller
    command = command + '\n'
    ser.write(command.encode('utf-8'))
    # Give the controller time to process
    time.sleep(0.1)
    # Read and return the response
    response = ser.readline().decode('utf-8').strip()
    return response


def check_error():
    """
    Check if an error has been reported by the controller
    """
    # Send error query command
    error_response = send_command("ERR?")
    if error_response != '0':
        print(f"Error reported:", error_response)
    # You can use the error code to look up specific issues in the manual.
    return error_response


def is_motion_complete():
    """
    Check if all motion is complete
    Returns: True if all motion is complete, False otherwise

    """
    # Send the #5 command to check motion status
    ser.write(b'7')
    time.sleep(0.2)  # Give time for response
    response = ser.readline().decode('utf-8').strip()
    print(ser.readline())
    return response == "0"  # "0" means all motion is complete


def check_motion_status():
    """
    Check the motion status of the motor
    Returns: True if the motor is ready, False otherwise

    """
    # Query the status register
    response = send_command("SRG? 1 1")
    #print(f"Raw SRG? response: {response}")

    if response:
        # Convert the response from hexadecimal to an integer
        status = int(response.split('=')[-1], 16)

        # Check specific bits
        is_moving = bool(status & (1 << 13))  # Bit 13 for "Is Moving"
        is_referencing = bool(status & (1 << 14))  # Bit 14 for "Is Referencing"
        on_target = bool(status & (1 << 15))  # Bit 15 for "On Target"

        ready = not is_moving and not is_referencing

        return ready
    else:
        print("No response received from SRG? command.")
        return None, None, None


def make_reference_move(progress_signal=None):
    """
    Perform a reference move to establish a known position. Also resolves error that sometimes appears when initially
    using the controller.

    Args:
        progress_signal: Signal to emit progress.

    """
    open_connection()

    if progress_signal:
        progress_signal.emit("Making reference move")

    # Find Reference
    send_command("FPL")
    ref = check_error()
    print(ref)

    if ref == "5":
        print("Error 5, try move and reference again")

        if progress_signal:
            progress_signal.emit("Error 5, sending move signal and referencing again")

        move_motor_to_position(5)
        ref = send_command("FPL")
        if ref == "0":
            print("Reference move successful")
            if progress_signal:
                progress_signal.emit("Reference move successful")
        else:
            print("Reference move failed")
            if progress_signal:
                progress_signal.emit("Reference move failed")

    check_error()
    time.sleep(0.1)

    while not check_motion_status():
        print("Referencing in progress...")
        if progress_signal:
            progress_signal.emit("Referencing in progress...")
        check_error()
        time.sleep(1)  # Check every second


# Example of moving the motor by a small step
def move_motor_to_position(position: float, progress_signal=None):
    """
    Move the motor to a specified position
    Args:
        progress_signal:
        position: Float value of the position to move to. Must be within 0 and 9.9 mm.


    """
    # Open the serial connection
    open_connection()

    # Enable the motor if not already enabled
    print("Enabling motor...")
    if progress_signal:
        progress_signal.emit("Enabling motor...")
    send_command(f"SVO 1 1")
    check_error()

    # Move by a specified step size (step_size in motor units, check manual for units)
    print(f"Moving motor by to position {position} mm")
    if progress_signal:
        progress_signal.emit(f"Moving motor to position {position} mm")
    send_command(f"MOV 1 {position}")
    check_error()

    while not check_motion_status():
        print("Movement in progress...")
        if progress_signal:
            progress_signal.emit("Movement in progress...")
        check_error()
        time.sleep(1)  # Check every second

    # Check position after movement
    position_new = send_command("POS?")
    check_error()
    print(f"New position: {position_new}")
    if progress_signal:
        progress_signal.emit(f"New position: {position_new}")
    return position_new

def close_connection():
    """
    Close the serial connection to the motor controller
    """
    ser.close()

# Main routine to move the motor
if __name__ == "__main__":
    try:
        if ser.is_open:
            #send_command("CLR")
            print("Connected to motor controller.")
            #make_reference_move()
            # Set the desired step size (modify as per your needs)
            position = 0  # Change the step size to what is appropriate
            move_motor_to_position(position)
        else:
            print("Failed to open serial connection.")
    finally:
        # Always close the serial connection
        ser.close()
        print("Serial connection closed.")
