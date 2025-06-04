"""Module analyse_main_obs.py.

Auto-generated docstring for better readability.
"""
import core.file_management
import os
import time

from analysis.frd_analysis import run_from_existing_files
from analysis.visualization import plot_cones
from core.file_management import run_batch_file


def main_measure_obsolete(project_folder:str, progress_signal=None, batch_file_path:str= "D:\stepper_motor\start_nina_with_fstop.bat",
                          base_directory:str=None):
    """
    Main function to run the measuring pipeline
    Args:
        project_folder: Path of the project folder.
        progress_signal: Signal to emit progress.
        batch_file_path: Path of the batch file to start N.I.N.A.
        base_directory: Base directory of the project.

    """
    # Start N.I.N.A. with F-stop analysis sequence
    run_batch_file(batch_file_path)

    # Write progress to file
    core.file_management.write_progress("Starting N.I.N.A. with F-stop analysis sequence")

    if progress_signal:
        progress_signal.emit("Starting N.I.N.A. with F-stop analysis sequence")

    # Waiting for N.I.N.A. to complete
    flag_file = "D:/stepper_motor/nina_flag.txt" # Flag file created by N.I.N.A. when sequence is complete
    print("Waiting for N.I.N.A. to complete...")

    if progress_signal:
        progress_signal.emit("Waiting for N.I.N.A. to complete...")

    while not os.path.exists(flag_file):
        time.sleep(10)  # Check every 5 seconds
    print("N.I.N.A. completed!")

    if progress_signal:
        progress_signal.emit("N.I.N.A. completed!")

    # Clean up the flag file
    os.remove(flag_file)

    # Move files to project folder, files are initially saved to the default Nina output folder
    core.file_management.move_files_and_folders(base_directory + r"\nina_output", project_folder)

    time.sleep(1)
    core.file_management.clear_folder(base_directory + r"\nina_output")

    # Close Nina
    run_batch_file("D:\stepper_motor\close_nina.bat")

    # Write progress to file
    core.file_management.write_progress("N.I.N.A. closed, starting analysis pipeline")


if __name__ == "__main__":
    project_folder = r"D:\Vincent\OptranWF_100_187_P_measurement_3\FRD"
    run_from_existing_files(project_folder)

    plot_cones(project_folder)