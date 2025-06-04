"""Module fiber_frd_measurements_obs.py.

Auto-generated docstring for better readability.
"""
from analysis.general_analysis import measure_fiber_size


def main_measure_all_filters_obs(project_folder:str, progress_signal=None, base_directory=None):
    """
    Run the frd measuring pipeline for all filters. Still with NINA implementation.
    Args:
        project_folder: Path to the project folder.
        progress_signal: Progress signal.
        base_directory: Base directory of the project.

    Returns:

    """
    for i in range(2, 7):
        # Create project subfolder for each filter
        filter_folder = project_folder + f"/filter_{i}"

        progress_signal.emit(f"Starting measurement for filter {i}")

        # Run the main measuring pipeline for each filter
        analyse_main.main_measure_obsolete(filter_folder, progress_signal,
                                           batch_file_path=f"D:\\stepper_motor\\start_nina_with_fstop_filter{i}.bat",
                                           base_directory=base_directory)

        progress_signal.emit(f"Measurement for filter {i} complete!")

    print("All filters complete!")
    progress_signal.emit("All filters complete!")


if __name__ == "__main__":
    #project_folder = "/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/R_25x40_0000_0001/NF_FF"
    #project_folder = "D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5_measurement_2/NF_FF"
    project_folder = r"\\srv4\labshare\raw_data\fibers\Measurements\test"
    measure_fiber_size(project_folder, {"entrance_cam": "10ms", "exit_cam": "0.5ms"})
    #sutherland_plot(project_folder)
    #plot_f_ratio_circles_on_raw(project_folder)
    #nf_ff_capture(project_folder, 28, {"entrance_cam": "10ms", "exit_cam": 25000})
    #nf_ff_process(project_folder, [25,40])
    #plot_horizontal_cut(project_folder)