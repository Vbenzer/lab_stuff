import numpy as np
import json
import matplotlib.pyplot as plt
import file_save_managment
import analyse_main

def main(project_folder:str, measurement_name:str):
    f_num = np.zeros(5)
    f_num_err = np.zeros(5)

    for i in range(2,7):
        # Create project subfolder for each filter
        project_folder += f"/filter_{i}"

        # Write progress to file
        file_save_managment.write_progress(f"Starting analysis for filter: {i}")

        # Run the main analysis pipeline for each filter
        analyse_main.main(project_folder, measurement_name, batch_file_path=f"D:\stepper_motor\start_nina_with_fstop_filter{i}.bat")

        # Load the f-number and its error from the JSON file
        with open(project_folder+"/Measurements/f_number.json") as f:
            data = json.load(f)
            f_num[i-2] = data["f_number"]
            f_num_err[i-2] = data["f_number_err"]

        print(f"Filter {i} complete!")
    print("All filters complete!")

    # Write progress to file
    file_save_managment.write_progress("All filters complete, forming final plot")

    # Input f-numbers
    input_f_num = np.array([6, 5, 4.5, 4, 3.5])

    # Sort the f-numbers in descending order
    input_f_num = np.sort(input_f_num)[::-1]
    f_num = np.sort(f_num)[::-1]
    f_num_err = np.sort(f_num_err)[::-1]

    # Plot the output f-numbers vs input f-numbers
    plt.errorbar(input_f_num, f_num, yerr=f_num_err, fmt="o", color="blue", label="Data points", capsize=5)
    plt.plot(input_f_num, f_num, linestyle='--', color="blue")
    plt.plot([2.5, 6], [2.5, 6], color="red", label="y=x")
    plt.xlabel("Input f/#")
    plt.ylabel("Output f/#")
    plt.title("Output f/# vs. Input f/#")
    plt.grid(True)
    plt.savefig(project_folder+"/f_number_vs_input.png")
    plt.legend()
    plt.show()

    file_save_managment.save_measurement_hdf5("D:/Vincent/frd_measurements.h5", measurement_name, f_num, f_num_err)

if __name__ == "__main__":
    main("D:/Vincent/fiber_full_frd_0", "fiber_full_frd_0")