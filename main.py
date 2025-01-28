import os
import threading
import tkinter as tk
import time

# Todo: Change it so that folder or fiber diameter input are only required when a experiment is selected that
#  actually needs it.
def check_folder_exists(*args):
    """
    Check if the folder already exists in the base directory.
    """
    folder_name = folder_name_var.get()
    folder_path = os.path.join(base_directory, folder_name)
    if os.path.exists(folder_path):
        message_label.config(text="Folder already exists", fg="red")
    else:
        message_label.config(text="Valid Name", fg="green")
    selected_folder_label.config(text=f"Selected Folder: {folder_path}")
    update_experiment_status() # Update the experiment status when the folder name changes

def clear_message():
    """
    Clear the message label after a delay.
    """
    message_label.config(text="")

def create_folder():
    """
    Create a new folder in the base directory.
    """
    global folder_selected
    folder_name = folder_name_var.get()
    folder_path = os.path.join(base_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        message_label.config(text="Folder created successfully!", fg="green")
        checkmark_label.config(text="✔", fg="green")
    else:
        message_label.config(text="Using existing folder", fg="blue")
        checkmark_label.config(text="✔", fg="green")
    root.after(1000, clear_message)
    folder_selected = True
    run_experiment_button.config(state=tk.NORMAL)

def update_experiment_status():
    """
    Update the experiment status based on the selected experiments.
    """
    selected_experiments = []
    if var1.get():
        selected_experiments.append("Measure F/# from filter")
    if var2.get():
        selected_experiments.append("Compute Aperture Radius")
    if var3.get():
        selected_experiments.append("Fiber FRD Measurement")
    if var4.get():
        selected_experiments.append("Test")
    if var5.get():
        selected_experiments.append("Scrambling Gain Measurement")

    if not selected_experiments:
        experiment_status_label.config(text="No experiment selected", fg="red")
    else:
        experiment_status_label.config(text=f"Selected: {', '.join(selected_experiments)}", fg="blue")

def run_experiment():
    """
    Run the selected experiments in separate threads.
    """
    def experiment_thread():
        folder_name = folder_name_var.get()
        folder_path = os.path.join(base_directory, folder_name)

        if fiber_shape_var.get() == "":
            raise ValueError("Fiber shape cannot be empty")
        else:
            fiber_shape = fiber_shape_var.get()

        if fiber_diameter_var.get() == "":
            fiber_diameter = 0
        else:
            fiber_diameter = int(fiber_diameter_var.get())

        if not os.path.exists(folder_path):
            root.after(0, lambda: message_label.config(text="Folder does not exist. Please create it first.", fg="red"))
            return

        if var1.get():
            run_code_1(folder_path, folder_name)
        if var2.get():
            run_code_2(folder_path)
        if var3.get():
            run_code_3(folder_path, folder_name)
        if var4.get():
            run_code_4(folder_path, fiber_diameter)
        if var5.get():
            run_code_5(folder_path, fiber_diameter, fiber_shape)

        root.after(1000, clear_message)
        root.after(0, root.destroy)

    entry.config(state='disabled')
    fiber_diameter_entry.config(state='disabled')
    fiber_shape_entry.config(state='disabled')
    create_folder_button.config(state='disabled')
    run_experiment_button.config(state='disabled')
    checkbuttons = [var1_checkbutton, var2_checkbutton, var3_checkbutton, var4_checkbutton]
    for cb in checkbuttons:
        cb.config(state='disabled')
    running_label.config(text="Currently Running", fg="blue")

    threading.Thread(target=experiment_thread).start()
    threading.Thread(target=read_progress).start()

def read_progress():
    """
    Read the progress file and update the progress text widget.
    """
    progress_file = "progress.txt"

    with open(progress_file, "a") as f:
        f.write("Starting experiment\n")

    while os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            lines = f.readlines()
        if lines:
            root.after(0, lambda: update_progress_text(lines))
        time.sleep(1)

def update_progress_text(lines):
    """
    Update the progress text widget with the given lines
    Args:
        lines: Description of the current step.
    """
    progress_text.config(state=tk.NORMAL)
    progress_text.delete(1.0, tk.END)
    for line in lines:
        progress_text.insert(tk.END, line)
    progress_text.see(tk.END)
    progress_text.config(state=tk.DISABLED)
    root.update_idletasks()
    root.geometry("")

def run_code_1(folder_path, folder_name):
    import analyse_main
    analyse_main.main_measure(folder_path, folder_name)

def run_code_2(folder_path):
    import collimation_test
    collimation_test.main(folder_path)

def run_code_3(folder_path, folder_name):
    import fiber_frd_measurements
    fiber_frd_measurements.main(folder_path, folder_name)

def run_code_4(folder_path, fiber_diameter):
    if fiber_diameter == 0:
        raise ValueError("Fiber diameter cannot be empty")

    import experiment_running_test
    print(f"Running code 4 with fiber diameter: {fiber_diameter}")
    experiment_running_test.main(folder_path)

def run_code_5(folder_path, fiber_diameter, fiber_shape):
    if fiber_diameter == 0:
        raise ValueError("Fiber diameter cannot be empty")

    import sg_pipeline
    print(f"Running code 5 with fiber diameter: {fiber_diameter} and fiber shape: {fiber_shape}")
    sg_pipeline.main(fiber_diameter, fiber_shape)
def on_closing():
    stop_event.set()
    root.destroy()

base_directory = r"D:\Vincent"
folder_selected = False

root = tk.Tk()
root.title("Measurement Executor")

root.update_idletasks()
x = (root.winfo_screenwidth() - root.winfo_width()) // 2
y = (root.winfo_screenheight() - root.winfo_height()) // 2
root.geometry(f"+{x}+{y}")

tk.Label(root, text="Measurement Name:").grid(row=0, column=0, padx=10, pady=10)

folder_name_var = tk.StringVar()
folder_name_var.trace("w", check_folder_exists)
entry = tk.Entry(root, textvariable=folder_name_var)
entry.grid(row=0, column=1, padx=5, pady=5)

# Add a label and entry for fiber diameter
tk.Label(root, text="Fiber Diameter:").grid(row=1, column=0, padx=10, pady=10)
fiber_diameter_var = tk.StringVar()
fiber_diameter_entry = tk.Entry(root, textvariable=fiber_diameter_var)
fiber_diameter_entry.grid(row=1, column=1, padx=5, pady=5)

# Add a label for fiber shape
tk.Label(root, text="Fiber Shape:").grid(row=2, column=0, padx=10, pady=10)
fiber_shape_var = tk.StringVar()
fiber_shape_entry = tk.Entry(root, textvariable=fiber_shape_var)
fiber_shape_entry.grid(row=2, column=1, padx=5, pady=5)

var1 = tk.BooleanVar()
var2 = tk.BooleanVar()
var3 = tk.BooleanVar()
var4 = tk.BooleanVar()
var5 = tk.BooleanVar()

var1_checkbutton = tk.Checkbutton(root, text="Measure F/# from filter", variable=var1, command=update_experiment_status)
var2_checkbutton = tk.Checkbutton(root, text="Compute Aperture Radius", variable=var2, command=update_experiment_status)
var3_checkbutton = tk.Checkbutton(root, text="Fiber FRD Measurement", variable=var3, command=update_experiment_status)
var4_checkbutton = tk.Checkbutton(root, text="Test", variable=var4, command=update_experiment_status)
var5_checkbutton = tk.Checkbutton(root, text="Scrambling Gain Measurement", variable=var5, command=update_experiment_status)

var1_checkbutton.grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
var2_checkbutton.grid(row=4, column=1, sticky=tk.W, padx=10, pady=5)
var3_checkbutton.grid(row=5, column=0, sticky=tk.W, padx=10, pady=5)
var4_checkbutton.grid(row=5, column=1, sticky=tk.W, padx=10, pady=5)
var5_checkbutton.grid(row=4, column=2, sticky=tk.W, padx=10, pady=5)

create_folder_button = tk.Button(root, text="Create/Use Folder", command=create_folder)
create_folder_button.grid(row=0, column=2, columnspan=1, pady=10)
run_experiment_button = tk.Button(root, text="Run Experiment", command=run_experiment, state=tk.DISABLED)
run_experiment_button.grid(row=7, column=0, columnspan=3, pady=10)

message_label = tk.Label(root, text="", fg="red")
message_label.grid(row=2, column=0, columnspan=3)

selected_folder_label = tk.Label(root, text="Selected Folder: ", fg="black")
selected_folder_label.grid(row=3, column=0, columnspan=3, pady=10)

checkmark_label = tk.Label(root, text="", fg="green")
checkmark_label.grid(row=3, column=2, pady=10)

experiment_status_label = tk.Label(root, text="No experiment selected", fg="red")
experiment_status_label.grid(row=6, column=0, columnspan=3)

running_label = tk.Label(root, text="Currently Running", fg="blue")
running_label.grid(row=8, column=0, columnspan=3, pady=10)

progress_text = tk.Text(root, height=10, state=tk.DISABLED)
progress_text.grid(row=9, column=0, columnspan=3, pady=10)

stop_event = threading.Event()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()