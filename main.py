import os
import tkinter as tk
from tkinter import messagebox, simpledialog

def show_info_message(message_title: str, message: str, duration=1000):
    """Show an info message that disappears after a specified duration."""
    root = tk.Tk()  # Create a temporary root window
    root.withdraw()  # Hide the root window

    # Create a Toplevel window as the message container
    info_box = tk.Toplevel(root)
    info_box.title(message_title)

    # Add a label for the message
    label = tk.Label(info_box, text=message, padx=20, pady=10)
    label.pack()

    # Center the info_box on the screen
    info_box.update_idletasks()
    x = (info_box.winfo_screenwidth() - info_box.winfo_width()) // 2
    y = (info_box.winfo_screenheight() - info_box.winfo_height()) // 2
    info_box.geometry(f"+{x}+{y}")

    # Destroy the window after the duration
    info_box.after(duration, info_box.destroy)
    root.after(duration, root.destroy)  # Clean up the root window after the duration

    root.mainloop()

def run_experiment():
    measurement_name = entry.get()
    folder_path = os.path.join(base_directory, measurement_name)

    if os.path.exists(folder_path):
        response = messagebox.askyesno("Warning",
                                       "Folder already exists! Do you want to proceed with the existing folder?")
        if not response:
            message_label.config(text="Please enter a new measurement name!")
            return
    else:
        os.makedirs(folder_path)
        show_info_message("Success!", f"Folder '{measurement_name}' created successfully!")

    # Perform actions based on checkbox selections
    if var1.get():
        run_code_1(folder_path)
    if var2.get():
        run_code_2(folder_path)
    if var3.get():
        run_code_3(folder_path)

    # Close the window
    root.destroy()

def run_code_1(folder_path):
    # Code to run the first experiment
    import analyse_main
    analyse_main.main(folder_path)
    print(f"Running code 1 with folder: {folder_path}")
    # Add your code here

def run_code_2(folder_path):
    import collimation_test
    collimation_test.main(folder_path)
    print(f"Running code 2 with folder: {folder_path}")
    # Add your code here

def run_code_3(folder_path):
    # Code to run the third experiment
    print(f"Running code 3 with folder: {folder_path}")
    # Add your code here

# Base directory where folders will be created
base_directory = r"D:\Vincent"

# Create the main window
root = tk.Tk()
root.title("Measurement Executor")

root.update_idletasks()
x = (root.winfo_screenwidth() - root.winfo_width()) // 2
y = (root.winfo_screenheight() - root.winfo_height()) // 2
root.geometry(f"+{x}+{y}")

# Create and place the widgets
tk.Label(root, text="Measurement Name:").grid(row=0, column=0, padx=10, pady=10)
entry = tk.Entry(root)
entry.grid(row=0, column=1, padx=10, pady=10)

var1 = tk.BooleanVar()
var2 = tk.BooleanVar()
var3 = tk.BooleanVar()

tk.Checkbutton(root, text="Measure F/# from filter", variable=var1).grid(row=2, column=0, sticky=tk.W, padx=10)
tk.Checkbutton(root, text="Compute Aperture Radius", variable=var2).grid(row=2, column=1, sticky=tk.W, padx=10)
tk.Checkbutton(root, text="Option 3", variable=var3).grid(row=2, column=2, sticky=tk.W, padx=10)

tk.Button(root, text="Run Experiment", command=run_experiment).grid(row=3, column=0, columnspan=3, pady=20)

# Label for displaying messages
message_label = tk.Label(root, text="", fg="red")
message_label.grid(row=1, column=0, columnspan=3)

# Run the application
root.mainloop()