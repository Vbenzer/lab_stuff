import os
import tkinter as tk
from tkinter import messagebox, simpledialog



def run_experiment():
    measurement_name = entry.get()
    folder_path = os.path.join(base_directory, measurement_name)

    if os.path.exists(folder_path):
        response = messagebox.askyesno("Warning",
                                       "Folder already exists! Do you want to proceed with the existing folder?")
        if not response:
            new_name = simpledialog.askstring("Input", "Please enter a new measurement name:")
            if new_name:
                folder_path = os.path.join(base_directory, new_name)
                if os.path.exists(folder_path):
                    messagebox.showerror("Error", "New folder name also exists! Please try again.")
                    return
                else:
                    os.makedirs(folder_path)
                    messagebox.showinfo("Success", f"Folder '{new_name}' created successfully!")
            else:
                return
    else:
        os.makedirs(folder_path)
        messagebox.showinfo("Success", f"Folder '{measurement_name}' created successfully!")

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
root.title("Measurement Folder Creator")

# Create and place the widgets
tk.Label(root, text="Measurement Name:").grid(row=0, column=0, padx=10, pady=10)
entry = tk.Entry(root)
entry.grid(row=0, column=1, padx=10, pady=10)

var1 = tk.BooleanVar()
var2 = tk.BooleanVar()
var3 = tk.BooleanVar()

tk.Checkbutton(root, text="Measure F/# from filter", variable=var1).grid(row=1, column=0, sticky=tk.W, padx=10)
tk.Checkbutton(root, text="Compute Aperture Radius", variable=var2).grid(row=1, column=1, sticky=tk.W, padx=10)
tk.Checkbutton(root, text="Option 3", variable=var3).grid(row=1, column=2, sticky=tk.W, padx=10)

tk.Button(root, text="Run Experiment", command=run_experiment).grid(row=2, column=0, columnspan=3, pady=20)

# Run the application
root.mainloop()