import time
import file_save_managment

def main(folder_path:str):
    with open(folder_path + "\progress.txt", "a") as f:
        f.write("test\n")
    for i in range(10):
        time.sleep(0.9)
        file_save_managment.write_progress(f"Step {i} completed")
    file_save_managment.progress_file_remove()
