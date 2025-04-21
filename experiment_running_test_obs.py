import time

import core.file_management

def main(folder_path:str):
    with open(folder_path + "\progress.txt", "a") as f:
        f.write("test\n")
    for i in range(10):
        time.sleep(0.9)
        core.file_management.write_progress(f"Step {i} completed")
    core.file_management.progress_file_remove()
