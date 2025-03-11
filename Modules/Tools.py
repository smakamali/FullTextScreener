import os
import shutil

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def deleteFolderContents(folder_path):
    # Check if the folder exists (it should, due to ensure_folder_exists call)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"The folder {folder_path} has been created.")
    else:
        print(f"The folder {folder_path} already exists.")

        # Iterate over the files and directories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # If the file_path is a file, remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                # If the file_path is a directory, remove it and its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def ensureFolderExists(folder_path):
    """
    Checks if a folder exists at the specified path, and creates it if it does not exist.

    Parameters:
    folder_path (str): The path of the folder to check and create if necessary.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
        