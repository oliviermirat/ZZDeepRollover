import os
import glob

def cleanDataset(folder_path, file_extension):

  # Create a file path pattern to match files with the given extension
  file_pattern = os.path.join(folder_path, f'*.{file_extension}')

  # Use glob to find all files matching the pattern
  files_to_remove = glob.glob(file_pattern)

  # Remove each file
  for file_path in files_to_remove:
    try:
      os.remove(file_path)
    except OSError as e:
      print(f"Error while deleting file: {file_path}\n{e}")



def cleanModel(folder_path):
  
  # List all files and directories in the given folder
  items = os.listdir(folder_path)
  
  # Iterate through each item
  for item in items:
    item_path = os.path.join(folder_path, item)

    if os.path.isfile(item_path):
      # If the item is a file, remove it
      try:
        os.remove(item_path)
      except OSError as e:
        print(f"Error while deleting file: {item_path}\n{e}")
    elif os.path.isdir(item_path):
      # If the item is a directory, recursively call the function
      remove_all_files_in_folder(item_path)
