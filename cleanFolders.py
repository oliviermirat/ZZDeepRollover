import os
import glob
import shutil

def removeIpynbCheckpointsFromTrainingDataset():
  if (os.path.isdir('trainingDataset/.ipynb_checkpoints')):
    shutil.rmtree('trainingDataset/.ipynb_checkpoints')
  if (os.path.isdir('trainingDataset/train/.ipynb_checkpoints')):
    shutil.rmtree('trainingDataset/train/.ipynb_checkpoints')
  if (os.path.isdir('trainingDataset/val/.ipynb_checkpoints')):
    shutil.rmtree('trainingDataset/val/.ipynb_checkpoints')

def refreshTrainingDataset():
  if (os.path.isdir('trainingDataset')):
    shutil.rmtree('trainingDataset')
  os.mkdir('trainingDataset')
  os.mkdir('trainingDataset/train')
  os.mkdir('trainingDataset/train/normal')
  os.mkdir('trainingDataset/train/rollover')
  os.mkdir('trainingDataset/val')
  os.mkdir('trainingDataset/val/normal')
  os.mkdir('trainingDataset/val/rollover')
 

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
  
  if not(os.path.isdir('model')):
    os.mkdir('model')
  
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
