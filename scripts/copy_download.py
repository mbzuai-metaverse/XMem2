import os
import zipfile

print('Extracting YouTube2018 datasets...')
with zipfile.ZipFile('../Datasets/YouTube2018/valid.zip', 'r') as zip_file:
    zip_file.extractall('../Datasets/YouTube2018/')
with zipfile.ZipFile('../Datasets/YouTube2018/all_frames/valid.zip', 'r') as zip_file:
    zip_file.extractall('../Datasets/YouTube2018/all_frames')
