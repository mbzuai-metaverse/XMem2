import cv2
import numpy as np
import re
import os
import sys
import csv
import ast
import argparse

matte_element_to_color = {}


def populate_element_to_color_dict(csv_file_path):
    """
    From the csv file mapping XMem2 matte colors to shot elements, create a dictionary with the elements as keys.
    Note the BGR colors for OpenCV.
    """
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r', newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                element = row['Element'].lower().replace(" ", "")
                matte_element_to_color[element] = tuple(map(int, row['BGRColor'].strip('()').split(', ')))
    else:
        print(f"{csv_file_path} does not exist. Please supply a valid csv file to map matte colors to elements.")
        sys.exit(1)


def get_files_in_path(colormatte_path):
    """Return the list of files in the provided path."""
    file_names = os.listdir(colormatte_path)

    return file_names


def get_frame_number(filename):
    """Get the frame number from the path to the matte."""

    # regex may be more concise but less readable.
    # Find the position of the last dot
    last_dot_index = filename.rfind('.')

    # Find the position of the second-to-last dot
    second_last_dot_index = filename.rfind('.', 0, last_dot_index)

    # Extract the substring between the last two dots
    if last_dot_index != -1 and second_last_dot_index != -1:
        frame = filename[second_last_dot_index + 1 : last_dot_index]
        return frame
    else:
        print("Could not extract frame number.")


def get_frame_range(folder_path):
    """Get the min and max frame numbers for the files in the given path."""

    frame_numbers = []

    # looks for a sequence of digits (\d+) followed by ".png" and ensures that it is a whole word
    # using word boundaries (\b)
    pattern = re.compile(r'\b(\d+)\.png\b')

    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            frame_numbers.append(int(match.group(1)))

    return min(frame_numbers), max(frame_numbers)


def create_black_matte_filename(existing_file, frame_number):
    """Create a filename (no extension) for a black matte."""
    prefix, frame, suffix = existing_file.split('.')
    frame = frame_number
    new_file = '.'.join([prefix, str(frame).zfill(8)])
    new_file = new_file.replace("colormatte", "matte")
    return new_file    


def get_image_width_height(image):
    """Given an image, obtain it's width and height."""
    height, width, channels = image.shape
    return width, height


def get_matte_path(colormatte_path):
    """Get the path for b/w mattes. Input path must contain 'colormatte'."""
    if 'colormatte' in colormatte_path:
        matte_path = colormatte_path.replace("colormatte", "mattes")
        return matte_path
    else:
        print("The provided colormattes must have a 'colormatte' folder in their path.")
        sys.exit(1)


def get_denoise_path(colormatte_path):
    """Get the path for denoise frames. Input path must contain 'colormatte'."""
    if 'colormatte' in colormatte_path:
        denoise_path = colormatte_path.replace("colormatte", "denoise")

        # Always use v001 for the denoise path. The number of frames will be the same in all versions.
        denoise_index = denoise_path.find('denoise\\')
        prefix = denoise_path[:denoise_index + len('denoise\\')]
        
        # Construct the new path with 'v001'
        denoise_path = os.path.join(prefix, 'v001')
        return denoise_path
    else:
        print("The provided colormattes must have a 'colormatte' folder in their path.")
        sys.exit(1)


def get_matte_filename(image_path):
    """Get the matte filename from the input path"""
    filename = os.path.basename(image_path)

    prefix, frame, suffix = filename.split('.')
    filename_without_extension = '.'.join([prefix, str(frame).zfill(8)]).replace("colormatte", "matte")

    return filename_without_extension


def get_matte_colors(image):
    """Get the list of colors that were used for mattes in the given image."""
    flattened_image = image.reshape((-1, 3))

    unique_colors = np.unique(flattened_image, axis=0)
    unique_colors_list = [tuple(color) for color in unique_colors]
    black = (0, 0, 0)
    unique_colors_list.remove(black)
    return unique_colors_list


def get_element_bw_matte_from_colormatte(image, target_color):
    """Create a black and white matte from the area matted by the target color."""
    # Convert the target color to a NumPy array
    target_color_np = np.array(target_color, dtype=np.uint8)

    # Create a b/w matte for the exact color using an equality check
    matte = np.all(image == target_color_np, axis=-1).astype(np.uint8) * 255

    return matte


def create_black_matte(width, height):
    """Create a solid black matte. Used for missing frames or missing element in a frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def save_bw_matte(matte, element_matte_path, matte_filename):
    """Save the black and white matte"""
    os.makedirs(element_matte_path, exist_ok=True)

    try:
        cv2.imwrite(rf'{element_matte_path}\{matte_filename}.png', matte)
        print(rf'Saved image {element_matte_path}\{matte_filename}.png')
    except Exception as e:
        print(f"Could not save file: {e}")


def create_frame_dict(files):
    """Create a lookup by frame number for a set of files."""
    frame_dict = {}
    for file in files:
        frame_number = get_frame_number(file)
        frame_dict[int(frame_number)] = file
    return frame_dict


parser = argparse.ArgumentParser(description='Create black and white mattes for elements in color mattes.')

# Required parameters
parser.add_argument('map_colors_path', type=str, help='Path to csv file that maps matte colors to elements')
parser.add_argument('colormatte_path', help='Path to the colormatte files from which to extract black and white mattes')
parser.add_argument('elements', type=str, help='Comma separated (no spaces, no brackets) list of elements that need'
                                               ' black and white mattes. Example: element1,element2')

# Optional parameters
parser.add_argument('-sf', '--start_frame', type=int, help='Start processing at this frame')
parser.add_argument('-ef', '--end_frame', type=int, help='End processing at this frame')
parser.add_argument('-incr', '--increment', type=int, default=1, help='Frame increment')

args = parser.parse_args()

bw_matte_elements = args.elements.split(',')


# Get the default frame range from the denoise path
denoise_path = get_denoise_path(args.colormatte_path)

if args.start_frame is None or args.end_frame is None:
    denoise_start_frame, denoise_end_frame = get_frame_range(denoise_path)
    print('denoise_start_frame', denoise_start_frame)
    print('denoise_end_frame', denoise_end_frame)

if args.start_frame is None:
    matte_start_frame = denoise_start_frame
else:
    matte_start_frame = args.start_frame


if args.end_frame is None:
    matte_end_frame = denoise_end_frame
else:
    matte_end_frame = args.end_frame


print('map_colors_path', args.map_colors_path)
print('colormatte_path', args.colormatte_path)
print('elements', bw_matte_elements)
print('start_frame', matte_start_frame)
print('end_frame', matte_end_frame)
print('increment', args.increment)

matte_path = get_matte_path(args.colormatte_path)

# Use the csv file to create a matte color lookup by element
populate_element_to_color_dict(args.map_colors_path)


invalid_elements = []
for element in bw_matte_elements:
    if element not in matte_element_to_color:
        invalid_elements.append(element)
        if invalid_elements:
            print(f"The following elements were requested and are invalid shot elements: {invalid_elements}")
            sys.exit(1)


colormatte_files = get_files_in_path(args.colormatte_path)

image_path = os.path.join(args.colormatte_path, colormatte_files[0])
image = cv2.imread(image_path)

IMAGE_WIDTH, IMAGE_HEIGHT = get_image_width_height(image)
print('IMAGE_WIDTH', IMAGE_WIDTH)
print('IMAGE_HEIGHT', IMAGE_HEIGHT)

colormatte_lookup = create_frame_dict(colormatte_files)

for frame in range(matte_start_frame, matte_end_frame + 1, args.increment):
    # if colormatte_lookup[frame]:
    if frame in colormatte_lookup:
        # image_path = os.path.join(args.colormatte_path, file)
        image_path = os.path.join(args.colormatte_path, colormatte_lookup[frame])
        matte_filename = get_matte_filename(image_path)
        
        # Load the colormatte image
        image = cv2.imread(image_path)

        matte_colors = get_matte_colors(image)    

        for element in bw_matte_elements:
            element_matte_path = rf'{matte_path}\{element}'
            element_color = matte_element_to_color[element]
            # if that color is in the colormatte for that frame, create a b/w matte matte for the element
            if element_color in matte_colors:
                matte = get_element_bw_matte_from_colormatte(image, element_color)
            else:
                matte = create_black_matte(IMAGE_WIDTH, IMAGE_HEIGHT)
            
            save_bw_matte(matte, element_matte_path, matte_filename)

    else:
        # frame is missing - create black mattes for all elements
        for element in bw_matte_elements:
            # create an output file path for that frame, based on the input file paths
            black_matte_filename = create_black_matte_filename(colormatte_files[0], frame)
            # create a black matte using the image width and height of the input images
            black_matte = create_black_matte(IMAGE_WIDTH, IMAGE_HEIGHT)
            # write the black matte
            element_matte_path = rf'{matte_path}\{element}'
            
            save_bw_matte(black_matte, element_matte_path, black_matte_filename)

print(f"Generation of mattes for frames {matte_start_frame} through {matte_end_frame} with increment {args.increment}"
      f" is complete.")
