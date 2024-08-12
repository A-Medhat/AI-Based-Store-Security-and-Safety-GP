import os
import shutil

import os
import shutil
import re
def extract_even_frames(input_folder_path, output_folder_path):

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    files = os.listdir(input_folder_path)

    image_files = [file for file in files if file.endswith('.jpg') or file.endswith('.png')]

    image_files.sort(key=lambda x: int(re.findall(r'\d+', os.path.splitext(x)[0])[-1]))

    even_frames = [f for i, f in enumerate(image_files) if (i + 1) % 2 == 0]

    for frame in even_frames:
        shutil.copyfile(os.path.join(input_folder_path, frame), os.path.join(output_folder_path, frame))


def process_folders(main_folder_path):
    folders = [folder for folder in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, folder))]

    for folder in folders:
        input_folder_path = os.path.join(main_folder_path, folder)
        output_folder_path = os.path.join(main_folder_path, folder + "_even_frames")
        extract_even_frames(input_folder_path, output_folder_path)



main_folder = "composite/Shoplifting"

process_folders(main_folder)