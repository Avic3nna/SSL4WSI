import numpy as np
import PIL
from pathlib import Path
from typing import Dict
import glob
from sklearn.model_selection import train_test_split


def load_tile_sets(tile_path: Path = None):
    tile_dirs = glob.glob(tile_path)
    train_set, valid_set = train_test_split(tile_dirs, test_size=0.2, random_state=1337)
    data_dict = {"training": [], "validation": []}
    
    # add the training files to the training dictionary
    for filename in train_set:
        data_dict["training"].append({"image": filename})

    # add the validation files to the validation dictionary
    for filename in valid_set:
        data_dict["validation"].append({"image": filename})

    return data_dict["training"], data_dict["validation"]


# def process_slide_jpg(slide_jpg: PIL.Image):
#     img_norm_wsi_jpg = PIL.Image.open(slide_jpg)
#     image_array = np.array(img_norm_wsi_jpg)
#     canny_norm_patch_list = []
#     coords_list=[]
#     total=0
#     patch_saved=0
#     for i in range(0, image_array.shape[0]-224, 224):
#         for j in range(0, image_array.shape[1]-224, 224):
#             total+=1
#             patch = image_array[j:j+224, i:i+224, :]
#             if not np.all(patch):
#                 canny_norm_patch_list.append(patch)
#                 coords_list.append((j,i))
#                 patch_saved+=1
#     return canny_norm_patch_list, coords_list, patch_saved, total