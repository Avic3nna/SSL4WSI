import numpy as np
import PIL


def process_slide_jpg(slide_jpg: PIL.Image):
    img_norm_wsi_jpg = PIL.Image.open(slide_jpg)
    image_array = np.array(img_norm_wsi_jpg)
    canny_norm_patch_list = []
    coords_list=[]
    total=0
    patch_saved=0
    for i in range(0, image_array.shape[0]-224, 224):
        for j in range(0, image_array.shape[1]-224, 224):
            total+=1
            patch = image_array[j:j+224, i:i+224, :]
            if not np.all(patch):
                canny_norm_patch_list.append(patch)
                coords_list.append((j,i))
                patch_saved+=1
    return canny_norm_patch_list, coords_list, patch_saved, total