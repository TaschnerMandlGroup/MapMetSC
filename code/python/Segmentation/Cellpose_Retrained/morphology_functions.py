from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from skimage.measure import moments, label, regionprops_table, perimeter
import numpy as np
from skimage.morphology import binary_erosion

def get_centroid(mask):
    
    M = moments(mask)
    cx, cy = M[1, 0] / M[0, 0],  M[0, 1] / M[0, 0]
    
    return cx, cy


def assymetry(regionmask):
    
    chull = convex_hull_image(regionmask)
    
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].imshow(regionmask)
    # ax[1].imshow(chull)
    # plt.show()
    
    cx_m, cy_m = get_centroid(regionmask)
    cx_h, cy_h = get_centroid(chull)
    
    dist = np.sqrt((cx_m - cx_h)**2 + (cy_m - cy_h)**2)
    
    return dist / regionmask.sum()
    
    
def concavity(regionmask):
    
    chull = convex_hull_image(regionmask)
    
    concavities = binary_erosion((chull.astype(float) - regionmask.astype(float)).astype(bool), footprint=np.ones((3, 3)))
    
    labeled_image = label(concavities)

    num = 0
    for color in np.unique(labeled_image):
         if color != 0:
             if (labeled_image == color).sum() > 10:
                 num += 1
                 
    # fig, ax = plt.subplots(1, 3, figsize=(10,5))
    # ax[0].imshow(regionmask)
    # ax[1].imshow(chull)
    # ax[2].imshow(binary_erosion((chull.astype(float) - regionmask.astype(float)).astype(bool), footprint=np.ones((3, 3))))
    # ax[2].set_title(num)
    # plt.show()

    return num


def fill(regionmask):
    
    chull = convex_hull_image(regionmask)
    return (chull.astype(float) - regionmask.astype(float)).sum() / chull.sum()


def aspect_ratio(regionmask):
    
    props = regionprops_table(regionmask.astype(int), properties=["axis_major_length", "equivalent_diameter_area"])
    
    return props["axis_major_length"] / props["equivalent_diameter_area"] 
    
    
def perimeter_ratio(regionmask):
        
    props = regionprops_table(regionmask.astype(int), properties=["perimeter"])
    
    return (props["perimeter"] ** 2) / regionmask.sum()


DEEPCELL_MEASURES = [assymetry, concavity, fill, aspect_ratio, perimeter_ratio]