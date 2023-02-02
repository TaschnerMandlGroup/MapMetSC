import cellpose
import numpy as np
import cellpose.models
import torch
import cv2
from skimage.measure import regionprops_table
from skimage.morphology import dilation, square
from typing import Union
from .intensity_functions import mean_intensity, mean_80_intensity
from Registration._02_register import FeatureExtractor
from Registration._01_preprocess import preprocess
from sklearn.preprocessing import MinMaxScaler
import logging
import pandas as pd
import types

PROPERTIES = ["label", "area", "area_convex", "area_filled", "axis_major_length", "axis_minor_length", "eccentricity", "equivalent_diameter_area", "perimeter", "solidity", "centroid"]

def main_function(I: 'np.ndarray[np.uint8]', 
                  intensity_image: 'np.ndarray[np.float32]'=None,
                  cellpose_net: Union['str', cellpose.models.CellposeModel] = None, 
                  eval_kwargs:dict=None, 
                  refine: bool=False, 
                  t: float=0.12, 
                  out_sz: tuple=None,
                  extract_morph_features: bool=False, 
                  extract_intensity_features: bool=False,
                  channel_names=None,
                  #intensity_function=mean_intensity,
                  additional_morphology_functions=[],
                  debug_msg=True
                  ): 
    
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    logger = logging.getLogger(__name__)

    if debug_msg:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
    
    # get image I sz = (X,Y), dtype = uint8
    check_dtype(I, np.uint8)
    
    # get image I sz = (X,Y), dtype = uint8
    if not isinstance(intensity_image, type(None)):
        check_dtype(intensity_image, np.float32)
    
    #check types of model, either provide a given model or give the name of the model u want to use
    if isinstance(cellpose_net, str):
        model = cellpose.models.CellposeModel(model_type=cellpose_net, gpu=torch.cuda.is_available())

    elif isinstance(cellpose_net, cellpose.models.CellposeModel):
        model = cellpose_net
    else:
        raise ValueError("Enter either a string or a Cellpose Model for cellpose_net")
    
    # segment the given image I using a network from cellpose, give cellpose the network u want to use as a parameter cellpose_net = "CPx"
    logger.debug(f"Start segmentation of image with size: {I.shape}")

    masks, _, _ = model.eval(I, **eval_kwargs)

    logger.debug(f"Finished segmentation found {len(np.unique(masks))} masks")

    # give a flag if you want to correct the masks or not refine = False / True
    if refine:
        
        logger.debug(f"Refining segmentation with t={t}")
        # give it a threshold that is going to be used for thresholding t = 0.12
        masks = refine_masks(I, masks, t=t)
        
    morph_features = {}
    if extract_morph_features:
        
        logger.debug(f"Extracting Morphology with additional functions: {additional_morphology_functions}")
        morph_features.update(extract_morphology(masks, additional_morphology_functions=additional_morphology_functions))
        morph_features = pd.DataFrame(morph_features)
        morph_features = morph_features.rename(columns={"label": "Object"})

        morph_features = morph_features.set_index("Object")

        logger.debug(f"Done Extracting Morphology")

    if not isinstance(intensity_image, type(None)):
        
        out_sz = intensity_image.shape[-2:]

    if not isinstance(out_sz, type(None)):
        if out_sz != I.shape:
            
            logger.debug(f"Reshaping input image and masks to size: {out_sz}")
            #Write a method that estimate the affine transformation of I onto intenstiy image[-3] and then uses h to warp the mask - I does not need to be warped!!
            pp_I = preprocess(I)
            pp_intensity_image = preprocess(intensity_image[-3])

            ex = FeatureExtractor("sift")
            ex(pp_intensity_image, pp_I)
            ex.match()
            ex.estimate()

            _, masks = ex.warp(im0=intensity_image[-3], im1=masks, discrete=True)
            #masks = cv2.resize(masks, out_sz, interpolation=cv2.INTER_NEAREST)
            #I = cv2.resize(I, out_sz, interpolation=cv2.INTER_LINEAR)

            masks_dilated_1px = dilation(masks, square(3))
            masks_dilated_2px = dilation(masks, square(5))
  
    # give the function a parameter if you want to extract features (regionprops) and a list of the features you want to extract
    if extract_intensity_features and not isinstance(intensity_image, type(None)):
        
        #logger.debug(f"Extracting Intensity for intensity image with functions: {intensity_function}")
        logger.debug(f"Extracting Intensity")
        mean = extract_intensity(intensity_image, masks, intensity_function=mean_intensity, channel_names=channel_names)
        mean_80 = extract_intensity(intensity_image, masks, intensity_function=mean_80_intensity, channel_names=channel_names)
        intensity_features = pd.concat([mean, mean_80], axis=1)
        #logger.debug(f"Done Extracting Intensity")

        #logger.debug(f"Extracting Intensity for intensity image with expanded masks with functions: {intensity_function}")
        mean = extract_intensity(intensity_image, masks_dilated_1px, intensity_function=mean_intensity, channel_names=channel_names)
        mean_80 = extract_intensity(intensity_image, masks_dilated_1px, intensity_function=mean_80_intensity, channel_names=channel_names)
        intensity_features_dilated_1px = pd.concat([mean, mean_80], axis=1)
        #logger.debug(f"Done Extracting Intensity")

        #logger.debug(f"Extracting Intensity for intensity image with expanded masks with functions: {intensity_function}")
        mean = extract_intensity(intensity_image, masks_dilated_2px, intensity_function=mean_intensity, channel_names=channel_names)
        mean_80 = extract_intensity(intensity_image, masks_dilated_2px, intensity_function=mean_80_intensity, channel_names=channel_names)
        intensity_features_dilated_2px = pd.concat([mean, mean_80], axis=1)
        logger.debug(f"Done Extracting Intensity")
        
    
    
    logger.debug(f"Finished!")
    
    return masks, masks_dilated_1px, masks_dilated_2px, intensity_features, intensity_features_dilated_1px, intensity_features_dilated_2px, morph_features


def extract_morphology(masks, additional_morphology_functions=None):
    
    props = regionprops_table(masks, properties=PROPERTIES, extra_properties=additional_morphology_functions)
    return props
        

def extract_intensity(I, masks, intensity_function=None, channel_names=None):
    
    if check_type(intensity_function, types.FunctionType):
        
        results = {}
            
        results.update(intensity_function(I, masks, channel_names=channel_names))

        results = pd.DataFrame(results)
        results = results.set_index("Object")
            
    return results


def refine_masks(I, masks, t=0.15):
    
    I = I.copy()/255
    binary_masks = masks.astype(bool)
    binary_masks[I < t] = 0
    # blurring to remove rough edges: b_patch0 = cv2.GaussianBlur(patch0, ksize=(5,5), sigmaX=1.5, sigmaY=1.5)
    binary_masks = cv2.GaussianBlur(binary_masks.astype(np.uint8), ksize=(5,5), sigmaX=1.5, sigmaY=1.5).astype(bool)
    refined_masks = binary_masks*masks
    
    return refined_masks
    

def check_type(inp, target_type):
    if type(inp) != target_type:
        raise TypeError(f"Wrong type for input. Expected {target_type} got {type(inp)}")
    else: 
        return 1


def check_dtype(inp, dtype):
    
    if check_type(inp, np.ndarray):
        if inp.dtype != dtype:
            raise TypeError(f"Wrong dtype for input. Expected {dtype} got {inp.dtype}")
        
          

    