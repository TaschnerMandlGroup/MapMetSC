from typing import List, Tuple
import warnings
import numpy as np
import pandas as pd
#from segmentation import Segmentation
#from registration import Registration
#from feature_extraction import FeatureExtractor

from acquisition import Acquisition
from modality import Modality

class ROI:

    def __init__(self, roi_number: int, acquisitions: List[Acquisiton]):
        self.roi_number = roi_number
        self.acquisitons = acquisitions

        self.transform = None
        self.image_stack = None
        self.channel_stack = None
        self.nuclear_mask = None
        self.cell_mask = None
        
    def acquisition(self, modality: str) -> Acquisition:
        acq = [a for a in self.acquisitions if a.mod.name == modality]
        if not acq:
            raise ValueError('No acquistion of the given modality was found. Modality has to be one of the following: ' + str(Modality.modalities))
        if len(acq) > 1:
            raise ValueError('More than one acquisition of that modality was found.')
        return acq[0]

    def register_acquisitions(self):
        self.transform, self.image_stack, self.channel_stack = None # use register method from class Registration here

    def segment(self, model, compartment, modality):
        self.nuclear_mask = None #use segmentation methods from class Segmentation here
        self.cell_mask = None 

    def get_sc_data(self) -> pd.DataFrame:
        return None #use extract feature methods from class FeatureExtractor here and add roi-args (roi_number, but also machine, acquisition_date)


        