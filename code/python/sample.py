from typing import List
import datetime
import pandas as pd

from roi import ROI

class Sample:
    origin_types = ['PT', 'M']
    timepoints = ['DX', 'RE1', 'RE2', 'REL']

    def __init__(self, rois: List[ROI], sample_ID: str = None, origin: str = None, sampling_date: datetime.date = None, timepoint: str = None):
        self.rois = rois
        self.sample_ID = sample_ID
            
        self._origin = origin
        self._sampling_date = sampling_date
        self._timepoint = timepoint

    def roi(self, roi_number: str) -> ROI:
        rois = [r for r in self.rois if r.roi_number == roi_number]
        if not rois:
            raise ValueError('No RoI with the given number found! The sample has the following RoI: ' + str([r.roi_number for r in self.rois]))
        return rois[0]

    def get_sc_data(self, add_metadata: bool = False) -> pd.DataFrame:
        df_features = pd.concat([r.get_sc_data() for r in self.rois])
        if add_metadata:
            df_features['sample_ID'], df_features['origin'], df_features['sampling_date'], \
            df_features['timepoint'] = [self.sample_ID, self._origin, self._sampling_date, self._timepoint]

        return df_features

    def get_roi_summary(self) -> pd.DataFrame: #make more generizable to different modalities, not only IMC 
        df_roi_summary = []
        for r in self.rois:
            df_roi_summary.append(
                {
                    'roi_number': r.roi_number, 
                    'machine': r.acquisition('IMC').machine,
                    'acquisiton_date': r.acquisition('IMC').acquisiton_date
                }
                )

        return pd.DataFrame(df_roi_summary)


    @property
    def origin(self) -> str:
        return self._origin

    @property
    def sampling_date(self) -> datetime.datetime:
        return self._sampling_date

    @property
    def timepoint(self) -> str:
        return self._timepoint

    @origin.setter
    def origin(self, value):
        if value != None:
            if value not in self.origin_types:
                raise ValueError('origin must be one of the following: ' + str(self.origin_types))
        
        self._origin = value

    @sampling_date.setter
    def sampling_date(self, value):
        if value != None:
            if not isinstance (value, datetime.date):
                raise TypeError('sampling_date must be of type datetime.date (e.g. datetime.datetime(2018, 8, 31)).')

        self._sampling_date = value

    @timepoint.setter
    def timepoint(self, value):
        if value != None:
            if value not in self.timepoints:
                raise ValueError('timepoint must be one of the following: ' + str(self.timepoints))

        self._timepoint = value