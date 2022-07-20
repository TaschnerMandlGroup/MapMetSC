from typing import List
import pandas as pd

from sample import Sample

class Patient:
    def __init__(self, samples: List[Sample], patient_ID: str = None, stage: str = None, efs: bool = None, tod: bool = None, efs_dur: float = None, tod_dur: float = None):
        self.samples = samples
        self.patient_ID = patient_ID
        self.stage = stage
        self.efs = efs
        self.tod = tod
        self.efs_dur = efs_dur
        self.tod_dur = tod_dur

    def sample(self, sample_ID: str) -> Sample:
        samples = [s for s in self.sample if s.sample_ID == sample_ID]
        if not samples:
            raise ValueError('No sample with the given sample ID found! The patient has the following samples: ' + str([s.sample_ID for s in self.samples]))
        return samples[0]

    def get_sc_data(self, add_metadata: bool = False) -> pd.DataFrame:
        df_features = pd.concat([s.get_sc_data(add_metadata=add_metadata) for s in self.samples])
        if add_metadata:     
            df_features['patient_ID'], df_features['stage'], df_features['efs'], \
            df_features['tod'], df_features['efs_dur'], df_features['tod_dur'], \
                 = [self.patient_ID, self.stage, self.efs, self.tod, self.efs_dur, self.tod_dur]

        return df_features