from typing import List
from operator import itemgetter as ig
import datetime
import numpy as np

from modality import Modality

class Aquisiton:
    machines = ['Zoidberg', 'Tiger Lilly', 'Eowyn']

    def __init__(self, mod: Modality, imgs: List[np.ndarray], acquisition_date: datetime.date = None, machine: str = None):
        self.mod = mod
        self.imgs = imgs
        
        self._acquisition_date = acquisition_date
        self._machine = machine

    def get_mean_compartment_image(self, compartment: str) -> np.ndarray:
        if compartment not in Modality.marker_compartments:
            raise ValueError('compartment must be one of the following: ' + str(Modality.marker_compartments))
        
        idx = [i for i in range(len(self.mod.compartments)) if self.mod.compartments[i]==compartment]
        mean_img = np.mean(np.asarray(ig(*idx)(self.imgs)), axis=0)

        return mean_img

    def get_mean_image(self, channel_names: List[str]) -> np.ndarray:
        if not set(channel_names).issubset(self.mod.channel_names):
            raise ValueError('channel_names must be a subset of: ' + str(self.mod.channel_names))
        
        idx = [i for i in range(len(self.mod.channel_names)) if self.mod.channel_names[i] in channel_names]
        mean_img = np.mean(np.asarray(ig(*idx)(self.imgs)), axis=0)

        return mean_img

    @property
    def acquisition_date(self) -> datetime.datetime:
        return self._acquisition_date

    @property
    def machine(self) -> str:
        return self._machine

    @acquisition_date.setter
    def acquisition_date(self, value: datetime.date):
        if value != None:
            if not isinstance (value, datetime.date):
                raise TypeError('acquisition_date must be of type datetime.date (e.g. datetime.datetime(2018, 8, 31)).')

        self._acquisition_date = value

    @machine.setter
    def machine(self, value: str):
        if value != None:
            if value not in self.machines:
                raise ValueError('machine must be one of the following: ' + str(self.machines))

        self._machine = value



        