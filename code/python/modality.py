from typing import List

class Modality:
    modalities = ['IF', 'IMC', 'FISH']
    marker_compartments = ['membrane', 'nucleus', 'cytoplasm']

    def __init__(self, name: str, channel_names: List[str], resolution: float, compartments: List[str]):
        self.channel_names = channel_names
        self.resolution=resolution
        
        self._name=name
        self._compartments=compartments
        
    @property
    def name(self) -> str:
        return self._name

    @property
    def compartments(self) -> List[str]:
        return self._compartments

    @name.setter
    def name(self, value: str):
        if value not in self.modalities:
            raise ValueError('Modality name must be one of the following: ' + str(self.modalities))

        self._name = value

    @compartments.setter
    def compartments(self, value: List[str]):
        if value not in self.marker_compartments:
            raise ValueError('compartments must be one of the following: ' + str(self.marker_compartments))
        if len(value) != len(self.channel_names):
            raise ValueError('compartments must be of same length as channel_names')

        self._compartments = value

        