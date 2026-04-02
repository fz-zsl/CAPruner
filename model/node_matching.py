import os
import pandas as pd
from abc import ABC, abstractmethod
import math


class NodeVal(ABC):
    @abstractmethod
    def categ_dist(self, categ: str, utter: str) -> float:
        """
        Calculate the distance between two categorical variables.

        Parameters:
        categ (str): The category of the object.
        utter (str): The natural-language cue for spatial reasoning.

        Returns:
        float: the likelihood of the object is mentioned in the utterance
            according to the object's category.
        """
        pass


class HardCategNodeVal(NodeVal):
    def categ_dist(self, categ: str, utter: str) -> float:
        if categ.lower() in utter.lower():
            return 1.0
        else:
            return 0.0


class SoftCategNodeVal(NodeVal):
    def __init__(self, data_root: str = "data") -> None:
        df = pd.read_csv(os.path.join(data_root, "scannet_metadata", "scannetv2-labels.combined.tsv"), sep='\t', header=None, skiprows=1)
        self.sim_dict = {}
        for raw_categ, categ, nyu in zip(df[1], df[2], df[6]):
            if type(nyu) == float and math.isnan(nyu):
                nyu = categ
            raw_categ = raw_categ.lower()
            categ = categ.lower()
            nyu = nyu.lower()
            if nyu not in self.sim_dict.keys():
                self.sim_dict[nyu] = set([nyu])
            self.sim_dict[nyu].add(categ)
            self.sim_dict[nyu].add(raw_categ)
    

    def categ_dist(self, categ: str, utter: str) -> float:
        categ = categ.lower()
        if 'object' in categ:
            return 0.0
        utter = utter.lower()
        categ_set = set()
        for nyu in self.sim_dict.keys():
            if categ in self.sim_dict[nyu]:
                categ_set.update(self.sim_dict[nyu])
        assert categ_set, f"Category {categ} not found in similarity dictionary."
        for _categ in categ_set:
            if _categ in utter:
                return 1.0
        return 0.0


class NoNodeVal(NodeVal):
    def categ_dist(self, categ: str, utter: str) -> float:
        return 0.0
