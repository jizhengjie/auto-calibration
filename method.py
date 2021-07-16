import os, sys, logging, argparse
import numpy as np

from configs import *
from utils import *


class FullyAutomaticCalibration():
    # Should take about 200s to get good performance
    def __init__(self, data, **kwargs):
        super().__init__()
        self.data = data # 664 imgs, size (540, 960, 3)
        
    def run(self):
        # TODO: First VP extraction

        # TODO: Second VP

        # TODO: Third VP, Principal Point, Focal Length

        # TODO: Radial Distortion

        # TODO: Camera Calibration from VPs

        # Return R and T as dict
        # example values are:
        R = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        T = [0.0, 0.0, 0.0]
        return {'R': R, 'T': T}


if __name__ == "__main__":
    pass

