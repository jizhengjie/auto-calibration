import os, sys, logging, argparse
import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

from configs import *
from utils import *


class FullyAutomaticCalibration():
    # Should take about 200s to get good performance
    def __init__(self, data, **kwargs):
        super().__init__()
        self.data = data # 664 imgs, size (540, 960, 3)

    def KLT(self):
        video_path = os.getcwd()
        output_file = 'KLT.mp4'
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        frame = self.data[0]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img_idx = 1
        start = True
        while img_idx < self.data.shape[0]:
            # Capture rate
            if img_idx % KLT_CAP_RATE == 0:
                # Update the previous frame
                old_frame = frame.copy()
                old_gray = frame_gray.copy()
                p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                mask = np.zeros_like(old_frame)
                # If start
                if start == True:
                    start = False
                    h, w, _ = frame.shape
                    out = cv.VideoWriter(output_file, fourcc, 30, (w, h), True)
                frame = self.data[img_idx]
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]
                # draw the tracks
                for i, (new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel() #tmp new value
                    c,d = old.ravel() #tmp old value
                    mask = cv.line(mask, (a,b),(c,d), (0,255,0), 1)
                    frame = cv.circle(frame,(a,b),2,(0,0,255), -1)
                img = cv.add(frame,mask)
                out.write(img)
                # Now update previous points
                p0 = good_new.reshape(-1,1,2)
            img_idx += 1

    def run(self):
        # TODO: First VP extraction
        self.KLT()

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

