import os, sys, logging, argparse
import numpy as np
import cv2 as cv

from skimage.io import imread
from skimage.filters import sobel
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pclines import PCLines
from pclines import utils
from collections import Counter

from configs import *
from utils import *


class FullyAutomaticCalibration():
    # Should take about 200s to get good performance
    def __init__(self, data, **kwargs):
        super().__init__()
        self.data = data # 664 imgs, size (540, 960, 3)

    def extend_line(self, x1, y1, x2, y2, x, y):
        # x = 959, y = 539
        k = (y2 - y1) / (x2 - x1)
        b = (x1 * y2 - x2 * y1) / (x1 - x2)
        if x1 == x2 or y1 == y2: 
            return None, None
        t = []
        res = int(b)
        if res >= 0 and res <= y:
            t.append((0, res))
        res = int(-b/k)
        if res >= 0 and res <= x:
            t.append((res, 0))
        res = int(k*x+b)
        if res >= 0 and res <= y:
            t.append((x, res))
        res = int((y-b)/k)
        if res >= 0 and res <= x:
            t.append((res, y))

        if len(t) != 2:
            return None, None
        else:
            return t[0], t[1]

    def euclidean_distance(self, x, y):
        return np.sqrt((x[0] - x[1])**2 + (y[0] - y[1])**2)

    def KLT(self):
        # Parameters
        video_path = os.getcwd()
        output_file = 'KLT.mp4'
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        frame = self.data[0]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        edges = None
        img_idx = 1
        start = True
        line_list = []
        points = []
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
                # Accumulative mask
                edges = np.zeros_like(frame)
                edges = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
                # Draw the tracks
                for i, (new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel() #tmp new value
                    c,d = old.ravel() #tmp old value
                    line_list.append((a,b))
                    mask = cv.line(mask, (a,b),(c,d), (0,255,0), 1)
                    frame = cv.circle(frame,(a,b),2,(0,0,255), -1)
                    dist = self.euclidean_distance((a,b), (c,d))
                    t0, t1 = self.extend_line(a,b,c,d,IMAGE_SIZE[1]-1,IMAGE_SIZE[0]-1)
                    # accumulate extended lines, filter out stable points
                    if dist > 300 and t0 != None:
                        edges = cv.line(edges, t0,t1, 1, 1)
                # Map to diamond space
                if img_idx == self.data.shape[0]-1:
                    show_image = True
                else:
                    show_image = False
                points += self.map_to_diamond_space(frame, edges, show_image)
                # Write to video
                img = cv.add(frame,mask)
                out.write(img)
                # Now update previous points
                p0 = good_new.reshape(-1,1,2)
            img_idx += 1
        # Accumulate points
        c = Counter(points)
        global_maximum = c.most_common(3)
        print("Candidate first VPs and votes:", global_maximum)
        # Visualize
        r,c = np.nonzero(edges > 0.5)
        x = np.array([c,r],"i").T
        weights = edges[r,c]
        h,w = frame.shape[:2]
        bbox=(0,0,w,h)
        d = 1024
        # Create new accumulator
        P = PCLines(bbox, d)
        # Insert observations
        P.insert(x, weights)
        # Find local maxima
        f,ax = plt.subplots(1, figsize=(10,5))
        for i in range(1):
            ax.plot(global_maximum[i][0][1], global_maximum[i][0][0], "r+")
        ax.imshow(np.sqrt(P.A), cmap="Greys")
        ax.set(title="Accumulator space",xticks=[],yticks=[])
        plt.tight_layout()
        plt.show()
        print("Accumulated first VP in diamond space is: ", global_maximum[0][0])
        return global_maximum[0][0]

    def map_to_diamond_space(self, frame, edges, show_image=False):
        # Map to dimond space, return points
        r,c = np.nonzero(edges > 0.5)
        x = np.array([c,r],"i").T
        weights = edges[r,c]
        if show_image:
            _,ax = plt.subplots(1, figsize=(5,5))
            ax.imshow(edges, cmap="Greys")
            ax.set(title="Edge map - observations", xticks=[], yticks=[])
            plt.tight_layout()
        h,w = frame.shape[:2]
        bbox=(0,0,w,h)
        d = 1024
        # Create new accumulator
        P = PCLines(bbox, d)
        # Insert observations
        P.insert(x, weights)
        # Find local maxima
        p, w = P.find_peaks(min_dist=0, prominence=2.5, t=0.6, prominence_radius=1)
        if show_image:
            f,ax = plt.subplots(1, figsize=(10,5))
            ax.plot(p[:,1], p[:,0], "r+")
            ax.imshow(np.sqrt(P.A), cmap="Greys")
            ax.set(title="Accumulator space",xticks=[],yticks=[])
            plt.tight_layout()
        h = P.inverse(p)
        X,Y = utils.line_segments_from_homogeneous(h, bbox)
        if show_image:
            f,ax = plt.subplots(figsize=(5,5))
            ax.imshow(frame, cmap="gray")
        for x,y in zip(X,Y):
            if x is None or y is None:
                continue
            l = Line2D(x,y, color="r")
            if show_image:
                ax.add_artist(l)
        if show_image:
            ax.set(title="Image with detected lines", xticks=[], yticks=[])
            plt.tight_layout()
            plt.show()
        res = []
        for i in range(p.shape[0]):
            res.append((p[i][0], p[i][1]))
        return res

    def run(self):
        # TODO: First VP extraction
        first_VP = self.KLT()

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

