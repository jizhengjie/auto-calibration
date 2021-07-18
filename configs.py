import cv2 as cv
import numpy as np

# Global
ROOT = '/Users/ruoji/Documents/SJTU/VE450/auto-calibration'

# Parameters for image processing
IMAGE_SIZE = (540, 960, 3)

# Parameters for camera calibration
ASPECT_RATIO = 1

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,   # How many pts. to locate
                       qualityLevel = 0.01,  # b/w 0 & 1, min. quality below which everyone is rejected
                       minDistance = 10,   # Min eucledian distance b/w corners detected
                       blockSize = 3 ) # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),  # size of the search window at each pyramid level
                  maxLevel = 2,   #  0, pyramids are not used (single level), if set to 1, two levels are used, and so on
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

