import cv2 as cv
import numpy as np
from Registration.utils import additive_blend
import matplotlib.pyplot as plt

class FeatureExtractor():

    def __init__(self, type, *args, **kwargs) -> None:

        self.type = type.upper()

        if self.type == "SIFT":
            self.extractor = cv.SIFT_create(*args, **kwargs)
            self.norm = cv.NORM_L2

        elif self.type == "ORB":
            self.extractor = cv.ORB_create(*args, **kwargs)
            self.norm = cv.NORM_HAMMING

        elif self.type == "AKAZE":
            self.extractor = cv.AKAZE_create(*args, **kwargs)
            self.norm = cv.NORM_HAMMING

        elif self.type == "BRISK":
            self.extractor = cv.BRISK_create(*args, **kwargs)
            self.norm = cv.NORM_HAMMING

        else:
            raise ValueError(f"{type} is not a valid feature detection algorithm. Chose one of 'SIFT, ORB, AKAZE, BRISK'")


    def __call__(self, im0, im1) -> None:
        self.im0 = im0
        self.im1 = im1

        self.kp0, self.des0 = self.extractor.detectAndCompute(im0, None)
        self.kp1, self.des1 = self.extractor.detectAndCompute(im1, None)


    def match(self, ratio_thresh=0.7) -> None:
        matcher = cv.BFMatcher(self.norm) 
        des0 = self.des0[:260000]
        des1 = self.des1[:260000]
        knn_matches = matcher.knnMatch(des0, des1, 2)
        self.matches = [[m] for m, n in knn_matches if m.distance < ratio_thresh*n.distance]
    

    def estimate(self) -> None:
        self.points = np.array([np.array((self.kp0[m[0].queryIdx].pt, self.kp1[m[0].trainIdx].pt)) for m in self.matches]).astype(np.int64)
        print(f"Found {len(self.points)} matches.")
        if self.points.size != 0:
            h, _ = cv.estimateAffinePartial2D(self.points[:, 1], self.points[:, 0])
            self.h = h
        else:
            self.h = None

    def draw_matches(self) -> None:
        matched_image = cv.drawMatchesKnn(self.im0, self.kp0, self.im1, self.kp1, self.matches, None, flags=2)
    
    def compute_overlay(self):
        if hasattr(self, "fixed"):
            if isinstance(self.fixed, np.ndarray):
                return additive_blend(self.fixed, self.warped)

    def warp(self, im0 = None, im1 = None, discrete = False) -> None:
        if isinstance(self.h, np.ndarray):
            if im0 is None:
                im0 = self.im0
                im1 = self.im1
            assert im0.ndim == im1.ndim, "Number of image dimensions does not match."
            dim = im0.ndim
            if dim < 3:
                im0 = np.expand_dims(im0, axis=0)
                im1 = np.expand_dims(im1, axis=0)
            self.fixed=im0

            if discrete:
                #when registering discrete images (e.g. masks)
                self.warped = np.asarray([cv.warpAffine(im1[i], self.h, (self.fixed.shape[2], self.fixed.shape[1]), flags=cv.INTER_NEAREST) for i in range(im1.shape[0])]) 
            else:
                self.warped = np.asarray([cv.warpAffine(im1[i], self.h, (self.fixed.shape[2], self.fixed.shape[1])) for i in range(im1.shape[0])]) 

            if dim < 3:
                self.warped = np.squeeze(self.warped)
                self.fixed = np.squeeze(self.fixed)
        else:
            self.fixed, self.warped = None, None

        return self.fixed, self.warped

    def calc_misalignment(self):
        mpp = 3.14/20
        y_diff_ideal = self.fixed.shape[0]/2 + (self.im0.shape[0]/mpp)/2
        x_diff_ideal = self.fixed.shape[1]/2 - (self.im0.shape[1]/mpp)/2
        
        x_misalign_px = abs(x_diff_ideal) - self.h[0][2]
        y_misalign_px = abs(y_diff_ideal) - self.h[1][2]

        x_misalign_um = abs(x_misalign_px)*mpp
        y_misalign_um = abs(y_misalign_px)*mpp

        return x_misalign_um, y_misalign_um

    def cut_overlap(self):
        dim = self.warped.ndim
        if dim < 3:
            tmp = np.argwhere(self.warped > 0.0)
        else:
            tmp = np.argwhere(self.warped[0] > 0.0)
        
        max_y = tmp[:, 0].max()
        min_y = tmp[:, 0].min()
        min_x = tmp[:, 1].min()
        max_x = tmp[:, 1].max()

        if dim < 3:
            warped_cut = self.warped[min_y:max_y, min_x:max_x]
            fixed_cut = self.fixed[min_y:max_y, min_x:max_x]
        else:
            warped_cut = self.warped[:, min_y:max_y, min_x:max_x]
            fixed_cut = self.fixed[:, min_y:max_y, min_x:max_x]

        return fixed_cut, warped_cut
        

