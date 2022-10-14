from email import header
import os
import re
from typing import Union
import glob
import natsort
import cv2
import pandas as pd
from readimc import MCDFile
import numpy as np
from utils import _rmlead, remove_zero_pad, filter_hot_pixels
import exifread


def get_rois(f):
    with open(f, 'rb') as fin:
        tags = exifread.process_file(fin)

    roi_num = str(tags["Image Tag 0xB0B7"]).split(":")[-1].split("_")[-1]
    return roi_num

class Sample():

    def __init__(self, sample_path):

        if not os.path.isdir(sample_path):
            raise Exception(f"'{sample_path}' is not a directory")

        self.sample_path = sample_path
        self.sample_name = os.path.split(sample_path)[-1]
        self.__imc_markers = ['Iridium_1033((1254))Ir193', 'Iridium_1033((1253))Ir191', 'H4K12Ac_2023((3829))Er167', 'Histone_1978((3831))Nd146']
        self.__if_markers = ["DAPI", "GD2", "CD56"]  
        try:
            self.mcd_path = glob.glob(sample_path + '/**/*.[mM][cC][dD]')[0]
        except:
            raise Exception(f"No MCD-File found in '{sample_path}'")

        rois_if = self.__get_if_rois()
        rois_imc = self.__get_imc_rois()
        rois = pd.merge(rois_if, rois_imc, on="roi_num")

        self.rois = [self.ROI(x, self.__if_markers) for _, x in rois.iterrows()]
        
        roi_nums = [roi.roi_num for roi in self.rois]
        print(f"Found the following ROIs: {roi_nums}")

    def __get_if_rois(self):

        try:
            roi_tmp = glob.glob(self.sample_path + '/*[sS][pP][oO][tT][sS]*')[0]
        except:
            raise Exception(f"No ROIs found in '{self.sample_path}'")

        files = [os.path.join(roi_tmp,x) for x in os.listdir(roi_tmp)]
        rois = [[f,get_rois(f), f.split(".tif")[0][-1]] for f in files]

        df = pd.DataFrame(rois, columns=["file", "roi_num", "channel"])

        roi_files = []
        for f in list(set(df.roi_num)):
            tmp = df[df.roi_num==f]
            roi_files.append({
                "roi_num": int(f),
                "if_b": tmp.file[tmp.channel.str.lower()=="b"].iloc[0],
                "if_g": tmp.file[tmp.channel.str.lower()=="g"].iloc[0],
                "if_r": tmp.file[tmp.channel.str.lower()=="r"].iloc[0]})

        return pd.DataFrame(roi_files)

    def __get_imc_rois(self):

        roi_images = []
        with MCDFile(self.mcd_path) as f:

            for acq in f.slides[0].acquisitions:
                try:
                    img = f.read_acquisition(acq)
                    if img.shape[1] < 700 or img.shape[2] < 700:
                        continue
                except:
                    continue
                roi_num = int(_rmlead(re.search('[0-9]+', acq.description).group()))
                marker = acq.channel_labels

                idxs = []
                for m in self.__imc_markers:
                    idxs.append(marker.index(m))

                tmp_img = np.zeros_like(img[0])
                for idx in idxs:
                    tmp_img += img[idx]

                mean_img = tmp_img/len(self.__imc_markers)
                roi_images.append({
                    "roi_num": roi_num,
                    "imc_img": mean_img,
                    "imc_marker": marker,
                    "imc_imgs": img
                })

            return pd.DataFrame(roi_images)

    class ROI():
        def __init__(self, df, if_marker):
            self.roi_num = df["roi_num"]
            self.if_nuc = cv2.imread(df["if_b"], 0).astype('float32')
            self.if_nuc = remove_zero_pad(self.if_nuc)
            self.if_marker = if_marker
            self.if_imgs = filter_hot_pixels(np.array([cv2.imread(df[x], 0).astype('float32') for x in ["if_b", "if_g", "if_r"]]), thres=50)
            self.if_imgs = remove_zero_pad(self.if_imgs)
            self.imc_nuc = df["imc_img"]
            self.imc_marker = df["imc_marker"]
            self.imc_imgs = filter_hot_pixels(np.array([i  for i in df["imc_imgs"]]), thres=50)
