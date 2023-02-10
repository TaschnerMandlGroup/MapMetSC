import numpy as np
import cv2
from skimage.segmentation import find_boundaries
from scipy import ndimage
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
import xml.etree.ElementTree
import tifffile
import io

def read_channel_names(path):
    
    tiff = tifffile.TiffFile(path)
    omexml_string = tiff.pages[0].description

    root = xml.etree.ElementTree.parse(io.StringIO(omexml_string))
    namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    channels = root.findall('ome:Image[1]/ome:Pixels/ome:Channel', namespaces)
    channel_names = [c.attrib['Name'] for c in channels]

    return channel_names


def additive_blend(im0, im1):

    im0 = np.array(im0)
    im0 = im0/np.percentile(im0, 99)
    im0 = np.clip(im0, 0, 1)

    im1 = np.array(im1)
    im1 = im1/np.percentile(im1, 99)
    im1 = np.clip(im1, 0, 1)

    rgb = np.zeros((*im0.shape, 3))
    rgb[:, :, 0] = im0
    rgb[:, :, 1] = im1

    return rgb


def _rmlead(inp, char='0'):
    for idx, letter in enumerate(inp):
        if letter != char:
            return inp[idx:]


def plot_segmentation(nuc, segmentation):

    rgb_data = cv2.cvtColor(nuc, cv2.COLOR_GRAY2RGB)
    boundaries = np.zeros_like(nuc)
    overlay_data = np.copy(rgb_data)

    boundary = find_boundaries(segmentation, connectivity=0, mode='outer')
    boundaries[boundary > 0] = 1
    boundaries = cv2.dilate(boundaries, np.ones((2,2)))

    overlay_data[boundaries > 0] = (255,0,0)
    return overlay_data

def gaussian_kernel(sz: int, sigx: float, sigy: float, deg: float, normalize=True) -> np.ndarray:
    X, Y = np.meshgrid(np.linspace(-1,1,sz), np.linspace(-1,1,sz))
    GX = 1/(sigx*np.sqrt(2*np.pi))*np.exp(-1/2*(X/sigx)**2)
    GY = 1/(sigy*np.sqrt(2*np.pi))*np.exp(-1/2*(Y/sigy)**2)
    GK = GX*GY
    rGK = ndimage.rotate(GK, deg, reshape=False)
    if normalize:
        rGK /= rGK.sum()
    else:
        rGK /= rGK.max()
    return rGK


def LoG_kernel(sz: int, sig: float):

    X, Y = np.meshgrid(np.linspace(-4,4,sz), np.linspace(-4,4,sz))
    X2 = X**2
    Y2 = Y**2
    X2_plus_Y2 = X2 + Y2
    hg = np.exp((-X2_plus_Y2)/(2*sig**2))
    LoG = (-1)/(np.pi*sig**4)*(1-(X2_plus_Y2/(2*sig**2)))*np.exp(-X2_plus_Y2/(2*sig**2))
    LoG = LoG/abs(LoG.min())
    return LoG

def remove_zero_pad(image):
    dim = image.ndim
    if dim < 3:
        tmp = np.argwhere(image > 0.0)
    else:
        tmp = np.argwhere(image[0] > 0.0)  

    max_y = tmp[:, 0].max()
    min_y = tmp[:, 0].min()
    min_x = tmp[:, 1].min()
    max_x = tmp[:, 1].max()
    if dim < 3:
        crop_image = image[min_y:max_y, min_x:max_x]
    else:
        crop_image = image[:, min_y:max_y, min_x:max_x]
    return crop_image

def filter_hot_pixels(img: np.ndarray, thres: float) -> np.ndarray:
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img, img)
