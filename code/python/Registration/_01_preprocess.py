import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess(img, percentage=1, plot_hotspots=False, clipLimit=6):
    img = rm_hotpixel(img, plot_hotspots=plot_hotspots)
    img = normalize_clahe(img, clipLimit=clipLimit)
    img = normalize_to_one_percent(img, percentage)

    return (img*255).astype(np.uint8)


def rm_hotpixel(img: np.ndarray, threshold=0.8, plot_hotspots=False) -> np.ndarray:
    
    #change datatype to float32 to be able to compute median filter
    ret = np.copy((img/img.max()*255).astype("float32"))
    
    #calculate blurred version and substract it from original image
    blurred = cv2.medianBlur(ret,3)
    diff = ret-blurred

    #get spot idx where diff value > threshold
    spots = np.array(np.where(diff > threshold*ret.max())).T

    #remove each hotpixel defined in spot
    for spot in spots:
        ret[spot[0], spot[1]] = handle_hotpixel(spot, ret)

    #verbose 
    print(f"{spots.shape[0]} hot pixels found.")

    if plot_hotspots:

        plt.imshow(img, cmap="gray")
        plt.scatter(spots[:,1],spots[:,0], s=.05*np.max(img.shape), marker="x", c="red")
        plt.show()

    return ret


def handle_hotpixel(spot: np.ndarray, img: np.ndarray, sz=(1, 1)) -> np.float32:

    #get temporary image and set spot to NaN
    tmp_img = np.copy(img)
    tmp_img[spot[0], spot[1]] = np.nan

    #check if spot is on the boarder of the image and define the range where the mean is then calculated
    x0 = spot[1]-1 if spot[1] >= sz[1] else 0
    y0 = spot[0]-1 if spot[0] >= sz[0] else 0
    x1 = spot[1]+2 if spot[1] <= tmp_img.shape[1]-sz[1] else tmp_img.shape[1]
    y1 = spot[0]+2 if spot[0] <= tmp_img.shape[0]-sz[1] else tmp_img.shape[0]

    #calculate mean of surrounding area and return it
    tmp = tmp_img[y0:y1, x0:x1]
    mean = np.nanmean(tmp).astype(np.float32)

    return mean

def illumination(inp:np.ndarray, n=8) -> tuple:

    ret = cv2.resize(inp, (512,512))

    for d_i in range(n):

        d_i=int((d_i//2)*2+1)
        tmp_2 = cv2.copyMakeBorder(ret, d_i, d_i, d_i, d_i, cv2.BORDER_REPLICATE)

        for y in range(d_i,ret.shape[0]+d_i):
            for x in range(d_i,ret.shape[1]+d_i):
                c_0 = (tmp_2[x-d_i,y-d_i] + tmp_2[x+d_i,y+d_i])/2
                c_1 = (tmp_2[x+d_i,y-d_i] + tmp_2[x-d_i,y+d_i])/2
                c_2 = (tmp_2[x-d_i,y] + tmp_2[x+d_i,y])/2
                c_3 = (tmp_2[x,y-d_i] + tmp_2[x,y+d_i])/2
                c_4 = tmp_2[x,y]

                tmp_2[x,y] = min([c_0, c_1, c_2, c_3, c_4])
                
        tmp_2 = cv2.GaussianBlur(tmp_2, ksize=(d_i, d_i), sigmaX=d_i, sigmaY=d_i)
        ret = tmp_2[d_i:ret.shape[0]+d_i, d_i:ret.shape[1]+d_i]
    
    ret = cv2.resize(ret, (inp.shape[0], inp.shape[1]))

    return inp-ret, ret
    

def normalize_to_one_percent(img, percentage=1):

    img = ((img / img.max())*255).astype(np.uint8)
    b1 = np.percentile(img, percentage)
    t1 = np.percentile(img, 100-percentage)
    img = (img-b1)/(t1-b1)
    img = np.clip(img, 0, 1)

    return img

def normalize_clahe(img, clipLimit=6):
    #Histogram equalization 
    img = ((img / img.max())*255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    equalized = clahe.apply(img)

    return equalized/equalized.max()


