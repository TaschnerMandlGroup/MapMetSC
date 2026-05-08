import cv2
import numpy as np
import pandas as pd
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest = 'path', required=True, help='path to coordinates.xlsx')
    args = parser.parse_args()
    return args


def relocate_IF_IMC(path: str):

    coord = pd.read_excel(path)

    #Coordinates of 3 reference points in IF microscopy
    grid_IF = np.float32([[coord.x_IF[0], coord.y_IF[0]], [coord.x_IF[1], coord.y_IF[1]], [coord.x_IF[2], coord.y_IF[2]]])

    #Coordinates of 3 reference points in IMC Hyperion (coordinates are switched in IMC)
    grid_IMC = np.float32([[coord.y_IMC[0], coord.x_IMC[0]], [coord.y_IMC[1], coord.x_IMC[1]], [coord.y_IMC[2], coord.x_IMC[2]]])

    #Create RoI names
    RoIs = [i for i in coord.Point if 'RoI' in i]
    RoI_names = ['ROI_0' + '0' + i.split('I')[-1] if int(i.split('I')[-1])<10 else 'ROI_0' + i.split('I')[-1] for i in RoIs]

    #Create dataframe for IMC-readout
    data = {'Name': RoI_names, 'W': [700 for i in RoI_names], 'H': [700 for i in RoI_names]}  
    RoI_df = pd.DataFrame(data)  

    for i, j in enumerate(RoI_df.Name):
        # Coordinates of selected cell of interest in IF microscopy
        RoI_IF = np.float32([[[coord.x_IF[i+3], coord.y_IF[i+3]]]])

        #Estimate transformation function
        m, _ = cv2.estimateAffinePartial2D(grid_IF, grid_IMC, cv2.RANSAC)

        #Apply transformation funtion
        RoI_IMC = cv2.transform(RoI_IF, m)
        RoI_IMC_x_upperleft = RoI_IMC[0, 0, 1] - 350
        RoI_IMC_y_upperleft = RoI_IMC[0, 0, 0] + 350

        idx = RoI_df.index[RoI_df.Name==j]
        RoI_df.loc[idx, 'X'] = RoI_IMC_x_upperleft
        RoI_df.loc[idx, 'Y'] = RoI_IMC_y_upperleft

    #Write to csv
    RoI_df.to_csv(path.split('.')[0] + '_relocated.csv', index=False, sep=',', decimal='.')

def main (args):

    in_path = args.path

    relocate_IF_IMC(in_path)

if __name__ == "__main__":
        args = parse()
        main(args)




    