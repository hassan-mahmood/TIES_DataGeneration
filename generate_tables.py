from TableGeneration.Distribution import Distribution
from TableGeneration.Table import Table
import time
import random
import argparse
from TableGeneration.tools import *
import os
import pickle
import numpy as np
from tqdm import tqdm
parser=argparse.ArgumentParser()

parser.add_argument('--imagespath',default='../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../Table_Detection_Dataset/unlv/unlv_xml_gt')
parser.add_argument('--cols',default=0)
parser.add_argument('--rows',default=0)
parser.add_argument('--htmlpath',default="file:///media/hassan/NewVolume/Server/Projects/DocAnalysis/myfile.html")
parser.add_argument('--N',default=2,type=int,help='Number of images to generate')
parser.add_argument('--outpath',help='output directory to store output images',default='gentables/')
parser.add_argument('--distributionpath',default='distribution_pickle')
args=parser.parse_args()


if(not os.path.exists(args.outpath)):
    os.mkdir(args.outpath)

cols=int(args.cols)
rows=int(args.rows)
#random.seed(a=None,version=2)
arr=np.random.randint(1,10,(args.N,2))

start=len(os.listdir(args.outpath))//2

for i,subarr in enumerate(arr):
    rows=subarr[0]
    cols=subarr[1]

    table=Table(rows,cols,args.imagespath,args.ocrpath,args.tablepath)
    same_row_matrix,same_col_matrix,same_cell_matrix,id_count=table.create_html_table()
    bboxes=html_to_img(args.htmlpath,os.path.join(args.outpath,str(i+start)+'.png'),id_count)
    infofile=open(os.path.join(args.outpath,str(i+start)),'wb')
    pickle.dump([same_row_matrix,same_col_matrix,same_cell_matrix,bboxes],infofile)
    infofile.close()
    print('Completed: ',i+start)
