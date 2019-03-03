from TableGeneration.Distribution import Distribution
from TableGeneration.Table import Table
from multiprocessing import Process
import time
import random
import argparse
from TableGeneration.tools import *
import os
import pickle
import numpy as np
from tqdm import tqdm
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS

parser=argparse.ArgumentParser()

parser.add_argument('--imagespath',default='../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../Table_Detection_Dataset/unlv/unlv_xml_gt')
parser.add_argument('--cols',default=0)
parser.add_argument('--rows',default=0)
parser.add_argument('--N',default=2,type=int,help='Number of images to generate')
parser.add_argument('--outpath',help='main output directory to store output images',default='gentables/')
parser.add_argument('--distributionpath',default='distribution_pickle')
parser.add_argument('--threads',type=int,default=4)
args=parser.parse_args()

#random.seed(a=None,version=2)
import time


def create_dir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)


def generate(outpath):

    create_dir(outpath)

    arr=np.random.randint(1,10,(args.N,2))

    start=len(os.listdir(outpath))//2

    opts = Options()
    opts.set_headless()
    assert opts.headless
    #driver=PhantomJS()
    driver = Firefox(options=opts)

    for i,subarr in enumerate(arr):
        rows=subarr[0]
        cols=subarr[1]
        try:
            table=Table(rows,cols,args.imagespath,args.ocrpath,args.tablepath)
            same_row_matrix,same_col_matrix,same_cell_matrix,id_count,html_content=table.create_html()
            bboxes=html_to_img(driver,html_content,os.path.join(outpath,str(i+start)+'.png'),id_count,768,1366)
            infofile=open(os.path.join(outpath,str(i+start)),'wb')
            pickle.dump([same_row_matrix,same_col_matrix,same_cell_matrix,bboxes],infofile)
            infofile.close()
        except:
            print('\nException')
            pass

    print('Completed: ',outpath)

    driver.stop_client()
    driver.quit()

create_dir(args.outpath)

startime=time.time()
procs=[]

#for storing html files

for i in range(args.threads):
    outpath=os.path.join(args.outpath+str(i)+'folder')
    proc=Process(target=generate,args=(outpath,))
    procs.append(proc)
    proc.start()

for proc in procs:
    proc.join()

print(time.time()-startime)



