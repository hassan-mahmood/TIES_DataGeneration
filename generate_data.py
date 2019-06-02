from TFGeneration.GenerateTFRecord import *
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--filesize',type=int,default=1)                #number of images in a single tfrecord file
parser.add_argument('--threads',type=int,default=1)                 #one thread will work on one tfrecord
parser.add_argument('--outpath',default='tfrecords/')               #directory to store tfrecords

#imagespath,
parser.add_argument('--imagespath',default='../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../Table_Detection_Dataset/unlv/unlv _xml_gt')

parser.add_argument('--writetoimg',type=int,default=0)              #if True, will store the images along with tfrecords

args=parser.parse_args()

writetoimg=False
if(args.writetoimg==1):
    writetoimg=True
t = GenerateTFRecord(args.outpath,args.filesize,args.imagespath,
                     args.ocrpath,args.tablepath,writetoimg)
t.write_to_tf(args.threads)
