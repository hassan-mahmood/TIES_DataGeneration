from TFGeneration.GenerateTFRecord import *
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--filesize',type=int,default=1)
parser.add_argument('--threads',type=int,default=1)
parser.add_argument('--outpath',default='tfrecords/')
parser.add_argument('--tfcount',type=int,default=1)
parser.add_argument('--imagespath',default='../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../Table_Detection_Dataset/unlv/unlv _xml_gt')
parser.add_argument('--level',type=int,default=4)
parser.add_argument('--writetoimg',type=int,default=1)
args=parser.parse_args()

writetoimg=False
if(args.writetoimg==1):
    writetoimg=True
t = GenerateTFRecord(args.outpath,args.filesize,args.tfcount,args.imagespath,
                     args.ocrpath,args.tablepath,args.level,writetoimg)
t.write_to_tf(args.threads)
