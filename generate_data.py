
from TFGeneration.GenerateTFRecord import *
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--filesize',type=int,default=10)
parser.add_argument('--threads',type=int,default=2)
parser.add_argument('--outpath',default='tfrecords/')
parser.add_argument('--tfcount',type=int,default=10)
parser.add_argument('--imagespath',default='../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../Table_Detection_Dataset/unlv/unlv_xml_gt')
args=parser.parse_args()


t = GenerateTFRecord(args.outpath,args.filesize,args.tfcount,args.imagespath,args.ocrpath,args.tablepath)
t.write_to_tf(args.threads)