
from TFGeneration.GenerateTFRecord import *
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--filesize',type=int,default=10)
parser.add_argument('--threads',type=int,default=15)
parser.add_argument('--inpath',default='gentables/')
parser.add_argument('--outpath',default='tfrecords/')
args=parser.parse_args()


t = GenerateTFRecord(args.inpath,args.outpath,args.filesize)
t.write_to_tf(args.threads)