
from TFGeneration.GenerateTFRecord import *
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--filesize',type=int,default=100)
parser.add_argument('--threads',type=int,default=5)
args=parser.parse_args()


t = GenerateTFRecord('gentables/','tfrecords/',args.filesize)
t.write_to_tf()