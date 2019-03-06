import argparse
import random
from TableGeneration.Table import Table
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


cols=random.randint(1,7)
rows=random.randint(2,10)
rows=8
cols=7


print('rows:',rows,' cols:',cols)

# Table Types:
# 0: regular headers , 1: irregular headers

# Border Types:
# 0: complete border, 1: completely w/o borders, 2: with lines underhead, 3: internal borders


tables={'types':[0,1],'probs':[0.5,0.5]}
borders={'types':[0,1,2,3],'probs':[0.1,0.2,0.3,0.4]}

table_type=random.choices(tables['types'],weights=tables['probs'])[0]
border_type=random.choices(borders['types'],weights=borders['probs'])[0]
print('border type:',border_type)
table=Table(rows,cols,args.imagespath,args.ocrpath,args.tablepath,table_type,border_type)
htmlcontent=table.create()
