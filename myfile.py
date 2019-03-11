from TableGeneration.Distribution import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

imagespath='/media/hassan/NewVolume/Server/Projects/Table_Detection_Dataset/unlv/unlv_images'
ocrpath='/media/hassan/NewVolume/Server/Projects/Table_Detection_Dataset/unlv/unlv_xml_ocr'
tablepath='/media/hassan/NewVolume/Server/Projects/Table_Detection_Dataset/unlv/unlv_xml_gt'

# if(not os.path.exists(picklefile)):
#     distribution=Distribution(imagespath,ocrpath,tablepath)
#     all_tables_data=distribution.get_distribution()
#
#     f=open(picklefile,'wb')
#     pickle.dump(all_tables_data,f)
#     f.close()
#     print('tables data stored in pickle file')
#
# else:
#     f=open(picklefile,'rb')
#     all_tables_data =pickle.load(f)
#     f.close()
#     print('loaded from file')


# distribution=Distribution(imagespath,ocrpath,tablepath)
# all_tables_data=distribution.get_distribution()
#
# all_words=[]
# all_numbers=[]
# all_others=[]
# len_words=[]
# len_numbers=[]
# len_others=[]
# row_cols=[]
# for arr in all_tables_data:
#     all_data=arr[1][0]
#     row_col=all_data[0]
#     data=all_data[1]
#
#     row_cols.append(row_col)
#     len_words.append(len(data['alphabet']))
#     len_numbers.append(data['number'])
#     len_others.append(data['other'])
#
#     all_words+=data['alphabet']
#     all_numbers+=data['number']
#     all_others+=data['other']
#
# print(len(all_words))
# print(len(all_numbers))
# print(len(all_others))



# data = []
# for arr in all_tables_data:
#     filename,counter=arr[0],arr[1][0]
#     row_col_counter = counter[0]
#     #print(filename,row_col_counter)
#     data.append([row_col_counter['row'], row_col_counter['column']])
#
# data=np.array(data)
# N=len(data)
# colors = np.random.rand(N)
# #area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#
# plt.scatter(data[:,0], data[:,1], c='r',marker='.', alpha=0.5)
# plt.savefig('data.png')
# print('Image stored')
#plt.show()

# print(np.array(data))
# print(np.array(data).shape)

#_______________________________________________
from PIL import Image,ImageDraw
from skimage import io
from TableGeneration.Transformation import Transform
import pickle
# f=open('points','rb')
# bboxes=pickle.load(f)
# f.close()

max_width=1366
max_height=768

img=io.imread('nowimg.png')

import time
start=time.time()
bboxes=[[0,0,0,0]]
out,transformed_bboxes =Transform(img,bboxes,-0.00001,-0.01,max_width,max_height)

draw = ImageDraw.Draw(out)

# for bbox in transformed_bboxes:
#     draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=(0, 0, 255))

print(time.time()-start)
print('\n final out:',out.size)
out.save('trn.png')
#
# import numpy as np
#
#
# T=np.random.randint(0,1,size=(3,3))
# pts=np.random.randint(0,1,size=(10,3))
# offsets=np.array([5,4,1])
# offsets=np.tile(offsets,(len(pts),1))
# pts=pts+offsets
# out=np.dot(pts,T)
# print(out.shape)
# out=np.concatenate((out,out),axis=1)
# print(out.shape)
