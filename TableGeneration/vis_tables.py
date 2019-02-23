import pickle
import cv2
import os
import glob
outpath='withbb/'
if(not os.path.exists(outpath)):
    os.mkdir(outpath)

path='gentables/'
files=[]
for file in os.listdir(path):
    if(file.endswith('.png')):
        files.append(file)

for file in files:
    print(file)
    f=open(os.path.join(path,str(file.replace('.png',''))),'rb')
    im=cv2.imread(os.path.join(path,file))
    arr=pickle.load(f)
    bboxes=arr[3]
    for bbox in bboxes:
        cv2.rectangle(im,(int(bbox[2])-5,int(bbox[3])-5),(int(bbox[4]),int(bbox[5])),(0,0,255),2)

    cv2.imwrite(os.path.join(outpath,file),im)
    print('\nCompleted: ',im)
    # print(im.shape)
    # cv2.imshow('Image',im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()