import numpy as np

rows=6
cols=5
arr=np.array([[[]]*cols]*rows)
arr=np.empty(shape=(rows,cols),dtype=object)
arr[0,1]=[1,2]
print(arr)
