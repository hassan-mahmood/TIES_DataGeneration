import tensorflow as tf
import numpy as np
import traceback
import cv2
import os
import shutil
import string
import pickle
from tqdm import tqdm
from multiprocessing import Process,Lock
import time
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
import warnings

def warn(*args,**kwargs):
    pass

class GenerateTFRecord:
    def __init__(self, outpath,filesize,tfcount,unlvimagespath,unlvocrpath,unlvtablepath):
        self.outtfpath = outpath
        self.filesize=filesize
        self.num_of_tfs=tfcount
        self.unlvocrpath=unlvocrpath
        self.unlvimagespath=unlvimagespath
        self.unlvtablepath=unlvtablepath
        self.create_dir(self.outtfpath)
        #self.logdir = 'logdir/'
        #self.create_dir(self.logdir)
        #logging.basicConfig(filename=os.path.join(self.logdir,'Log.log'), filemode='a+', format='%(name)s - %(levelname)s - %(message)s')
        self.writer=None
        self.lock=Lock()
        self.num_of_max_vertices=900
        self.max_length_of_word=30
        self.num_data_dims=5
        self.filecounter=1
        self.threadscounter=0
        self.tables_categories = {'types': [0, 1], 'probs': [0.5, 0.5]}
        self.borders_categories = {'types': [0, 1, 2, 3], 'probs': [0.1, 0.2, 0.3, 0.4]}
        #self.str_to_chars=lambda str:np.chararray(list(str))

    def create_dir(self,fpath):
        if(not os.path.exists(fpath)):
            os.mkdir(fpath)

    def str_to_int(self,str):
        intsarr=np.array([ord(chr) for chr in str])
        padded_arr=np.zeros(shape=(self.max_length_of_word),dtype=np.int64)
        padded_arr[:len(intsarr)]=intsarr
        return padded_arr

    def convert_to_int(self, arr):
        return [int(val) for val in arr]

    def pad_with_zeros(self,arr,shape):
        dummy=np.zeros(shape,dtype=np.int64)
        dummy[:arr.shape[0],:arr.shape[1]]=arr
        return dummy

    def generate_tf_record(self, im, cellmatrix, rowmatrix, colmatrix, arr):


        cellmatrix=self.pad_with_zeros(cellmatrix,(self.num_of_max_vertices,self.num_of_max_vertices))
        colmatrix = self.pad_with_zeros(colmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))
        rowmatrix = self.pad_with_zeros(rowmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))

        #im = np.array(cv2.imread(img_path, 0),dtype=np.int64)
        im=im.astype(np.int64)
        img_height, img_width=im.shape

        words_arr = arr[:, 1].tolist()
        no_of_words = len(words_arr)


        lengths_arr = self.convert_to_int(arr[:, 0])
        vertex_features=np.zeros(shape=(self.num_of_max_vertices,self.num_data_dims),dtype=np.int64)
        lengths_arr=np.array(lengths_arr).reshape(len(lengths_arr),-1)
        sample_out=np.concatenate((arr[:,2:],lengths_arr),axis=1)
        vertex_features[:no_of_words,:]=sample_out


        #vertex_text=np.chararray(shape=(self.num_of_max_vertices,self.max_length_of_word))
        #vertex_text[:no_of_words,:]=list(map(self.str_to_chars, words_arr))
        #vertex_text=words_arr+[""]*(self.num_of_max_vertices-len(words_arr))

        vertex_text = np.zeros((self.num_of_max_vertices,self.max_length_of_word), dtype=np.int64)
        vertex_text[:no_of_words]=np.array(list(map(self.str_to_int,words_arr)))


        feature = dict()
        feature['image'] = tf.train.Feature(float_list=tf.train.FloatList(value=im.astype(np.float32).flatten()))
        feature['global_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array([img_height, img_width,no_of_words]).astype(np.float32).flatten()))
        feature['vertex_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=vertex_features.astype(np.float32).flatten()))
        feature['adjacency_matrix_cells'] = tf.train.Feature(int64_list=tf.train.Int64List(value=cellmatrix.astype(np.int64).flatten()))
        feature['adjacency_matrix_cols'] = tf.train.Feature(int64_list=tf.train.Int64List(value=colmatrix.astype(np.int64).flatten()))
        feature['adjacency_matrix_rows'] = tf.train.Feature(int64_list=tf.train.Int64List(value=rowmatrix.astype(np.int64).flatten()))
        feature['vertex_text'] = tf.train.Feature(int64_list=tf.train.Int64List(value=vertex_text.astype(np.int64).flatten()))

        all_features = tf.train.Features(feature=feature)


        seq_ex = tf.train.Example(features=all_features)
        return seq_ex

    def generate_tables(self,driver,N_imgs,output_file_name):
        #np.random.seed(time.time())
        arr = np.random.randint(1, 10, (N_imgs, 2))

        arr[:,0]=arr[:,0]+2
        data_arr=[]
        exceptioncount=0
        for i, subarr in enumerate(arr):

            rows = subarr[0]
            cols = subarr[1]
            while(True):
                try:
                    table_type = random.choices(self.tables_categories['types'], weights=self.tables_categories['probs'])[0]
                    border_type = random.choices(self.borders_categories['types'], weights=self.borders_categories['probs'])[0]
                    table = Table(rows,cols,self.unlvimagespath,self.unlvocrpath,self.unlvtablepath,table_type,border_type)

                    start1=time.time()
                    same_cell_matrix,same_col_matrix,same_row_matrix, id_count, html_content= table.create()
                    #print('table creation time:',time.time()-start1)

                    start2=time.time()
                    im,bboxes = html_to_img(driver, html_content, id_count, 768, 1366)
                    #print('html to img time:',time.time()-start2)
                    data_arr.append([[same_row_matrix, same_col_matrix, same_cell_matrix, bboxes],[im]])

                    break
                    #pickle.dump([same_row_matrix, same_col_matrix, same_cell_matrix, bboxes], infofile)
                except Exception as e:
                    exceptioncount+=1
                    if(exceptioncount>30):
                        return None
                    traceback.print_exc()
                    print('\nException No.', exceptioncount, ' File: ', str(output_file_name))
                    #logging.error("Exception Occured "+str(output_file_name),exc_info=True)
        if(len(data_arr)!=N_imgs):
            print('Images not equal to the required size.')
            return None
        return data_arr

    def draw_col_matrix(self,im,arr,matrix):


        no_of_words=len(arr)
        colors = np.random.randint(0, 255, (no_of_words, 3))
        arr = arr[:, 2:]

        print('arr shape:',arr.shape)
        print('matrix shape:',matrix.shape)
        print('colors shape:', colors.shape)
        print('image shape:',im.shape)
        print('matrix\n',matrix)



        im=im.astype(np.uint8)
        im=np.dstack((im,im,im))
        #im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        # for x in range(no_of_words):
        #     indices = np.argwhere(matrix[x] == 1)
        #     for index in indices:
        #         cv2.rectangle(im, (int(arr[index, 0]), int(arr[index, 1])),
        #                       (int(arr[index, 2]), int(arr[index, 3])),
        #                       (0,0,255), 1)
        x=1
        indices = np.argwhere(matrix[x] == 1)
        for index in indices:
            cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
                          (int(arr[index, 2])+3, int(arr[index, 3])+3),
                          (0,255,0), 1)

        x = 4
        indices = np.argwhere(matrix[x] == 1)
        for index in indices:
            cv2.rectangle(im, (int(arr[index, 0]) - 3, int(arr[index, 1]) - 3),
                          (int(arr[index, 2]) + 3, int(arr[index, 3]) + 3),
                          (0, 0, 255), 1)

        #im=cv2.equalizeHist(im)
        cv2.imwrite('hassan22.jpg',im)


    def write_tf(self,filesize,start):

        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        opts = Options()
        opts.set_headless()
        assert opts.headless
        # driver=PhantomJS()
        driver = Firefox(options=opts)
        while(True):
            starttime = time.time()

            output_file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) + '.tfrecord'
            print('Started:', output_file_name)

            with tf.python_io.TFRecordWriter(os.path.join(self.outtfpath,output_file_name),options=options) as writer:
                try:
                    data_arr=self.generate_tables(driver,filesize,output_file_name)
                    for subarr in data_arr:
                        arr=subarr[0]
                        img=np.asarray(subarr[1][0],np.int64)[:,:,0]
                        colmatrix = np.array(arr[1],dtype=np.int64)
                        cellmatrix = np.array(arr[2],dtype=np.int64)
                        rowmatrix = np.array(arr[0],dtype=np.int64)
                        bboxes = np.array(arr[3])
                        # self.draw_col_matrix(img,bboxes, rowmatrix)
                        # driver.stop_client()
                        # driver.quit()
                        # 0 / 0
                        seq_ex = self.generate_tf_record(img, cellmatrix, rowmatrix, colmatrix, bboxes)
                        writer.write(seq_ex.SerializeToString())
                    print('Completed in ',time.time()-starttime,' ' ,output_file_name,'with len:',(len(data_arr)))

                except Exception as e:
                    traceback.print_exc()
                    print('Removing',output_file_name)
                    os.remove(os.path.join(self.outtfpath,output_file_name))

        driver.stop_client()
        driver.quit()


    def write_to_tf(self,max_threads):

        starttime=time.time()
        threads=[]
        start=len(os.listdir(self.outtfpath))
        for _ in range(max_threads):
            proc = Process(target=self.write_tf, args=(self.filesize, start))
            proc.start()
            threads.append(proc)

        for proc in threads:
            proc.join()

        print(time.time()-starttime)
