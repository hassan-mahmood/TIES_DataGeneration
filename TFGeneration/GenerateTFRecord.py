import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
import pickle
from tqdm import tqdm

class GenerateTFRecord:
    def __init__(self, inpath, outpath,filesize):
        self.outtfpath = outpath
        self.inpath=inpath
        self.filesize=filesize
        if(not os.path.exists(self.outtfpath)):
            os.mkdir(self.outtfpath)

        if (not os.path.exists(self.inpath)):
            print('\nInput directory does not exist')

        self.inpicklepath = inpath
        self.num_of_max_vertices=900
        self.max_length_of_word=30
        #self.str_to_chars=lambda str:np.chararray(list(str))

    def str_to_chars(self,str):
        charr=np.chararray(shape=(1,self.max_length_of_word))
        charr[:]=''
        charr[0,:len(str)]=list(str)
        return charr[0]

    def convert_to_int(self, arr):
        return [int(val) for val in arr]

    def pad_with_zeros(self,arr,shape):
        dummy=np.zeros(shape,dtype=np.int64)
        dummy[:arr.shape[0],:arr.shape[1]]=arr
        return dummy

    def generate_tf_record(self, img_path, cellmatrix, rowmatrix, colmatrix, arr):

        # cellmatrix = cellmatrix.tostring()
        # colmatrix = colmatrix.tostring()
        # rowmatrix = rowmatrix.tostring()
        cellmatrix=self.pad_with_zeros(cellmatrix,(self.num_of_max_vertices,self.num_of_max_vertices))
        colmatrix = self.pad_with_zeros(colmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))
        rowmatrix = self.pad_with_zeros(rowmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))

        im = np.array(cv2.imread(img_path, 0),dtype=np.int64)
        img_height, img_width=im.shape

        words_arr = arr[:, 1].tolist()
        no_of_words = len(words_arr)

        #words_arr = [val.encode('utf-8') for val in words_arr]
        lengths_arr = self.convert_to_int(arr[:, 0])
        vertex_features=np.zeros(shape=(self.num_of_max_vertices,4),dtype=np.int64)
        vertex_features[:no_of_words,:]=arr[:,2:]

        #vertex_text=np.chararray(shape=(self.num_of_max_vertices,self.max_length_of_word))
        #vertex_text[:no_of_words,:]=list(map(self.str_to_chars, words_arr))
        vertex_text=words_arr+[""]*(self.num_of_max_vertices-len(words_arr))



        all_features = tf.train.Features(feature={'image': tf.train.Feature(int64_list=tf.train.Int64List(value=im.reshape(-1))),
                                          'global_features': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=[img_height, img_width,no_of_words])),
                                          'adjacency_matrix_cells': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=cellmatrix.reshape(-1))),
                                          'adjacency_matrix_cols': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=colmatrix.reshape(-1))),
                                          'adjacency_matrix_rows': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=rowmatrix.reshape(-1))),
                                          'vertex_features': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=vertex_features.reshape(-1))),
                                          'vertex_text': tf.train.Feature(
                                              bytes_list=tf.train.BytesList(value=[val.encode('utf-8') for val in vertex_text]))
                                          })
        # all_features = tf.train.Features(feature={'vertex_text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[val.encode('utf-8') for val in vertex_text]))
        #                                   })
        # features_list=tf.train.FeatureLists(feature_list={'words_data':tf.train.FeatureList(feature=table_features)})

        seq_ex = tf.train.Example(features=all_features)
        return seq_ex

    def write_tf(self):
        with tf.python_io.TFRecordWriter(os.path.join(self.outtfpath,'tfdata.tfrecord')) as writer:
            for file in tqdm(os.listdir(self.inpicklepath)):
                if (file.endswith('.png')):
                    img_path = os.path.join(self.inpicklepath, file)
                    pickle_file = open(img_path.replace('.png', ''), 'rb')
                    arr = pickle.load(pickle_file)

                    colmatrix = np.array(arr[1],dtype=np.int64)
                    cellmatrix = np.array(arr[2],dtype=np.int64)
                    rowmatrix = np.array(arr[0],dtype=np.int64)
                    bboxes = np.array(arr[3])
                    seq_ex = self.generate_tf_record(img_path, cellmatrix, rowmatrix, colmatrix, bboxes)
                    writer.write(seq_ex.SerializeToString())


