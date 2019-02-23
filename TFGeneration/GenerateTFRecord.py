import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
import pickle
from tqdm import tqdm

class GenerateTFRecord:
    def __init__(self, inpath, outpath):
        self.outtfpath = outpath
        self.inpath=inpath
        if(not os.path.exists(self.outtfpath)):
            os.mkdir(self.outtfpath)

        if (not os.path.exists(self.inpath)):
            print('\nInput directory does not exist')

        self.inpicklepath = inpath

    def convert_to_int(self, arr):
        return [int(val) for val in arr]

    def generate_tf_record(self, img_path, cellmatrix, rowmatrix, colmatrix, arr):

        cellmatrix = cellmatrix.tostring()
        colmatrix = colmatrix.tostring()
        rowmatrix = rowmatrix.tostring()
        img_height, img_width = cv2.imread(img_path, 0).shape

        with tf.gfile.FastGFile(img_path, 'rb') as fid:
            im = fid.read()
        words_arr = arr[:, 1]
        words_arr = [val.encode('utf-8') for val in words_arr]
        lengths_arr = self.convert_to_int(arr[:, 0])
        xmins = self.convert_to_int(arr[:, 2])
        ymins = self.convert_to_int(arr[:, 3])
        xmaxs = self.convert_to_int(arr[:, 4])
        ymaxs = self.convert_to_int(arr[:, 5])
        no_of_words = len(words_arr)

        all_features = tf.train.Features(feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im])),
                                          'imageshape': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=[img_height, img_width])),
                                          'cellmatrix': tf.train.Feature(
                                              bytes_list=tf.train.BytesList(value=[cellmatrix])),
                                          'colmatrix': tf.train.Feature(
                                              bytes_list=tf.train.BytesList(value=[colmatrix])),
                                          'rowmatrix': tf.train.Feature(
                                              bytes_list=tf.train.BytesList(value=[rowmatrix])),
                                          'wordscount': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=[no_of_words])),
                                          'lengthsarr': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=lengths_arr)),
                                          'wordsarr': tf.train.Feature(
                                              bytes_list=tf.train.BytesList(value=words_arr)),
                                          'xmins': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=xmins)),
                                          'ymins': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=ymins)),
                                          'xmaxs': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=xmaxs)),
                                          'ymaxs': tf.train.Feature(
                                              int64_list=tf.train.Int64List(value=ymaxs))
                                          })
        # features_list=tf.train.FeatureLists(feature_list={'words_data':tf.train.FeatureList(feature=table_features)})

        seq_ex = tf.train.Example(features=all_features)
        return seq_ex

    def write_tf(self):
        tfrecord_file_path=os.path.join(self.outtfpath,'mytfrecords.tfrecord')
        with tf.python_io.TFRecordWriter(tfrecord_file_path) as writer:
            for file in tqdm(os.listdir(self.inpicklepath)):
                if (file.endswith('.png')):
                    img_path = os.path.join(self.inpicklepath, file)
                    pickle_file = open(img_path.replace('.png', ''), 'rb')
                    arr = pickle.load(pickle_file)

                    colmatrix = arr[1]
                    cellmatrix = arr[2]
                    rowmatrix = arr[0]
                    bboxes = np.array(arr[3])


                    seq_ex = self.generate_tf_record(img_path, cellmatrix, rowmatrix, colmatrix, bboxes)
                    writer.write(seq_ex.SerializeToString())


