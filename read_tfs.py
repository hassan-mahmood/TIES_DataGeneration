import os
import tensorflow as tf
import numpy as np
import cv2

tfdirpath='tfrecords/'
for file in os.listdir(tfdirpath):

    #reader = tf.python_io.tf_record_iterator(os.path.join(tfdirpath,file))

    sess = tf.InteractiveSession()

    # Read TFRecord file
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([os.path.join(tfdirpath,file)])

    _, serialized_example = reader.read(filename_queue)

    #Define features
    keys_to_features = {
        'vertex_features': tf.FixedLenFeature((900*4), tf.int64),
        'image': tf.FixedLenFeature((1366*768), tf.int64),
        'global_features': tf.FixedLenFeature((3), tf.int64),
        'adjacency_matrix_cells': tf.FixedLenFeature((900*900), tf.int64),
        'adjacency_matrix_rows': tf.FixedLenFeature((900*900), tf.int64),
        'adjacency_matrix_cols': tf.FixedLenFeature((900*900), tf.int64),
        'vertex_text': tf.FixedLenFeature((900), tf.string)
    }


    # Extract features from serialized data
    read_data = tf.parse_single_example(serialized=serialized_example,
                                        features=keys_to_features)

    # Many tf.train functions use tf.train.QueueRunner,
    # so we need to start it before we read
    tf.train.start_queue_runners(sess)

    # Print features
    while(True):
        vertex_text=read_data['adjacency_matrix_cells'].eval()
        #vertex_text=tf.reshape(vertex_text,(900,900)).eval()
        print(vertex_text)


    #vertex_text=np.fromstring(vertex_text)


    # for name, tensor in read_data.items():
    #     print('{}: {}'.format(name, tensor.eval()))

