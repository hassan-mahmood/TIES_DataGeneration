import os
import tensorflow as tf


tfdirpath='tfrecords/'
for file in os.listdir(tfdirpath):
    reader = tf.python_io.tf_record_iterator(os.path.join(tfdirpath,file))

    for i,example in enumerate(reader):
        features=tf.train.Example.FromString(example).features.feature
        image=features['image']
        imageshape=features['imageshape']
        cellmatrix=features['cellmatrix']
        colmatrix=features['colmatrix']
        rowmatrix=features['rowmatrix']
        wordscount=features['wordscount']
        lengthsarr=features['lengthsarr']
        wordsarr=features['wordsarr']
        xmins=features['xmins']
        xmaxs=features['xmaxs']
        ymins=features['ymins']
        ymaxs=features['ymaxs']


