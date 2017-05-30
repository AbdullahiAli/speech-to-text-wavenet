# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import librosa
from tensorflow.contrib.learn import LinearClassifier
from model import *
import data
from wer import wer
from softmax_classifier import *
from tensorflow.contrib.learn import LinearClassifier
__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 1     # batch size
# train
#non_native_data_train = data.NonNativeSpeechCorpus(batch_size=batch_size * tf.sg_gpus(), set_name="non_native_train")
#train_inputs = non_native_data_train.mfcc
#train_labels = non_native_data_train.label
#x_train = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))
#train_logit = get_logit(x_train, voca_size=voca_size)
#estimator = LinearClassifier(feature_columns = [])
#estimator.fit(train_logit, train_labels)

#
# inputs
#
# corpus input tensor
non_native_data = data.NonNativeSpeechCorpus(batch_size=batch_size * tf.sg_gpus(), set_name="non_native_test")
error = []
# mfcc feature of audio
inputs = non_native_data.mfcc
# target sentence label
labels = non_native_data.label
# vocabulary size
voca_size = data.voca_size

# mfcc feature of audio
x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))

# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

# encode audio feature
logit = get_logit(x, voca_size=voca_size)


#pred = get_predictions(logit)


# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(pred.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

# to dense tensor
y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1

#
# regcognize wave file
#

# command line argument for input wave file path#tf.sg_arg_def(file=('', 'speech wave file to recognize.'))

# load wave file


# run network
with tf.Session() as sess:

    # init variables
    tf.sg_init(sess)
    sess.run(tf.Print([1], [1], "hi"))
    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
    
    # run session
    for mfcc, label in zip(inputs,labels):
        mfcc = np.transpose(np.expand_dims(mfcc,axis=0),[0,2,1])
        predicted = sess.run(y, feed_dict={x: mfcc})
        print(predicted)
        data.print_index(predicted)
        predicted = data.return_index(predicted)
        error.append(wer(predicted.split(),label.split()))
    print(error)
    print("WER: %2f" % (sum(error)/len(error)))
        
    # print label
    
