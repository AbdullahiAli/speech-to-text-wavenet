# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import librosa
from model import *
import data
from wer import wer

__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 1     # batch size

#
# inputs
#
# corpus input tensor
non_native_data = data.SpeechCorpus(batch_size=batch_size * tf.sg_gpus(), set_name="non_native_test")
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

# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

# to dense tensor
y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1

#
# regcognize wave file
#

# command line argument for input wave file path#tf.sg_arg_def(file=('', 'speech wave file to recognize.'))

# load wave file
#wav, _ = librosa.load(tf.sg_arg().file, mono=True, sr=16000)
# get mfcc feature
#mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, 16000), axis=0), [0, 2, 1])

# run network
with tf.Session() as sess:

    # init variables
    tf.sg_init(sess)
    sess.run(tf.Print([1], [1], "hi"))
    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
    # run session
    for mfcc,label in zip(inputs,labels):
        sess.run(tf.Print([1], [1], type(mfcc)))
        predicted = sess.run(y, feed_dict={x: mfcc.eval()})
    
        data.print_index(predicted)
        data.return_index(predicted)
        error.append(wer(predicted.split(),tf.as_string(label).split()))
    print("WER: " + (sum(error)/len(error)))
        
    # print label
    
