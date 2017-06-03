# -*- coding: utf-8 -*-

import sugartensor as tf
from data import SpeechCorpus, voca_size
from model import *
from softmax_classifier import *
__author__ = 'a.ali@student.ru.nl'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # total batch size

            
# non-native corpus input tensor
non_native_data =  SpeechCorpus(batch_size=batch_size * tf.sg_gpus(), set_name ="non_native_train")

# non_native_mfcc features of audio
inputs = tf.split(non_native_data.mfcc, tf.sg_gpus(), axis = 0)
# target non-native sentence labels
labels = tf.split(non_native_data.label, tf.sg_gpus(), axis=0)

# sequence length except zero-padding
seq_len = []
for input_ in inputs:
    seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))


# parallel loss tower
@tf.sg_parallel
def get_loss(opt):
    # encode audio feature
    logits = get_logit(opt.input[opt.gpu_index], voca_size=voca_size)

    linear = get_predictions(logits)

    # CTC loss
    return linear.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])

#
# retrain
#
tf.sg_train(optim = 'Adam', lr=0.01, loss=get_loss(input=inputs, target=labels, seq_len=seq_len),
            ep_size=non_native_data.num_batch, max_ep=100)
            

