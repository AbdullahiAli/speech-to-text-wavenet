# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import sugartensor as tf
from data import SpeechCorpus, voca_size
from model import *
from tensorflow.layers import dense

__author__ = 'a.ali@student.ru.nl'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

import sugartensor as tf





#
# logit calculating graph using atrous convolution
#
def get_predictions(logits):
    output = dense(logits)
   
    return output


