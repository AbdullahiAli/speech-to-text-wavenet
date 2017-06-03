import sugartensor as tf
import numpy as np
import csv
import string
import os
import random 
__author__ = 'namju.kim@kakaobrain.com'


# default data path
_data_path = '/media/srv/data/'

#
# vocabulary table
#

# index to byte mapping
index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# byte to index mapping
byte2index = {}
for i, ch in enumerate(index2byte):
    byte2index[ch] = i

# vocabulary size
voca_size = len(index2byte)


# convert sentence to index list
def str2index(str_):

    # clean white space
    str_ = ' '.join(str_.split())
    # remove punctuation and make lower case
    str_ = str_.translate(None, string.punctuation).lower()

    res = []
    for ch in str_:
        try:
            res.append(byte2index[ch])
        except KeyError:
            # drop OOV
            pass
    return res


# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for ch in index_list:
      
        if ch > 0:
            str_ += index2byte[ch]
        elif ch == 0:  # <EOS>
            break
    return str_


# print list of index list
def print_index(indices):
    for index_list in indices:
        print(index2str(index_list))
        
def return_index(indices):
     sentence = ''
     for index_list in indices:
        sentence += " " + index2str(index_list)
     return sentence

# real-time wave to mfcc conversion function
@tf.sg_producer_func
def _load_mfcc(src_list):

    # label, wave_file
    label, mfcc_file = src_list

    # decode string to integer
    label = np.fromstring(label, np.int)

    # load mfcc
    mfcc = np.load(mfcc_file, allow_pickle=False)

    # speed perturbation augmenting
    mfcc = _augment_speech(mfcc)

    return label, mfcc


def _augment_speech(mfcc):

    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc = np.roll(mfcc, r, axis=0)

    # zero padding
    if r > 0:
        mfcc[:r, :] = 0
    elif r < 0:
        mfcc[r:, :] = 0

    return mfcc

# Speech Corpus
class NonNativeSpeechCorpus(object):

      def __init__(self, batch_size=16, set_name='train'):

        # load meta file
        label, mfcc_file = [], []
        with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                # mfcc file
                
                if set_name == 'non_native_train':
                      mfcc_file.append(_data_path + 'preprocess/non_native_train_mfcc/' + row[0] + '.npy')
                elif set_name == "non_native_test":
                    mfcc_file.append(_data_path + 'preprocess/non_native_test_mfcc/' + row[0] + '.npy')
                else:
                    mfcc_file.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
                # label info ( convert to string object for variable-length support )
                label.append(np.asarray(row[1:], dtype=np.int).tostring())

       

        # split data
        mfcc = [_augment_speech(np.load(file)) for file in mfcc_file]
        self.label, self.mfcc = label, mfcc
     
       
        # calc total batch count
        self.num_batch = len(label) // batch_size

        # print info
        tf.sg_info('%s set loaded.(total data=%d, total batch=%d)'
                   % (set_name.upper(), len(label), self.num_batch))

# Speech Corpus
class SpeechCorpus(object):

    def __init__(self, batch_size=16, set_name='train'):

        # load meta file
        label, mfcc_file = [], []
        with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                # mfcc file
                
                if set_name == 'strategy1':
                      mfcc_file.append(_data_path + 'preprocess/non_native_strategy1_mfcc/' + row[0] + '.npy')
                elif set_name == "strategy3":
                    mfcc_file.append(_data_path + 'preprocess/non_native_strategy3_mfcc/' + row[0] + '.npy')
                else:
                    mfcc_file.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
                # label info ( convert to string object for variable-length support )
                label.append(np.asarray(row[1:], dtype=np.int).tostring())

        
        
        # Test whether it works with a mini_batch:
        #label, mfcc_file = zip(*random.sample(list(zip(label,mfcc_file)),10000))
        #label, mfcc_file = list(label), list(mfcc_file)
        
        # Artificially enlarge the data set if non-native data
       # if set_name == 'non_native_train':
          #  label *= 500
           # mfcc_file *= 500
       
       
        # to constant tensor
        
        label_t = tf.convert_to_tensor(label)
        mfcc_file_t = tf.convert_to_tensor(mfcc_file)

        # create queue from constant tensor
        label_q, mfcc_file_q \
            = tf.train.slice_input_producer([label_t, mfcc_file_t], shuffle=True)

        # create label, mfcc queue
        label_q, mfcc_q = _load_mfcc(source=[label_q, mfcc_file_q],
                                     dtypes=[tf.sg_intx, tf.sg_floatx],
                                     capacity=256, num_threads=64)


        # create batch queue with dynamic pad
        batch_queue = tf.train.batch([label_q, mfcc_q], batch_size,
                                     shapes=[(None,), (20, None)],
                                     num_threads=64, capacity=batch_size*32,
                                     dynamic_pad=True)

        # split data
        self.label, self.mfcc = batch_queue
     
        # batch * time * dim
        self.mfcc = self.mfcc.sg_transpose(perm=[0, 2, 1])
        # calc total batch count
        self.num_batch = len(label) // batch_size

        # print info
        tf.sg_info('%s set loaded.(total data=%d, total batch=%d)'
                   % (set_name.upper(), len(label), self.num_batch))
