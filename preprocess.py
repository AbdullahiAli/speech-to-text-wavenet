import numpy as np
import pandas as pd
import glob
import csv
import librosa
import scikits.audiolab
import data
import os
import subprocess

__author__ = 'namju.kim@kakaobrain.com'


# data path
_data_path = "/media/srv/data/"

#
# process non-native data
#

def process_non_native(csv_file, passage, data_type):
    # create csv writer
    writer = csv.writer(csv_file,  delimiter=',')
    
    # Get trails of speakers
    parent_path  = _data_path + "non_native_data/wav48/"+ data_type +"/"+ passage
    trail_list = glob.glob(parent_path + '/*')
   
    # add labels
    text_path = _data_path + "non_native_data/txt/" + passage.lower() + '.txt'
    
    text = open(text_path).read()
    sents_n = text.split('.')
    sentences = [line.replace('\n','') for line in sents_n]
   
    labels = [data.str2index(sentence) for sentence in sentences][0:-1]
    
    # add wav files
    for trail in trail_list:
        # wav file names
        wav_file_ids = glob.glob(trail +'/*.wav')
       
        wav_file_ids = [name for name in wav_file_ids if "_000.wav" not in name]
       
        for wav_file, label in zip(wav_file_ids,labels):
            # load wave file
            wave, sr = librosa.load(wav_file, mono=True, sr=None)
            
            # re-sample ( 48K -> 16K )
            wave = wave[::3]
         
            # get mfcc feature
            mfcc = librosa.feature.mfcc(wave, sr=16000)
           
            # filename
            fn = wav_file.split('/')[-1]  
            fn = fn.split('.')[0] +  "_" + passage.lower() + "." + fn.split('.')[-1]
            
            # remove small mfcc files to prevent ctc errors
            if len(label) < mfcc.shape[1]:
                # save meta info
                writer.writerow([fn] + label)
    
                # save mfcc
                if data_type == "train":
                    np.save('/media/srv/data/preprocess/non_native_train_mfcc/' + fn +  '.npy', mfcc, allow_pickle=False)
                elif data_type == "test":
                     np.save('/media/srv/data/preprocess/non_native_test_mfcc/' + fn +  '.npy', mfcc, allow_pickle=False)
 
#
# process non_native corpus according to strategy 1
#

def process_non_native_strategy1(csv_file, passage, data_type):
    # create csv writer
    writer = csv.writer(csv_file,  delimiter=',')
    
   # Get trails of speakers
    parent_path  = _data_path + "non_native_data/wav48/"+ data_type +"/"+ passage
    trail_list = glob.glob(parent_path + '/*')
   
    # add labels
    text_path = _data_path + "non_native_data/txt/" + passage.lower() + '.txt'
    
    text = open(text_path).read()
    sents_n = text.split('.')
    sentences = [line.replace('\n','') for line in sents_n]
   
    labels = [data.str2index(sentence) for sentence in sentences][0:-1]
    
    # add wav files
    for trail in trail_list:
        # wav file names
        wav_file_ids = glob.glob(trail +'/*.wav')
       
        wav_file_ids = [name for name in wav_file_ids if "_000.wav" not in name]
       
        for wav_file, label in zip(wav_file_ids,labels):
            # load wave file
            wave, sr = librosa.load(wav_file, mono=True, sr=None)
            
             # re-sample ( 48K -> 16K )
            wave = wave[::3]
            # Create 4 extra copies
            wav08 = librosa.effects.time_stretch(wave,0.80)
            wav085 = librosa.effects.time_stretch(wave,0.85)
            wave09 = librosa.effects.time_stretch(wave,0.9)
            wave095 = librosa.effects.time_stretch(wave,0.95)
            wave105 = librosa.effects.time_stretch(wave,1.05)
            wave11 = librosa.effects.time_stretch(wave,1.1)
            wav115 = librosa.effects.time_stretch(wave,1.15)
            wav120 = librosa.effects.time_stretch(wave,1.20)
            # get mfcc feature
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            mfcc08 =  librosa.feature.mfcc(wav08, sr=16000)
            mfcc085 =  librosa.feature.mfcc(wav085, sr=16000)
            mfcc09 =  librosa.feature.mfcc(wave09, sr=16000)
            mfcc095 = librosa.feature.mfcc(wave095, sr=16000)
            mfcc105 =  librosa.feature.mfcc(wave105, sr=16000)
            mfcc11 =  librosa.feature.mfcc(wave11, sr=16000)
            mfcc115 =  librosa.feature.mfcc(wav115, sr=16000)
            mfcc120 =  librosa.feature.mfcc(wav120, sr=16000)
            
            mfccs = [("",mfcc),("080",mfcc08),("085",mfcc085),("09",mfcc09),("095",mfcc095),("105",mfcc105),("11",mfcc11),("115",mfcc115),("120",mfcc120)]
           
            for name, mfcc in mfccs:
                 # filename
                fn = wav_file.split('/')[-1] + name
                fn = fn.split('.')[0] +  "_" + passage.lower() + "." + fn.split('.')[-1]
                # remove small mfcc files to prevent ctc errors
                if len(label) < mfcc.shape[1]:
                    # save meta info
                    writer.writerow([fn] + label)
        
                    # save mfcc
                    np.save('/media/srv/data/preprocess/non_native_strategy1_mfcc/' + fn +  '.npy', mfcc, allow_pickle=False)  


def process_non_native_strategy2(csv_file, passage, data_type):
    # create csv writer
    writer = csv.writer(csv_file,  delimiter=',')
    
   # Get trails of speakers
    parent_path  = _data_path + "non_native_data/wav48/"+ data_type +"/"+ passage
    trail_list = glob.glob(parent_path + '/*')
   
    # add labels
    text_path = _data_path + "non_native_data/txt/" + passage.lower() + '.txt'
    
    text = open(text_path).read()
    sents_n = text.split('.')
    sentences = [line.replace('\n','') for line in sents_n]
   
    labels = [data.str2index(sentence) for sentence in sentences][0:-1]
    
    # add wav files
    for trail in trail_list:
        # wav file names
        wav_file_ids = glob.glob(trail +'/*.wav')
       
        wav_file_ids = [name for name in wav_file_ids if "_000.wav" not in name]
       
        for wav_file, label in zip(wav_file_ids,labels):
            # load wave file
            wave, sr = librosa.load(wav_file, mono=True, sr=None)
            
             # re-sample ( 48K -> 16K )
            wave = wave[::3]
            # Add Uniform noise
            wave_uniform = wave + np.random.uniform(size=wave.shape)
            # Add laplace noise
            wave_laplace = wave + np.random.laplace(size = wave.shape)
            wave_normal = wave + np.random.normal(size = wave.shape)
            # get mfcc feature
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            mfcc_uniform =  librosa.feature.mfcc(wave_uniform, sr=16000)
            mfcc_laplace =  librosa.feature.mfcc(wave_laplace, sr=16000)
            mfcc_normal =  librosa.feature.mfcc(wave_normal, sr=16000)
            
            mfccs = [("",mfcc), ("uniform",mfcc_uniform),("laplace",mfcc_laplace),("normal",mfcc_normal)]
           
            for name, mfcc in mfccs:
                 # filename
                fn = wav_file.split('/')[-1] + name
                fn = fn.split('.')[0] +  "_" + passage.lower() + "." + fn.split('.')[-1]
                # remove small mfcc files to prevent ctc errors
                if len(label) < mfcc.shape[1]:
                    # save meta info
                    writer.writerow([fn] + label)
        
                    # save mfcc
                    np.save('/media/srv/data/preprocess/non_native_strategy2_mfcc/' + fn +  '.npy', mfcc, allow_pickle=False)   
                    
def process_non_native_strategy3(csv_file, passage, data_type):
    # create csv writer
    writer = csv.writer(csv_file,  delimiter=',')
    
   # Get trails of speakers
    parent_path  = _data_path + "non_native_data/wav48/"+ data_type +"/"+ passage
    trail_list = glob.glob(parent_path + '/*')
   
    # add labels
    text_path = _data_path + "non_native_data/txt/" + passage.lower() + '.txt'
    
    text = open(text_path).read()
    sents_n = text.split('.')
    sentences = [line.replace('\n','') for line in sents_n]
   
    labels = [data.str2index(sentence) for sentence in sentences][0:-1]
    
    # add wav files
    for trail in trail_list:
        # wav file names
        wav_file_ids = glob.glob(trail +'/*.wav')
       
        wav_file_ids = [name for name in wav_file_ids if "_000.wav" not in name]
       
        for wav_file, label in zip(wav_file_ids,labels):
            # load wave file
            wave, sr = librosa.load(wav_file, mono=True, sr=None)
            
             # re-sample ( 48K -> 16K )
            wave = wave[::3]
            # Add Uniform noise
            wave_normal = wave + np.random.normal(size=wave.shape)
          
            wave_normal = wave + np.random.normal(size = wave.shape)
            wave_normal09 = librosa.effects.time_stretch(wave_normal,0.9)
            wave_normal095 = librosa.effects.time_stretch(wave_normal,0.95)
            wave_normal105 = librosa.effects.time_stretch(wave_normal,1.05)
            # get mfcc feature
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            mfcc_normal90 =  librosa.feature.mfcc(wave_normal09, sr=16000)
            mfcc_normal95 =  librosa.feature.mfcc(wave_normal095, sr=16000)
            mfcc_normal105 =  librosa.feature.mfcc(wave_normal105, sr=16000)
            
            mfccs = [("",mfcc), ("normal09",mfcc_normal90),("normal095",mfcc_normal95),("normal105",mfcc_normal105)]
    
            for name, mfcc in mfccs:
                 # filename
                fn = wav_file.split('/')[-1] + name
                fn = fn.split('.')[0] +  "_" + passage.lower() + "." + fn.split('.')[-1]
                # remove small mfcc files to prevent ctc errors
                if len(label) < mfcc.shape[1]:
                    # save meta info
                    writer.writerow([fn] + label)
        
                    # save mfcc
                    np.save('/media/srv/data/preprocess/non_native_strategy3_mfcc/' + fn +  '.npy', mfcc, allow_pickle=False)    
#
# process VCTK corpus
#

def process_vctk(csv_file):

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read label-info
    df = pd.read_table(_data_path + 'VCTK-Corpus/speaker-info.txt', usecols=['ID'],
                       index_col=False, delim_whitespace=True)

    # read file IDs
    file_ids = []
    for d in [_data_path + 'VCTK-Corpus/txt/p%d/' % uid for uid in df.ID.values]:
        file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

    for i, f in enumerate(file_ids):

        # wave file name
        wave_file = _data_path + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
        fn = wave_file.split('/')[-1]
        target_filename = '/media/srv/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            continue
        # print info
        print("VCTK corpus preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))

        # load wave file
        wave, sr = librosa.load(wave_file, mono=True, sr=None)

        # re-sample ( 48K -> 16K )
        wave = wave[::3]

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # get label index
        label = data.str2index(open(_data_path + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt').read())

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # save meta info
            writer.writerow([fn] + label)
            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)


#
# process LibriSpeech corpus
#

def process_libri(csv_file, category):

    parent_path = _data_path + 'LibriSpeech/' + category + '/'
    labels, wave_files = [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read directory list by speaker
    speaker_list = glob.glob(parent_path + '*')
    for spk in speaker_list:

        # read directory list by chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # read label text file list
            txt_list = glob.glob(chap + '/*.txt')
            for txt in txt_list:
                with open(txt, 'rt') as f:
                    records = f.readlines()
                    for record in records:
                        # parsing record
                        field = record.split('-')  # split by '-'
                        speaker = field[0]
                        chapter = field[1]
                        field = field[2].split()  # split field[2] by ' '
                        utterance = field[0]  # first column is utterance id

                        # wave file name
                        wave_file = parent_path + '%s/%s/%s-%s-%s.flac' % \
                                                  (speaker, chapter, speaker, chapter, utterance)
                        wave_files.append(wave_file)

                        # label index
                        labels.append(data.str2index(' '.join(field[1:])))  # last column is text label

    # save results
    for i, (wave_file, label) in enumerate(zip(wave_files, labels)):
        fn = wave_file.split('/')[-1]
        target_filename = '/media/srv/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            continue
        # print info
        print("LibriSpeech corpus preprocessing (%d / %d) - '%s']" % (i, len(wave_files), wave_file))

        # load flac file
        wave, sr, _ = scikits.audiolab.flacread(wave_file)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # filename

            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)


#
# process TEDLIUM corpus
#
def convert_sph( sph, wav ):
    """Convert an sph file into wav format for further processing"""
    command = [
        'sox','-t','sph', sph, '-b','16','-t','wav', wav
    ]
    subprocess.check_call( command ) # Did you install sox (apt-get install sox)

def process_ted(csv_file, category):

    parent_path = _data_path + 'TEDLIUM_release2/' + category + '/'
    labels, wave_files, offsets, durs = [], [], [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read STM file list
    stm_list = glob.glob(parent_path + 'stm/*')
    for stm in stm_list:
        with open(stm, 'rt') as f:
            records = f.readlines()
            for record in records:
                field = record.split()

                # wave file name
                wave_file = parent_path + 'sph/%s.sph.wav' % field[0]
                wave_files.append(wave_file)

                # label index
                labels.append(data.str2index(' '.join(field[6:])))

                # start, end info
                start, end = float(field[3]), float(field[4])
                offsets.append(start)
                durs.append(end - start)

    # save results
    for i, (wave_file, label, offset, dur) in enumerate(zip(wave_files, labels, offsets, durs)):
        fn = "%s-%.2f" % (wave_file.split('/')[-1], offset)
        target_filename = '/media/srv/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            continue
        # print info
        print("TEDLIUM corpus preprocessing (%d / %d) - '%s-%.2f]" % (i, len(wave_files), wave_file, offset))
        # load wave file
        if not os.path.exists( wave_file ):
            sph_file = wave_file.rsplit('.',1)[0]
            if os.path.exists( sph_file ):
                convert_sph( sph_file, wave_file )
            else:
                raise RuntimeError("Missing sph file from TedLium corpus at %s"%(sph_file))
        wave, sr = librosa.load(wave_file, mono=True, sr=None, offset=offset, duration=dur)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # filename

            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)


#
# Create directories
#
if not os.path.exists('/media/srv/data/preprocess'):
    os.makedirs('/media/srv/data/preprocess')
if not os.path.exists('/media/srv/data/preprocess/meta'):
    os.makedirs('/media/srv/data/preprocess/meta')
if not os.path.exists('/media/srv/data/preprocess/mfcc'):
    os.makedirs('/media/srv/data/preprocess/mfcc')


#
# Run pre-processing for training
#

# TRAIN
"""
## non-native corpus Northwind passage
csv_f = open('/media/srv/data/preprocess/meta/non_native_train.csv', 'w')
process_non_native(csv_f,"NorthWind","train")
csv_f.close()
#
## non-native corpus Rainbow passage
#
csv_f = open('/media/srv/data/preprocess/meta/non_native_train.csv', 'a+')
process_non_native(csv_f,"Rainbow","train")
csv_f.close()

#TEST
csv_f = open('/media/srv/data/preprocess/meta/non_native_test.csv', 'w')
process_non_native(csv_f,"NorthWind","test")
csv_f.close()
"""
#Strategy 2
## non-native corpus Northwind passage

csv_f = open('/media/srv/data/preprocess/meta/strategy3.csv', 'w')
process_non_native_strategy3(csv_f,"NorthWind","train")
csv_f.close()
#
## non-native corpus Rainbow passage
#
csv_f = open('/media/srv/data/preprocess/meta/strategy3.csv', 'a+')
process_non_native_strategy3(csv_f,"Rainbow","train")
csv_f.close()

"""
#
## VCTK corpus
#csv_f = open('/media/srv/data/preprocess/meta/train.csv', 'a+')
#process_vctk(csv_f)
#csv_f.close()

# LibriSpeech corpus for train
#csv_f = open('/media/srv/data/preprocess/meta/train.csv', 'a+')
#process_libri(csv_f, 'train-clean-360')
#csv_f.close()

# TEDLIUM corpus for train
#csv_f = open('/media/srv/data/preprocess/meta/train.csv', 'w')
#process_ted(csv_f, 'train')
#csv_f.close()

#csv_f = open('/media/srv/data/preprocess/meta/train_small_batch.csv', 'w')
#process_ted(csv_f, 'mini_train')
#csv_f.close()
#
# Run pre-processing for validation
#

# LibriSpeech corpus for valid
#csv_f = open('/media/srv/data/preprocess/meta/valid.csv', 'w')
#process_libri(csv_f, 'dev-clean')
#csv_f.close()

# TEDLIUM corpus for valid
#csv_f = open('/media/srv/data/preprocess/meta/valid.csv', 'w')
#process_ted(csv_f, 'dev')
#csv_f.close()

#
# Run pre-processing for testing
#

# LibriSpeech corpus for test
#csv_f = open('/media/srv/data/preprocess/meta/test.csv', 'w')
#process_libri(csv_f, 'test-clean')
#csv_f.close()

# TEDLIUM corpus for test
#csv_f = open('/media/srv/data/preprocess/meta/test.csv', 'w')
#process_ted(csv_f, 'test')
#csv_f.close()
"""