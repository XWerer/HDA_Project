"""
A generator for create the datasets for the models
                   
"""

import pandas as pd
import numpy as np
import keras
from extractMFCC import computeFeatures1
import python_speech_features as sf
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
from addNoise import addNoise

class DataGeneration:
    """
    Generates data for Keras
    
    """
    def __init__(self, dataset_dir, categories, nCat):
        'Initialization'
        self.dataset_dir = dataset_dir
        self.categories = categories
        self.nCat = nCat
        self.test = None
        self.val = None
        self.train = None
        self.test_labels = None
        self.val_labels = None
        self.train_labels = None
        
    def _getFileCategory(self, file):
        # Receives a file with name <cat>/<filename> and returns an integer that is catDict[cat]
        categ = os.path.basename(os.path.dirname(file))
        return self.categories.get(categ, 0)

    #Function to create the numpy array that contain data file and labels
    def create_data(self, test_list, val_list):
        testWAVs = pd.read_csv(self.dataset_dir + test_list, sep=" ", header = None)[0].tolist()
        valWAVs  = pd.read_csv(self.dataset_dir + val_list, sep=" ", header = None)[0].tolist()
        
        allWAVs  = []
        for root, dirs, files in os.walk(self.dataset_dir):
            for f in files:
                if (root != self.dataset_dir + "_background_noise_") and (f.endswith('.wav')):
                    path = root + "/" + f
                    #print(path)
                    path = path[len(self.dataset_dir):]
                    #print(path)
                    allWAVs.append(path)

        # Remove from the training set the elements present in test and validation
        trainWAVs = list(set(allWAVs) - set(valWAVs) - set(testWAVs))
        
        #Get the labels 
        testWAVlabels = [self._getFileCategory(f) for f in testWAVs]
        valWAVlabels = [self._getFileCategory(f) for f in valWAVs]
        trainWAVlabels = [self._getFileCategory(f) for f in trainWAVs]
        
        train = np.array(trainWAVs, dtype = object)
        train_labels = np.array(trainWAVlabels, dtype = '>i4') #stands for int32

        print("Loading of the train set:")
        for i in range(len(trainWAVs)):
            # Print the progress 
            if (i % 5000) == 0:
                print(str(i) + '/' + str(len(trainWAVs)))

            # If the file is not already present, we create the numpy version 
            if (not os.path.isfile(self.dataset_dir + "/" + trainWAVs[i] + '.npy')):
                y, sr = librosa.load(self.dataset_dir + "/" + trainWAVs[i], sr = 16000)
                np.save(self.dataset_dir + "/" + trainWAVs[i] + '.npy', y)

            # We load the path to numpy array in a vector 
            train[i] = trainWAVs[i] + '.npy'

        print(str(i+1) + '/' + str(len(trainWAVs)))
        
        # Do the same thing for the validation and the test set
        val = np.array(valWAVs, dtype = object)
        val_labels = np.array(valWAVlabels, dtype = '>i4') #stands for int32

        print("Loading of the validation set:")
        for i in range(len(valWAVs)):
            # Print the progress 
            if (i % 5000) == 0:
                print(str(i) + '/' + str(len(valWAVs)))

            # If the file is not already present, we create the numpy version 
            if (not os.path.isfile(self.dataset_dir + "/" + valWAVs[i] + '.npy')):
                y, sr = librosa.load(self.dataset_dir + "/" + valWAVs[i], sr = 16000)
                np.save(self.dataset_dir + "/" + valWAVs[i] + '.npy', y)

            # We load the path to numpy array in a vector 
            val[i] = valWAVs[i] + '.npy'

        print(str(i+1) + '/' + str(len(valWAVs)))

        test = np.array(testWAVs, dtype = object)
        test_labels = np.array(testWAVlabels, dtype = '>i4') #stands for int32
        
        print("Loading of the test set:")
        for i in range(len(testWAVs)):
            # Print the progress 
            if (i % 5000) == 0:
                print(str(i) + '/' + str(len(testWAVs)))

            # If the file is not already present, we create the numpy version 
            if (not os.path.isfile(self.dataset_dir + "/" + testWAVs[i] + '.npy')):
                y, sr = librosa.load(self.dataset_dir + "/" + testWAVs[i], sr = 16000)
                np.save(self.dataset_dir + "/" + testWAVs[i] + '.npy', y)

            # We load the path to numpy array in a vector 
            test[i] = testWAVs[i] + '.npy' 

        print(str(i+1) + '/' + str(len(testWAVs)))
        
        return train, train_labels, val, val_labels, test, test_labels
        
    # Function to load numpy array
    def load_data(dataset_dir, file_name):
        # Load the wav signal from the .npy file
        data = np.load(dataset_dir + file_name)
        return data
    
    # Function to preprocess the data 
    def load_and_preprocess_data(dataset_dir, file_name, noisy = False):
        # Required by tensorflow (strings are passed as bytes)
        if type(file_name) is bytes:
            file_name = file_name.decode()
            dataset_dir = dataset_dir.decode()

        # Load data
        data = np.zeros((16000,))
        d = DataGeneration.load_data(dataset_dir, file_name)
        data[:d.shape[0]] = d
        if noisy:
            noise = load_data('_background_noise_/white_noise.wav.npy', dataset_dir)
            data = addNoise1(data, noise, intensity = 0.1) 
        #feats = computeFeatures1(data, 16000)
        # Normalize
        #feats -= np.mean(feats, axis=0)
        #mean = np.mean(feats, axis = 0)
        #stv = np.std(feats, axis = 0)
        #diff = np.subtract(feats, mean)
        #feats = np.divide(diff, stv + 1e-8)
        #feats = np.divide(feats, np.max(feats))
        #diff1 = np.subtract(feats, np.amin(feats, axis=0))
        #print(np.amin(feats, axis=0))
        #print(np.amax(feats, axis=0))
        #diff2 = np.subtract(np.amax(feats, axis=0), np.amin(feats, axis=0))
        #feats = np.divide(diff1, diff2+1e-6)
        y3 = sf.base.logfbank(data, samplerate = 16000, winlen = 0.016, nfilt=80, nfft = 1024, lowfreq = 40, highfreq = 8000, preemph = 0.95)
        #y3 = np.transpose(y3)    
        mean = np.mean(y3, axis=0)
        #print(mean.shape)
        stv = np.std(y3, axis=0)
        #print(stv.shape)
        diff = np.subtract(y3, mean)
        #print(diff.shape)
        y3 = np.divide(diff, stv + 1e-8)

        return y3.astype(np.float32)
    
    # new dataset for the classifier 
    def create_dataset(dataset_dir, file_names, labels, batch_size = 32, shuffle = True, cache_file = None):

        # Create a Dataset object
        dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))

        # Map the load_and_preprocess_data function
        py_func = lambda file_name, label: (tf.numpy_function(DataGeneration.load_and_preprocess_data, 
                                                              [dataset_dir, file_name, False], 
                                                              tf.float32), 
                                            label)
        
        dataset = dataset.map(py_func, num_parallel_calls = os.cpu_count())

        # Cache dataset
        if cache_file:
            dataset = dataset.cache(cache_file)

        # Shuffle    
        if shuffle:
            dataset = dataset.shuffle(len(file_names), reshuffle_each_iteration = True)

        # Repeat the dataset indefinitely
        dataset = dataset.repeat()

        # Correct input shape for the network
        dataset = dataset.map(lambda data, label: (tf.reshape(data, shape=(100, 80, 1)), tf.reshape(label, shape=(1, ))))

        # Batch
        dataset = dataset.batch(batch_size = batch_size)

        # Prefetch (1 means that prefetch a batch at time)
        dataset = dataset.prefetch(buffer_size = 1)

        return dataset

"""
#Prova classe
# Root folder of the dataset
dataset_dir = "Dataset/"
val_file = 'validation_list.txt'
test_file = 'testing_list.txt'

# Dictionary containing the mapping between category name and label
DictCategs = {'nine' : 1, 'yes' : 2, 'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 
              'stop' : 10, 'go' : 11, 'zero' : 12, 'one' : 13, 'two' : 14, 'three' : 15, 'four' : 16, 'five' : 17, 
              'six' : 18, 'seven' : 19, 'eight' : 20, 'backward':21, 'bed':22, 'bird':23, 'cat':24, 'dog':25, 'follow':26, 
              'forward':27, 'happy':28, 'house':29, 'learn':30, 'marvin':31, 'sheila':32, 'tree':33, 'visual':34, 'wow':0,
              '_background_noise_':0 }
nCategs = 35

data = DataGeneration(dataset_dir, DictCategs, nCategs)
train, trainLabels, val, valLabels, test, testLabels = data.create_data(test_file, val_file)

index = 124
# Plot a wav
file_name = train[index]
wav = DataGeneration.load_data(file_name)

# example:
feats = DataGeneration.load_and_preprocess_data(train[index], False)

batch_size = 256

train_dataset = DataGeneration.create_dataset(train, trainLabels, batch_size = batch_size, shuffle = True, cache_file = 'train_prova_classe')

val_dataset = DataGeneration.create_dataset(val, valLabels, batch_size = batch_size, shuffle = False, cache_file = 'val_prova_classe')

test_dataset = DataGeneration.create_dataset(test, testLabels, batch_size = batch_size, shuffle = False, cache_file = 'test_prova_classe')

train_steps = int(np.ceil(len(train) / batch_size))
val_steps = int(np.ceil(len(val) / batch_size))
test_steps = int(np.ceil(len(test) / batch_size))

print("steps to completa a train epoch: " + str(train_steps))
print("steps to completa a validation spoch: " + str(val_steps))
print("steps to completa a test epoch: " + str(test_steps))
"""