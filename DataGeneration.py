"""
A generator for create the datasets for the models
                   
"""
import librosa 
import pandas as pd
import numpy as np
import keras
#from extractMFCC import computeFeatures1
import python_speech_features as sf
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import random

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
                #if (f.endswith('.wav')):
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
    
    def addNoise(wav_signal_name, wav_noise_name, outputName = "noisy_output.wav", start = 0, randomized = False, intensity = 1, toWrite = False):
        # adds noise to a signal
        # same sample frequency for both signals is needed
        # inputs: signal to make a noisy one, noise signal, name of the output, intensity is how much amplified we want the noise to be.
        # if randomized == True, the part of noise signal to be added is chosen randomly
        # start = where(which sample) to start getting the noise from the noise signal
        # output: noisy signal

        input_signal = read(wav_signal_name)
        noise_signal = read(wav_noise_name)

        Fc_s = input_signal[0]
        signal = input_signal[1]

        Fc_n = noise_signal[0]
        noise = noise_signal[1]

        length = signal.shape[0]  # being the audios 1 second long it should be always equal to Fc_s

        if(Fc_n != Fc_s):
            raise ValueError("different sample rates")
        h = noise.shape[0]-length # max index to start taking the noise signal
        if(start > h):
            raise ValueError("start exceeds array bounds")

        if(randomized):
            # choose start randomly in the allowed range
            start = np.random.randint(0, high = h)

        noise = noise[start:start + length]
        # control how much noise to add
        additive_noise = (np.multiply(intensity, noise))

        noisy = np.add(signal, additive_noise);
        if(toWrite):
            write(outputName, Fc_s, noisy)

        return noisy, Fc_s


        # this version takes in input numpy vectors instead of .wav files
    def addNoise1(signal, noise, start = 0, randomized = False, intensity = 1):
        # adds noise to a signal
        # same sample frequency for both signals is needed
        # inputs: signal to make a noisy one, noise signal, name of the output, intensity is how much amplified we want the noise to be.
        # if randomized == True, the part of noise signal to be added is chosen randomly
        # start = where(which sample) to start getting the noise from the noise signal
        # output: noisy signal

        length = signal.shape[0]  # being the audios 1 second long it should be always equal to Fc_s

        h = noise.shape[0]-length # max index to start taking the noise signal
        if(start > h):
            raise ValueError("start exceeds array bounds")

        if(randomized):
            # choose start randomly in the allowed range
            start = np.random.randint(0, high = h)

        noise = noise[start:start + length]
        # control how much noise to add
        additive_noise = (np.multiply(intensity, noise))

        noisy = np.add(signal, additive_noise);

        #write("noisy_output.wav", Fc_s, noisy)

        return noisy

    def addNoise2(signal, noise, desiredLength = 16000, intensity = 1, begin = False):
        # adds noise to the empty part of a signal (the part to let it be of one second)
        # same sample frequency for both signals is needed
        # inputs: signal to fulfill with noise, noise signal, desiredLength = 16000(one second with sampleRate = 16kHz), intensity is fixed to one in this case
        # output: full-length signal

        tmpLength = signal.shape[0]
        lengthToCover = desiredLength - tmpLength
        # it's 112000 since I want the 8th second, that corresponds to silence
        noise = noise[112000:112000 + lengthToCover]

        # control how much noise to add
        # additive_noise = np.multiply(intensity, noise)

        #signal_full = np.ones((desiredLength,), dtype = float)

        if(begin):
            signal_full = np.hstack((noise, signal))
            #signal_full[lengthToCover:desiredLength] = signal
            #signal_full[0:lengthToCover] = noise
        else:
            signal_full = np.hstack((signal, noise))
            #signal_full[0:tmpLength] = signal
            #signal_full[tmpLength:desiredLength] = noise

        return signal_full
        
    # Function to load numpy array
    def load_data(dataset_dir, file_name):
        # Required by tensorflow (strings are passed as bytes)
        if type(file_name) is bytes:
            file_name = file_name.decode()
        if type(dataset_dir) is bytes:
            dataset_dir = dataset_dir.decode()
            
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
    
    def preprocesing_noisy(signal, dataset_dir):
        #padding part
        data = np.zeros((16000,))
        data[:signal.shape[0]] = signal
        
        #noising part (randomized)
        r = random.randint(0, 5)
        if(r == 0):
            noise = DataGeneration.load_data(dataset_dir, '_background_noise_/doing_the_dishes.wav.npy')
        if(r == 1):
            noise = DataGeneration.load_data(dataset_dir, '_background_noise_/dude_miaowing.wav.npy')
        if(r == 2):
            noise = DataGeneration.load_data(dataset_dir, '_background_noise_/exercise_bike.wav.npy')
        if(r == 3):
            noise = DataGeneration.load_data(dataset_dir, '_background_noise_/pink_noise.wav.npy')
        if(r == 4):
            noise = DataGeneration.load_data(dataset_dir, '_background_noise_/running_tap.wav.npy')
        if(r == 5):
            noise = DataGeneration.load_data(dataset_dir, '_background_noise_/white_noise.wav.npy')
           
        #compute features
        data = DataGeneration.addNoise1(data, noise, randomized = True, intensity = 0.1)
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
        dataset = dataset.map(lambda data, label: ((tf.expand_dims(data, -1), label)))

        # Batch
        dataset = dataset.batch(batch_size = batch_size)

        # Prefetch (1 means that prefetch a batch at time)
        dataset = dataset.prefetch(buffer_size = 1)

        return dataset
    
    # new dataset for the classifier 
    def create_dataset_noisy(dataset_dir, file_names, labels, batch_size = 32, shuffle = True, cache_file = None):

        # Create a Dataset object
        dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))

        # Map the load_and_preprocess_data function
        py_func = lambda file_name, label: (tf.numpy_function(DataGeneration.load_data, 
                                                              [dataset_dir, file_name], 
                                                              tf.float32), 
                                            label)
        
        dataset = dataset.map(py_func, num_parallel_calls = os.cpu_count())

        # Cache dataset
        if cache_file:
            dataset = dataset.cache(cache_file)

        # Shuffle    
        if shuffle:
            dataset = dataset.shuffle(len(file_names), reshuffle_each_iteration = True)
            
        # add noise
        py_func = lambda signal, label: (tf.numpy_function(DataGeneration.preprocesing_noisy, 
                                                              [signal, dataset_dir], tf.float32), 
                                            label)
        
        dataset = dataset.map(py_func, num_parallel_calls = os.cpu_count())

        # Repeat the dataset indefinitely
        dataset = dataset.repeat()

        # Correct input shape for the network
        dataset = dataset.map(lambda data, label: ((tf.expand_dims(data, -1), label)))

        # Batch
        dataset = dataset.batch(batch_size = batch_size)

        # Prefetch (1 means that prefetch a batch at time)
        dataset = dataset.prefetch(buffer_size = 1)

        return dataset
    
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, 
                              filename = 'cm.png'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)

        plt.figure(figsize=(22, 22))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=20)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=7)
        plt.yticks(tick_marks, classes, fontsize=7)

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), size=11,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.savefig(filename, dpi = 400)
        plt.tight_layout()