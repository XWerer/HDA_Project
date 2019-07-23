from scipy.io.wavfile import write
import pyaudio
import wave
import DataGeneration as dg
import librosa
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features as sf
import Model
import tensorflow as tf
from tkinter import *
from tkinter.ttk import *

tf.enable_eager_execution()
#print(tf.executing_eagerly())
#print(tf.__version__)
#print(tf.keras.__version__)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "prova.wav"
device_index = 2
audio = pyaudio.PyAudio()

# Dictionary containing the mapping between category name and label
DictCategs = {'nine' : 1, 'yes' : 2, 'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 
              'stop' : 10, 'go' : 11, 'zero' : 12, 'one' : 13, 'two' : 14, 'three' : 15, 'four' : 16, 'five' : 17, 
              'six' : 18, 'seven' : 19, 'eight' : 20, 'backward':21, 'bed':22, 'bird':23, 'cat':24, 'dog':25, 'follow':26, 
              'forward':27, 'happy':28, 'house':29, 'learn':30, 'marvin':31, 'sheila':32, 'tree':33, 'visual':34, 'wow': 0}
nCategs = 35

model = Model.NewAttentionModelOurs(nCategs, 100, 80, 64, dropout = 0.0, activation = 'relu')
#model.summary()

#model.compile(optimizer = 'adam',
#              loss = tf.keras.losses.sparse_categorical_crossentropy,
#              metrics = ['sparse_categorical_accuracy'])

model.load_weights('Model/NoisyMultyAttentionDiff-e12-91-91.h5') #noisy
#model.load_weights('Model/MultyAttentionDiff-final.h5') #non noisy

#invert the dictionary
my_inverted_dict = dict(map(reversed, DictCategs.items()))
"""
print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    print ("Input Device id " + str(i) + " - " + audio.get_device_info_by_host_api_device_index(0, i).get('name'))

print("-------------------------------------------------------------")
"""
#index = (int(input()))
index = (14)

#print("recording via index " + str(index))
stream = audio.open(format = FORMAT, channels = CHANNELS,
                    rate = RATE, input = True, input_device_index = index,
                    frames_per_buffer = CHUNK)

# read the first 5 chunk and discard because distorsion of the microphone
data = stream.read(CHUNK)
data = stream.read(CHUNK)
data = stream.read(CHUNK)
data = stream.read(CHUNK)
data = stream.read(CHUNK)
print ("recording started")

Recordframes = [] # the whole reading
# 4 frames to record 
f1 = np.zeros((16000, )) # A python-list of chunks(numpy.ndarray)
f2 = np.zeros((16000, )) # A python-list of chunks(numpy.ndarray)
f3 = np.zeros((16000, )) # A python-list of chunks(numpy.ndarray)
f4 = np.zeros((16000, )) # A python-list of chunks(numpy.ndarray)
acc1 = 0 # counter accomulation 1
acc2 = 0 # counter accomulation 2
acc3 = 0 # counter accomulation 3
acc4 = 0 # counter accomulation 4

#starting the bufferization 
#first chunk f1
data = stream.read(CHUNK)
f1[0:4000] = np.fromstring(data, dtype=np.int16)
#f1.append(np.fromstring(data, dtype=np.int16))
Recordframes.append(data)
#second chunck f1 - f2
data = stream.read(CHUNK)
f1[4000:8000] = np.fromstring(data, dtype=np.int16)
f2[0:4000] = np.fromstring(data, dtype=np.int16)
#f1.append(np.fromstring(data, dtype=np.int16))
#f2.append(np.fromstring(data, dtype=np.int16))
Recordframes.append(data)
#third chunck f1 -f2 -f3
data = stream.read(CHUNK)
f1[8000:12000] = np.fromstring(data, dtype=np.int16)
f2[4000:8000] = np.fromstring(data, dtype=np.int16)
f3[0:4000] = np.fromstring(data, dtype=np.int16)
#f1.append(np.fromstring(data, dtype=np.int16))
#f2.append(np.fromstring(data, dtype=np.int16))
#f3.append(np.fromstring(data, dtype=np.int16))
Recordframes.append(data)

pred = np.zeros((3, ))
value = np.zeros((3, ))
i = 0
cin = 3
trigger = True
#start cycle 
while trigger:
    try: 
        data = stream.read(CHUNK)
        #f1.append(np.fromstring(data, dtype=np.int16))
        #f2.append(np.fromstring(data, dtype=np.int16))
        #f3.append(np.fromstring(data, dtype=np.int16))
        #f4.append(np.fromstring(data, dtype=np.int16))
        f1[12000:] = np.fromstring(data, dtype=np.int16)
        f2[8000:12000] = np.fromstring(data, dtype=np.int16)
        f3[4000:8000] = np.fromstring(data, dtype=np.int16)
        f4[0:4000] = np.fromstring(data, dtype=np.int16)
        Recordframes.append(data)
        
        #Convert the list of numpy-arrays into a 1D array (column-wise)
        #numpydata = np.hstack(f1)
        y3 = sf.base.logfbank(f1, samplerate = 16000, winlen = 0.016, nfilt=80, nfft = 1024, lowfreq = 40, highfreq = 8000, preemph = 0.95)
        #y3 = np.transpose(y3)    
        mean = np.mean(y3, axis=0)
        #print(mean.shape)
        stv = np.std(y3, axis=0)
        #print(stv.shape)
        diff = np.subtract(y3, mean)
        #print(diff.shape)
        y3 = np.divide(diff, stv + 1e-8)
        sol = model.predict(y3.reshape((1, 100, 80, 1)))
        #print(sol*100)
        key = np.argmax(sol)
        #print("Prediction: " + str(key) + " " + str(my_inverted_dict.get(key)))
        pred[i] = key
        value[i] = sol[0, key]
        i = i + 1
        if(i == 3):
            test = True
            for j in range(0, pred.shape[0]):
                if pred[j] != key or value[j] < 0.8:
                    test = False
                    break
            if test: 
                print("Prediction: " + str(key) + " " + str(my_inverted_dict.get(key)) + " " + str(value))
                i = 0
                pred = np.zeros((3, ))
                value = np.zeros((3, ))
            else:
                #print(str(pred) + " " + str(value))
                i = 2
                pred[0] = pred[1]
                pred[1] = pred[2]
                pred[2] = -1
                value[0] = value[1]
                value[1] = value[2]
                value[2] = -1
                       

        f1 = f2
        f2 = f3 
        f3 = f4
        f4 = np.zeros((16000, ))

    except:
        trigger = False


print ("recording stopped")

# close stream
stream.stop_stream()
stream.close()
audio.terminate()

# Generate the wav file
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(Recordframes))
waveFile.close()
"""
signal = np.hstack(Recordframes)
# Plot the wav
#print(np.asarray(signal))
plt.figure()
plt.plot(np.asarray(signal), color='b')
plt.title('WAV signal')
"""