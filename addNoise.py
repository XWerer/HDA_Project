import scipy
from scipy.io.wavfile import read, write
import python_speech_features as sf
import numpy as np
#import extractMFCC

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


"""
#simple code to try addNoise2:
input_signal = read("fromV2bis.wav")
print(input_signal[1].shape)
noise_signal = read("dude_miaowing.wav")
Fc_s = input_signal[0]
signal = input_signal[1]
noise = noise_signal[1]

sign_full = addNoise2(signal, noise, begin = True, intensity = 1)
write("noisy_output.wav", Fc_s, sign_full)

print(sign_full.shape)
"""








    


