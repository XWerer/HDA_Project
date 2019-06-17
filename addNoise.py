import scipy
from scipy.io.wavfile import read, write
import python_speech_features as sf
import numpy as np

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
    if(toWrite):
        write(outputName, Fc_s, noisy)

    return noisy














    


