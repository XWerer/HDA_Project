import scipy

from scipy.io.wavfile import read

import python_speech_features as sf

import numpy as np

from addNoise import addNoise2



def computeFeatures(wav_signal_name, desiredLength = 16000, log = True, w_len = 0.025, w_step = 0.01, noise = "dude_miaowing.wav"):

    # input:   a wav audio file (ours are all 1 second long, so 16000 sample long if Fc = 16000)

    # the noise is the one to add to the signal if the signal is not full length

    # output:  a 2D matrix of size (num_frames, 39), where 39 is the number of coefficients of the features vectors

    # data are supposed to be single channel

    # log = True: it means that we take the logarithm of the energies of delta and delta-delta

    

    # read audio sample

    input_signal = read(wav_signal_name)

    # sampling rate

    Fc = input_signal[0]

    # Nyquist critical frequency(highest freq. that can be represented)

    nyqF = int(Fc/2)  #input in sf.base.mfcc() has to be an int

    signal = input_signal[1] # this is the vector-signal we are interested in

    if(signal.shape[0] < desiredLength):

        noise = read(noise)[1]

        signal = addNoise2(signal, noise, begin = True)

    # appendEnergy = True means that the zeroth value of all the cepstral vectors is replaced  with the corresponding frame energy, E(s_i)

    coeffs = sf.base.mfcc(signal, samplerate = Fc, nfft = nyqF, appendEnergy = True, winlen=w_len, winstep=w_step, winfunc=np.hamming)

    useful_coeffs = coeffs[:,1:13] # (taking only the 1,2,...,12 MFCC's)

    # energy(s)

    E_s = coeffs[:,0]

    if(min(E_s) < 0.0001):

        for i in range(E_s.shape[0]):

            if(E_s[i] < 0.0001):

                E_s[i] = 1

    # take the log10 to limit the dynamics

    log_E_s = np.log10(E_s)

    #print("shape of log energies of the frames: " + str(log_E_s.shape))

    #print("shape of useful coefficients: " + str(useful_coeffs.shape))

    

    # extracting deltas

    deltas = sf.base.delta(useful_coeffs, 2)

    #print("deltas = " + str(deltas.shape))

    if(min(delta) < 0.0001):

        for i,l in range(delta.shape[0], delta.shape[1]):

            if(delta[i,l] < 0.0001):

                delta[i,l] = 1

    # extracting deltas of deltas

    deltas_2 = sf.base.delta(deltas, 2)

    #print("deltas_2 = " + str(deltas_2.shape))

    

    # getting energies of deltas and of deltas_2

    E_deltas = np.sum(np.power(deltas, 2), axis = 1)

    log_E_deltas = np.log10(E_deltas)

    #print("shape of log_E_deltas = " + str(log_E_deltas.shape))

    

    # energy of delta of deltas: 

    E_deltas_2 = np.sum(np.power(deltas, 2), axis = 1)

    log_E_deltas_2 = np.log10(E_deltas_2)

    #print("shape of log_E_deltas_2 = " + str(log_E_deltas_2.shape))



    if(log):

        E_d = log_E_deltas

        E_d_d = log_E_deltas_2

    else:

        E_d = E_deltas    

        E_d_d = E_deltas_2



    # preallocating space:

    num_frames = coeffs.shape[0]

    #num_frames = 99

    features = np.zeros((num_frames,39))



    # putting everything inside the features 2D matrix:

    features[0:coeffs.shape[0], 0:12] = useful_coeffs

    features[0:coeffs.shape[0], 12:24] = deltas

    features[0:coeffs.shape[0], 24:36] = deltas_2

    features[0:coeffs.shape[0], 36] = log_E_s

    features[0:coeffs.shape[0], 37] = E_d

    features[0:coeffs.shape[0], 38] = E_d_d

    

    return features





def computeFeatures1(signal, Fc, log = True, w_len = 0.025, w_step = 0.01):

    # input:   audio file, this time is already a vector

    # output:  a 2D matrix of size (num_frames, 39), where 39 is the number of coefficients of the features vectors

    # log = True: it means that we take the logarithm of the energies of delta and delta-delta

 

    # Nyquist critical frequency(highest freq. that can be represented)

    nyqF = int(Fc/2)  #input in sf.base.mfcc() has to be an int

    

    # appendEnergy = True means that the zeroth value of all the cepstral vectors is replaced  with the corresponding frame energy, E(s_i)

    coeffs = sf.base.mfcc(signal, samplerate = Fc, nfft = nyqF, appendEnergy = True, winlen=w_len, winstep=w_step, winfunc=np.hamming)

    useful_coeffs = coeffs[:,1:13] # (taking only the 1,2,...,12 MFCC's)

    # energy(s)

    E_s = coeffs[:,0]

    if(min(E_s) < 0.0001):

        for i in range(E_s.shape[0]):

            if(E_s[i] < 0.0001):

                E_s[i] = 1

    # take the log10 to limit the dynamics

    log_E_s = np.log10(E_s)



    # extracting deltas

    deltas = sf.base.delta(useful_coeffs, 1)

    if(np.min(deltas) < 0.0001):

        for i,l in range(deltas.shape[0], deltas.shape[1]):

            if(deltas[i,l] < 0.0001):

                deltas[i,l] = 1

    # extracting deltas of deltas

    deltas_2 = sf.base.delta(deltas, 1)

    if(np.min(deltas_2) < 0.0001):

        for i,l in range(deltas_2.shape[0], deltas_2.shape[1]):

            if(deltas_2[i,l] < 0.0001):

                deltas_2[i,l] = 1

    

    # getting energies of deltas and of deltas_2

    E_deltas = np.sum(np.power(deltas, 2), axis = 1)

    if(min(E_deltas) < 0.0001):

        for i in range(E_deltas.shape[0]):

            if(E_deltas[i] < 0.0001):

                E_deltas[i] = 1

    log_E_deltas = np.log10(E_deltas)



    # energy of delta of deltas: 

    E_deltas_2 = np.sum(np.power(deltas_2, 2), axis = 1)

    if(min(E_deltas_2) < 0.0001):

        for i in range(E_deltas_2.shape[0]):

            if(E_deltas_2[i] < 0.0001):

                E_deltas_2[i] = 1

    log_E_deltas_2 = np.log10(E_deltas_2)



    if(log):

        E_d = log_E_deltas

        E_d_d = log_E_deltas_2

    else:

        E_d = E_deltas    

        E_d_d = E_deltas_2



    # preallocating space:

    #num_frames = coeffs.shape[0]

    num_frames = 99

    features = np.zeros((num_frames,39))



    # putting everything inside the features 2D matrix:

    features[0:coeffs.shape[0], 0:12] = useful_coeffs

    features[0:coeffs.shape[0], 12:24] = deltas

    features[0:coeffs.shape[0], 24:36] = deltas_2

    features[0:coeffs.shape[0], 36] = log_E_s

    features[0:coeffs.shape[0], 37] = E_d

    features[0:coeffs.shape[0], 38] = E_d_d

    

    return features





    

    

    