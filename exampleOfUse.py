from extractMFCC import computeFeatures, computeFeatures1
from addNoise import addNoise

# NOTE: it is possible to specify different window sizes and steps in the inputs

# getting features from a clean audio, input is a name
features = computeFeatures("00b01445_nohash_0.wav", log = True)

# getting a noisy version of the previous signal 
noisySignal, Fc = addNoise("00b01445_nohash_0.wav", "doing_the_dishes.wav", randomized = True, intensity = 2, toWrite = True)

# getting features of the noisy signal, using computeFeatures1, input are: a vector and the sampling frequency
features1 = computeFeatures1(noisySignal, Fc, log = True)

print(features.shape)
print(features1.shape)
