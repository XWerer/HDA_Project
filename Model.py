# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax
from tensorflow.keras.layers import Add, Dropout, BatchNormalization, Conv2D, Reshape, MaxPooling2D, Dense, Bidirectional#, Attention, CuDNNLSTM
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import optimizers

#kapre for Mel Coefficient 
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

#print(tf.__version__)
#print(tf.keras.__version__)

# Model with the attention layer 
def AttentionModel(nCategories, samplingrate = 16000, inputLength = 16000):
    
    inputs = Input((samplingrate,))

    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                             padding='same', sr=samplingrate, n_mels=80,
                             fmin=40.0, fmax=samplingrate/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)

    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs

    x = Permute((2,1,3)) (x)

    x = Conv2D(10, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(1, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    """ If we have GPU
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    """

    x = Bidirectional(LSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(LSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:,64]) (x) #[b_s, vec_dim] take the central element of the sequence
    query = Dense(128) (xFirst)            # Project the element to a dense layer this allow the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    # Now use the Attention layer (not find when compile, i don't know why)
    # attVector = Attention()([query, x])

    x = Dense(64, activation = 'relu')(attVector)
    x = Dense(32)(x)

    output = Dense(nCategories, activation = 'softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model CNN/RNN Encoder Decoder
def Seq2SeqModel(nCategories, samplingrate = 16000, inputLength = 16000):
    
    inputs = Input((samplingrate,))

    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                             padding='same', sr=samplingrate, n_mels=80,
                             fmin=40.0, fmax=samplingrate/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)

    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs

    x = Permute((2,1,3)) (x)
    