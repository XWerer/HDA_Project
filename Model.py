# TensorFlow and tf.keras
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax, RepeatVector, Multiply
from tensorflow.keras.layers import Add, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Layer
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU
from tensorflow.keras import initializers as initializers, regularizers, constraints
#from keras.layers import Attention, CuDNNLSTM
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import optimizers

#kapre for Mel Coefficient 
#from kapre.time_frequency import Melspectrogram, Spectrogram
#from kapre.utils import Normalization2D
#from keras import regularizers

import numpy as np

#print(tf.__version__)
#print(tf.keras.__version__)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

# Model with the attention layer 
def SimpleModel(nCategories, nTime, nMel, use_GRU = False, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1)) # it's the dimension after the extraction of the mel coefficients

    #inputs = Input((16000,))

    """ We need to drop out this part and compute by hand the mel coefficient
        because this part user keras without tensorflow and there is a bug that
        create problem 
    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 16000),
                             padding='same', sr=16000, n_mels=80,
                             fmin=40.0, fmax=16000/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)
    
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
    """
    #x = Permute((2,1,3)) (x)

    # Two 2D convolutional layer to extract features  
    x = Conv2D(10, (7,7) , activation=activation) (inputs) 
    x = MaxPooling2D((3, 3)) (x)
    x = BatchNormalization() (x)
    x = Conv2D(1, (5,5) , activation=activation) (x)
    x = MaxPooling2D((3, 3)) (x)
    x = BatchNormalization() (x)
    
    x = Bidirectional(GRU(64, return_sequences = True, 
                              #dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]

    x = Flatten() (x)
    x = Dropout(dropout) (x)
    # Two dense layer 
    x = Dense(128, activation = activation)(x)
    
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModel(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    #inputs = Input((16000,))

    """ We need to drop out this part and compute by hand the mel coefficient
        because this part user keras without tensorflow and there is a bug that
        create problem 
    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 16000),
                             padding='same', sr=16000, n_mels=80,
                             fmin=40.0, fmax=16000/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)
    
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
    """
    #x = Permute((2,1,3)) (x)

    # Two 2D convolutional layer to extract features  
    x = Conv2D(10, (7,7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (5,5) , activation=activation, padding='same', 
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    """ If we have GPU
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    """
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        #x = Bidirectional(GRU(unit, return_sequences = True, 
                              #dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    #xFirst = Lambda(lambda q: q[:, (tf.shape(q)[1])/2]) (x)   #[b_s, vec_dim] take the central element of the sequence
    #query = Dense(unit*2) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    #attScores = Dot(axes=[1,2])([query, x]) 
    #attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    #attScores = Dropout(dropout) (attScores)
    #rescale sequence
    #attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    #attVector = Dropout(dropout) (attVector)
    x = Flatten() (x)
    x = Dropout(dropout) (x)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified1(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    #inputs = Input((16000,))

    """ We need to drop out this part and compute by hand the mel coefficient
        because this part user keras without tensorflow and there is a bug that
        create problem 
    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 16000),
                             padding='same', sr=16000, n_mels=80,
                             fmin=40.0, fmax=16000/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)
    
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
    """
    #x = Permute((2,1,3)) (x)

    # Two 2D convolutional layer to extract features  
    x = Conv2D(10, (7,7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (5,5) , activation=activation, padding='same', 
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    """ If we have GPU
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    """
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        #x = Bidirectional(GRU(unit, return_sequences = True, 
                              #dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    #xFirst = Lambda(lambda q: q[:, (tf.shape(q)[1])/2]) (x)   #[b_s, vec_dim] take the central element of the sequence
    #query = Dense(unit*2) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    #attScores = Dot(axes=[1,2])([query, x]) 
    #attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    #attScores = Dropout(dropout) (attScores)
    #rescale sequence
    #attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    #attVector = Dropout(dropout) (attVector)
    x = Flatten() (x)
    x = Dropout(dropout) (x)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified2(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    #inputs = Input((16000,))

    """ We need to drop out this part and compute by hand the mel coefficient
        because this part user keras without tensorflow and there is a bug that
        create problem 
    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 16000),
                             padding='same', sr=16000, n_mels=80,
                             fmin=40.0, fmax=16000/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)
    
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
    """
    #x = Permute((2,1,3)) (x)

    # Two 2D convolutional layer to extract features  
    x = Conv2D(10, (7,7) , activation=activation,
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = MaxPooling2D((3, 3)) (x)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (5,5) , activation=activation, 
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = BatchNormalization(trainable=False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        #x = Bidirectional(GRU(unit, return_sequences = True, 
                              #dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    #xFirst = Lambda(lambda q: q[:, (tf.shape(q)[1])/2]) (x)   #[b_s, vec_dim] take the central element of the sequence
    #query = Dense(unit*2) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    #attScores = Dot(axes=[1,2])([query, x]) 
    #attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    #attScores = Dropout(dropout) (attScores)
    #rescale sequence
    #attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    #attVector = Dropout(dropout) (attVector)
    x = Flatten() (x)
    x = Dropout(dropout) (x)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified3(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    #inputs = Input((16000,))

    """ We need to drop out this part and compute by hand the mel coefficient
        because this part user keras without tensorflow and there is a bug that
        create problem 
    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 16000),
                             padding='same', sr=16000, n_mels=80,
                             fmin=40.0, fmax=16000/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)
    
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
    """
    #x = Permute((2,1,3)) (x)

    # Two 2D convolutional layer to extract features  
    x = Conv2D(10, (5,5) , activation=activation,
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (5,5) , activation=activation, 
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = BatchNormalization(trainable=False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:, 14]) (x)   #[b_s, vec_dim] take the central element of the sequence
    query = Dense(unit*2) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    attScores = Dropout(dropout) (attScores)
    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    #attVector = Dropout(dropout) (attVector)
    x = Dropout(dropout) (attVector)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified4(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    # Two 2D convolutional layer to extract features  
    x = Conv2D(1, (7,7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    #x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    #x = Conv2D(1, (5,5) , activation=activation, 
     #          kernel_regularizer = None, 
      #         bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    #x = BatchNormalization(trainable=False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        #x = Bidirectional(GRU(unit, return_sequences = True, 
         #                     dropout=dropout, recurrent_dropout=dropout, 
          #                    kernel_regularizer = None, 
           #                   activity_regularizer = None,
            #                  bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:, 50]) (x)   #[b_s, vec_dim] take the central element of the sequence
    query = Dense(unit*2) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    attScores = Dropout(dropout) (attScores)
    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    #attVector = Dropout(dropout) (attVector)
    x = Dropout(dropout) (attVector)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified5(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    #x = Permute((2,1,3)) (x)

    # Two 2D convolutional layer to extract features  
    x = Conv2D(3, (30, 10) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (25,5) , activation=activation, padding='same', 
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = MaxPooling2D((1, 2)) (x)
    x = BatchNormalization(trainable=False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        #x = Bidirectional(GRU(unit, return_sequences = True, 
         #                     dropout=dropout, recurrent_dropout=dropout, 
          #                    kernel_regularizer = None, 
           #                   activity_regularizer = None,
            #                  bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:, 20]) (x)   #[b_s, vec_dim] take the central element of the sequence
    query = Dense(unit*2) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    attScores = Dropout(dropout) (attScores)
    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    attVector = Dropout(dropout) (attVector)
    #x = Flatten()(x)
    #x = Dropout(dropout) (x)
    # Two dense layer 
    x = Dense(64, activation = activation)(attVector)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified6(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    # Two 2D convolutional layer to extract features  
    x = Conv2D(3, (7,7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (7,7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = MaxPooling2D((1, 2)) (x)
    x = BatchNormalization(trainable=False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:, 99]) (x)   #[b_s, vec_dim] take the central element of the sequence
    query = Dense(unit*2) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    attScores = Dropout(dropout) (attScores)
    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    #attVector = Dropout(dropout) (attVector)
    x = Dropout(dropout) (attVector)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified7(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    # Two 2D convolutional layer to extract features  
    x = Conv2D(3, (7, 7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (5, 5) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = MaxPooling2D((1, 2)) (x)
    x = BatchNormalization(trainable=False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        #x = Bidirectional(GRU(2*unit, return_sequences = True, 
         #                     dropout=dropout, recurrent_dropout=dropout, 
          #                    kernel_regularizer = None, 
           #                   activity_regularizer = None,
            #                  bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:, 0]) (x)                  #[b_s, vec_dim] take the central element of the sequence
    xSecond = Lambda(lambda q: q[:, 49]) (x)
    xLast = Lambda(lambda q: q[:, 99]) (x)
    queryFirst = Dense(unit*2, activation = activation) (xFirst)   # Project the element to a dense layer, this allows the network to learn 
    querySecond = Dense(unit*2, activation = activation) (xSecond)
    queryLast = Dense(unit*2, activation = activation) (xLast)
    
    #dot product attention
    attScoresFirst = Dot(axes=[1,2])([queryFirst, x]) 
    attScoresFirst = Softmax(name='attSoftmaxFirst')(attScoresFirst) #[b_s, seq_len]
    attScoresSecond = Dot(axes=[1,2])([querySecond, x]) 
    attScoresSecond = Softmax(name='attSoftmaxSecond')(attScoresSecond) #[b_s, seq_len]
    attScoresLast = Dot(axes=[1,2])([queryLast, x]) 
    attScoresLast = Softmax(name='attSoftmaxLast')(attScoresLast) #[b_s, seq_len]
    
    #rescale sequence
    attVectorFirst = Dot(axes=[1,1])([attScoresFirst, x]) #[b_s, vec_dim]
    attVectorSecond = Dot(axes=[1,1])([attScoresSecond, x]) #[b_s, vec_dim]
    attVectorLast = Dot(axes=[1,1])([attScoresLast, x]) #[b_s, vec_dim]
    x = Concatenate() ([attVectorFirst, attVectorSecond, attVectorLast])
    
    x = Dropout(dropout) (x)          

    # Two dense layer 
    x = Dense(128, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified8(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    # Two 2D convolutional layer to extract features  
    x = Conv2D(12, (7, 7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(6, (5, 5) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(3, (3, 3) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = BatchNormalization(trainable=False) (x)

    one = Lambda(lambda q: q[:, :, 0], name='one') (x) 
    two = Lambda(lambda q: q[:, :, 1], name='two') (x)
    three = Lambda(lambda q: q[:, :, 2], name='three') (x)

    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        one = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (one) # [b_s, seq_len, vec_dim]
        two = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (two) # [b_s, seq_len, vec_dim]
        three = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (three) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        #x = Bidirectional(GRU(2*unit, return_sequences = True, 
         #                     dropout=dropout, recurrent_dropout=dropout, 
          #                    kernel_regularizer = None, 
           #                   activity_regularizer = None,
            #                  bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xOne = Lambda(lambda q: q[:, 49]) (one)                  #[b_s, vec_dim] take the central element of the sequence
    xTwo = Lambda(lambda q: q[:, 49]) (two)
    xThree = Lambda(lambda q: q[:, 49]) (three)
    queryOne = Dense(unit*2, activation = activation) (xOne)   # Project the element to a dense layer, this allows the network to learn 
    queryTwo = Dense(unit*2, activation = activation) (xTwo)
    queryThree = Dense(unit*2, activation = activation) (xThree)
    
    #dot product attention
    attScoresOne = Dot(axes=[1,2])([queryOne, one]) 
    attScoresOne = Softmax(name='attSoftmaxFirst')(attScoresOne) #[b_s, seq_len]
    attScoresTwo = Dot(axes=[1,2])([queryTwo, two]) 
    attScoresTwo = Softmax(name='attSoftmaxSecond')(attScoresTwo) #[b_s, seq_len]
    attScoresThree = Dot(axes=[1,2])([queryThree, three]) 
    attScoresThree = Softmax(name='attSoftmaxLast')(attScoresThree) #[b_s, seq_len]
    
    #rescale sequence
    attScoresOne = Dot(axes=[1,1])([attScoresOne, one]) #[b_s, vec_dim]
    attScoresTwo = Dot(axes=[1,1])([attScoresTwo, two]) #[b_s, vec_dim]
    attScoresThree = Dot(axes=[1,1])([attScoresThree, three]) #[b_s, vec_dim]
    x = Concatenate() ([attScoresOne, attScoresTwo, attScoresThree])
    
    x = Dropout(dropout) (x)          

    # Two dense layer 
    x = Dense(128, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified9(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients
    """
    # Two 2D convolutional layer to extract features  
    x = Conv2D(3, (7,7) , activation='tanh', padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (7,7) , activation='tanh', padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = MaxPooling2D((1, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    """
    x = Dropout(0.1) (inputs)
    # Two 2D convolutional layer to extract features  
    x = Conv2D(10, (7, 7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    #x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    x = Dropout(0.30) (x)
    x = Conv2D(1, (5, 5) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    #x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    x = Dropout(0.25) (x)
    
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.05), 
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.05),
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x) # [b_s, seq_len, vec_dim]
        #states = [forward_h, backward_h]
        #x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True,
         #                     dropout=dropout, recurrent_dropout=dropout, 
          #                    kernel_regularizer = None, 
           #                   activity_regularizer = None,
            #                  bias_regularizer = None)) (x, initial_state = states) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Concatenate() ([forward_h, backward_h])
    query = Dense(unit*2, activation=activation) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    #attScores = Dropout(dropout) (attScores)
    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]
    
    #attVector = Dropout(dropout) (attVector)
    x = Dropout(dropout) (attVector)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified10(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients
    """
    # Two 2D convolutional layer to extract features  
    x = Conv2D(3, (7,7) , activation='tanh', padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (7,7) , activation='tanh', padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = MaxPooling2D((1, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    """
    
    # Two 2D convolutional layer to extract features  
    x = Conv2D(1, (7,7) , activation='tanh', padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    #x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    x = Dropout(dropout) (x)
    
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1), 
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1),
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1))) (x) # [b_s, seq_len, vec_dim]
        #states = [forward_h, backward_h]
        #x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True,
         #                     dropout=dropout, recurrent_dropout=dropout, 
          #                    kernel_regularizer = None, 
           #                   activity_regularizer = None,
            #                  bias_regularizer = None)) (x, initial_state = states) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(2*unit)(attention)
    attention = Permute([2, 1])(attention)

    sent_representation = Multiply() ([x, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(unit,))(sent_representation)
    
    #attVector = Dropout(dropout) (attVector)
    x = Dropout(dropout) (sent_representation)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def AttentionModelSimplified11(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients
    """
    # Two 2D convolutional layer to extract features  
    x = Conv2D(3, (7,7) , activation='tanh', padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable=False) (x)
    x = Conv2D(1, (7,7) , activation='tanh', padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (x)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = MaxPooling2D((1, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    """
    
    # Two 2D convolutional layer to extract features  
    x = Conv2D(1, (7,7) , activation='tanh', padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    #x = MaxPooling2D((2, 2)) (x)
    x = BatchNormalization(trainable=False) (x)
    
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    x = Dropout(dropout) (x)
    
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
        states = [forward_h, backward_h]
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True,
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x, initial_state = states) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(x)
    dense_outputs = [Dense(1, activation = 'tanh')(x) for x in unstacked]
    merged = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
    attention = Flatten()(merged)
    attention = Activation('softmax')(attention)
    attVector = Dot(axes=[1,1])([attention, x])
    
    #attVector = Dropout(dropout) (attVector)
    x = Dropout(dropout) (attVector)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(48, activation = activation)(x)
    x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Recursive (LSTM/GRU) Autoencoder 
def AE(nCategories, nTime, nMel, use_GRU = False):

    # Encoder part: Input layer
    encoderInputs = Input((nTime, nMel)) # it's the dimension after the extraction of the mel coeficient

    if use_GRU:
        # The bidirectional GRU layer. This layer return the state of the bidirectional GRU. 
        # The output is a 3 tensor:
        #   - gru: last output of the sequence [b_s, (nTime/2)]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        encoder = Bidirectional(GRU(int(nTime/2), return_sequences = False, return_state = True))
        gru, forward_h, backward_h = encoder(encoderInputs)

        # We discard `gru` and only keep the states.
        code_fh = Dense(16, activation='relu') (forward_h)
        code_bh = Dense(16, activation='relu') (backward_h)

        decoder_state_fh = Dense(int(nTime/2), activation='relu') (code_fh)
        decoder_state_bh = Dense(int(nTime/2), activation='relu') (code_bh)

        decoder_states = [decoder_state_fh, decoder_state_bh]
        
    else:
        # The bidirectional LSTM layer. This layer return the state of the bidirectional LSTM. 
        # The output is a 5 tensor:
        #   - lstm: last output of the sequence [b_s, (nTime/2)]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - forward_c: the last forward state c [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        #   - backward_c: the last backward state c [b_s, (nTime/2)]
        encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True))
        lstm, forward_h, forward_c, backward_h, backward_c =  encoder(encoderInputs) 

        # We discard `lstm` and only keep the states.
        code_fh = Dense(16, activation='relu') (forward_h)
        code_fc = Dense(16, activation='relu') (forward_c)
        code_bh = Dense(16, activation='relu') (backward_h)
        code_bc = Dense(16, activation='relu') (backward_c)

        decoder_state_fh = Dense(int(nTime/2), activation='relu') (code_fh)
        decoder_state_fc = Dense(int(nTime/2), activation='relu') (code_fc)
        decoder_state_bh = Dense(int(nTime/2), activation='relu') (code_bh)
        decoder_state_bc = Dense(int(nTime/2), activation='relu') (code_bc)

        decoder_states = [decoder_state_fh, decoder_state_fc, decoder_state_bh, decoder_state_bc]

    # Decoder Part
    decoderInputs = Input((nTime, nMel))

    # Set up the decoder, using `decoder_states` as initial state.
    # We set up our decoder to return full output sequences, the state is not important now
    if use_GRU:
        decoder = Bidirectional(GRU(int(nTime/2), return_sequences = True, return_state = True))
        decoderOutputs, _, _ = decoder(decoderInputs, initial_state = decoder_states)
    else:
        decoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = True))
        decoderOutputs, _, _, _, _ = decoder(decoderInputs, initial_state = decoder_states)
    
    # As last layer a dense layer to reconstruct the original signal 
    decoderDense = Dense(nMel, activation='relu')
    decoderOutputs = decoderDense(decoderOutputs)

    # Define the model 
    model = tf.keras.Model([encoderInputs, decoderInputs], decoderOutputs)
    
    return model

# LSTM Autoencoder encoder only
def AE_Encoder(nCategories, nTime, nMel, use_GRU = False):
    # Encoder only
    encoderInputs = Input((nTime, nMel)) # it's the dimension after the extraction of the mel coeficient

    if use_GRU:
        # The bidirectional GRU layer. This layer return the state of the bidirectional GRU. 
        # The output is a 3 tensor:
        #   - gru: last output of the sequence [b_s, vec_dim]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        encoder = Bidirectional(GRU(int(nTime/2), return_sequences = False, return_state = True))
        gru, forward_h, backward_h = encoder(encoderInputs) 

        # We discard `gru` and only keep the states.
        code_fh = Dense(16, activation='relu') (forward_h)
        code_bh = Dense(16, activation='relu') (backward_h)

        code = [code_fh, code_bh]
    else:
        # The bidirectional LSTM layer. This layer return the state of the bidirectional LSTM. 
        # The output is a 5 tensor:
        #   - lstm: last output of the sequence [b_s, vec_dim]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - forward_c: the last forward state c [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        #   - backward_c: the last backward state c [b_s, (nTime/2)]
        encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True))
        lstm, forward_h, forward_c, backward_h, backward_c =  encoder(encoderInputs) 

        # We discard `lstm` and only keep the states.
        code_fh = Dense(16, activation='relu') (forward_h)
        code_fc = Dense(16, activation='relu') (forward_c)
        code_bh = Dense(16, activation='relu') (backward_h)
        code_bc = Dense(16, activation='relu') (backward_c)

        code = [code_fh, code_fc, code_bh, code_bc]

    code = Concatenate() (code) 

    # Define the model that will turn
    model = tf.keras.Model(encoderInputs, code)

    return model

# Model CNN/RNN Encoder-Decoder
def Seq2SeqModel(nCategories, nTime, nMel, units = 64, use_GRU = False, dropout = 0.0):
    #Encoder 
    encoderInputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coeficient

    #encoder = Permute((2,1,3)) (encoderInputs) # Two swap the time with the mel coefficient(only with kapre)
    #add noise
    #encoder = tf.keras.layers.GaussianNoise(1) (encoderInputs)
    
    # Two convolutional layer to extract feature
    encoder = Conv2D(5, (5,1) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (encoderInputs)
    encoder = BatchNormalization() (encoder)
    encoder = Conv2D(1, (5,1) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (encoder)
    encoder = BatchNormalization() (encoder)

    encoder = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (encoder) #keras.backend.squeeze(encoder, axis)
    
    if use_GRU:
        #initial_state = K.zeros_like(encoder)  # (samples, timesteps, input_dim)
        #initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        #initial_state = K.expand_dims(initial_state)  # (samples, 1)
        #initial_state = K.tile(initial_state, [1, 64])
        #initial_state = np.zeros((64, ), dtype='float32')
        #encoder_forward_h = tf.constant(0, shape=[None, 64])
        #encoder_backward_h = tf.constant(0, shape=[None, 64])
        #x = tf.placeholder(tf.float32, shape=[None, 64])
        #cell = tf.zeros_like(x)
        #cell = tf.nn.rnn_cell.GRUCell(64)
        #encoder_state = [cell, cell]
        # First GRU layer
        encoder_1 = Bidirectional(GRU(units, return_sequences = False, return_state = True,
                                    dropout=dropout, recurrent_dropout=dropout,
                                    kernel_regularizer = None, 
                                    activity_regularizer = None,
                                    bias_regularizer = None))
                                    #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                                    #activity_regularizer = tf.keras.regularizers.l2(0.01),
                                    #bias_regularizer = tf.keras.regularizers.l2(0.01)))  # [b_s, seq_len, vec_dim]
        # The bidirectional GRU layer. This layer return the state of the bidirectional GRU. 
        # The output is a 3 tensor:
        #   - gru: last output of the sequence [b_s, vec_dim]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        #encoder_1 = Bidirectional(GRU(units, return_sequences = False, return_state = True, 
        #                              dropout=dropout, recurrent_dropout=dropout))
        gru, forward_h, backward_h = encoder_1(encoder) 

        # We discard `gru` and only keep the states.
        #forward_h = Dropout(dropout) (forward_h)
        code_fh = Dense(32, activation='tanh') (forward_h)
        #backward_h = Dropout(dropout) (backward_h)
        code_bh = Dense(32, activation='tanh') (backward_h)
        
        code = [code_fh, code_bh]

        decoder_state_fh = Dense(units, activation='tanh') (code_fh)
        decoder_state_bh = Dense(units, activation='tanh') (code_bh)

        decoder_states = [decoder_state_fh, decoder_state_bh]
    else:
        # First LSTM layer
        encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = False)) (encoder) # [b_s, seq_len, vec_dim]
        # The bidirectional LSTM layer. This layer return the state of the bidirectional LSTM. 
        # The output is a 5 tensor:
        #   - lstm: last output of the sequence [b_s, vec_dim]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - forward_c: the last forward state c [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        #   - backward_c: the last backward state c [b_s, (nTime/2)]
        encoder_1 = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True))
        lstm, forward_h, forward_c, backward_h, backward_c =  encoder_1(encoder) 

        # We discard `lstm` and only keep the states.
        code_fh = Dense(16, activation='relu') (forward_h)
        code_fc = Dense(16, activation='relu') (forward_c)
        code_bh = Dense(16, activation='relu') (backward_h)
        code_bc = Dense(16, activation='relu') (backward_c)
        
        code = [code_fh, code_fc, code_bh, code_bc]

        decoder_state_fh = Dense(int(nTime/2), activation='relu') (code_fh)
        decoder_state_fc = Dense(int(nTime/2), activation='relu') (code_fc)
        decoder_state_bh = Dense(int(nTime/2), activation='relu') (code_bh)
        decoder_state_bc = Dense(int(nTime/2), activation='relu') (code_bc)

        decoder_states = [decoder_state_fh, decoder_state_fc, decoder_state_bh, decoder_state_bc]

    #Encoder output 
    code = Concatenate(name = 'encoding') (code) 
    
    #Classifier part 
    classifier = Dense(64, activation='relu', name='classifier_input') (code)
    classifier = Dense(48, activation='relu') (classifier)
    classifier_output = Dense(nCategories, activation='softmax', name='classification') (classifier)
    
    # Decoder Part
    # Set the decoder input with the feature extracted by the cnn in the encoder with a column of zero 
    #decoderInputs = Input((nTime, nMel), name='input_2')
    #decoderInputs = Lambda(lambda q: tf.slice(q, [0, nTime-1, 0], [-1, 1, nMel]),
    #                       name='remove_last_column') (encoder)
    decoderInputs = Lambda(lambda q: q[:, 0:nTime-1, :], name='remove_last_column') (encoder)
    decoderInputs = Lambda(lambda q: tf.concat([tf.zeros([tf.shape(q)[0], 1, nMel]), q], 1), name='add_zeros_column') (decoderInputs)

    # Set up the decoder, using `decoder_states` as initial state.
    # We set up our decoder to return full output sequences, the state is not important now
    if use_GRU:
        decoder = Bidirectional(GRU(units, return_sequences = True, return_state = True, 
                                    dropout=dropout, recurrent_dropout=dropout, 
                                    kernel_regularizer = None, 
                                    activity_regularizer = None,
                                    bias_regularizer = None))
                                    #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                                    #activity_regularizer = tf.keras.regularizers.l2(0.01),
                                    #bias_regularizer = tf.keras.regularizers.l2(0.01)))
        decoderOutputs, _, _ = decoder(decoderInputs, initial_state = decoder_states)
    else:
        decoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = True))
        decoderOutputs, _, _, _, _ = decoder(decoderInputs, initial_state = decoder_states)
    
    decoderOutputs = Dropout(dropout) (decoderOutputs)
    decoderDense = Dense(nMel, activation='tanh')
    decoderOutputs = decoderDense(decoderOutputs)

    # Add a dimension for the conv layers
    decoderOutputs = Lambda(lambda q: tf.expand_dims(q, -1), name='add_dim') (decoderOutputs)

    # Now two 2D convolution transpose to recontruct the original signal
    decoderOutputs = BatchNormalization() (decoderOutputs)
    decoderOutputs = Conv2D(10, (5,1) , activation='tanh', padding='same') (decoderOutputs)
    decoderOutputs = BatchNormalization() (decoderOutputs)
    decoderOutputs = Conv2D(1, (5,1), padding='same', name='output') (decoderOutputs)
    #decoderOutputs = BatchNormalization() (decoderOutputs) 

    autoencoder = tf.keras.Model(encoderInputs, decoderOutputs)
    classifier = tf.keras.Model(encoderInputs, classifier_output)

    return autoencoder, classifier 

# Model CNN/RNN Encoder-Decoder
def Seq2SeqModel_Encoder(nCategories, nTime, nMel, use_GRU = False):
    #Encoder 
    encoderInputs = Input((nTime, nMel, 1)) # it's the dimension after the extraction of the mel coeficient

    # Two convolutional layer to extract feature
    encoder = Conv2D(10, (5,1) , activation='relu', padding='same') (encoderInputs)
    encoder = BatchNormalization() (encoder)
    encoder = Conv2D(1, (5,1) , activation='relu', padding='same') (encoder)
    encoder = BatchNormalization() (encoder)

    #encoder = Reshape((125, 80)) (encoder)
    encoder = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (encoder) #keras.backend.squeeze(encoder, axis)

    if use_GRU:
        # First GRU layer
        encoder = Bidirectional(GRU(int(nTime/2), return_sequences = True, return_state = False)) (encoder) # [b_s, seq_len, vec_dim]
        # The bidirectional GRU layer. This layer return the state of the bidirectional GRU. 
        # The output is a 3 tensor:
        #   - gru: last output of the sequence [b_s, vec_dim]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        encoder_1 = Bidirectional(GRU(int(nTime/2), return_sequences = False, return_state = True))
        gru, forward_h, backward_h = encoder_1(encoder) 

        # We discard `gru` and only keep the states.
        encoder_states = [forward_h, backward_h]

        code_fh = Dense(16, activation='relu') (forward_h)
        code_bh = Dense(16, activation='relu') (backward_h)

        code = [code_fh, code_bh]
    else:
        # First LSTM layer
        encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = False)) (encoder) # [b_s, seq_len, vec_dim]
        # The bidirectional LSTM layer. This layer return the state of the bidirectional LSTM. 
        # The output is a 5 tensor:
        #   - lstm: last output of the sequence [b_s, vec_dim]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - forward_c: the last forward state c [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        #   - backward_c: the last backward state c [b_s, (nTime/2)]
        encoder_1 = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True))
        lstm, forward_h, forward_c, backward_h, backward_c =  encoder_1(encoder) 

        # We discard `lstm` and only keep the states.
        encoder_states = [forward_h, forward_c, backward_h, backward_c]

        code_fh = Dense(16, activation='relu') (forward_h)
        code_fc = Dense(16, activation='relu') (forward_c)
        code_bh = Dense(16, activation='relu') (backward_h)
        code_bc = Dense(16, activation='relu') (backward_c)
        
        code = [code_fh, code_fc, code_bh, code_bc]

    code = Concatenate() (code) 

    # Define the model that will turn
    model = tf.keras.Model(encoderInputs, code)

    return model

def AttentionAEModel(nCategories, nTime, nMel, units = 40, use_GRU = False, dropout = 0.0, activation = 'relu'):
    #Encoder 
    encoderInputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coeficient

    #encoder = Permute((2,1,3)) (encoderInputs) # Two swap the time with the mel coefficient 

    # Two convolutional layer to extract feature
    encoder = Conv2D(10, (5,1) , activation='relu', padding='same', 
                     kernel_regularizer = tf.keras.regularizers.l2(0.001), 
                     bias_regularizer = tf.keras.regularizers.l2(0.001)) (encoderInputs)
    encoder = BatchNormalization() (encoder)
    encoder = Conv2D(1, (5,1) , activation='relu', padding='same', 
                     kernel_regularizer = tf.keras.regularizers.l2(0.001), 
                     bias_regularizer = tf.keras.regularizers.l2(0.001)) (encoder)
    encoder = BatchNormalization() (encoder)

    #encoder = Reshape((125, 80)) (encoder)
    encoder = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (encoder) #keras.backend.squeeze(encoder, axis)
    
    if use_GRU:
        # First GRU layer
        encoder = Bidirectional(GRU(units, return_sequences = True, return_state = False,
                                    dropout=dropout, recurrent_dropout=dropout, 
                                    kernel_regularizer = tf.keras.regularizers.l2(0.001), 
                                    activity_regularizer = tf.keras.regularizers.l2(0.001),
                                    bias_regularizer = tf.keras.regularizers.l2(0.001))) (encoder) # [b_s, seq_len, vec_dim]
        # The bidirectional GRU layer. This layer return the state of the bidirectional GRU. 
        # The output is a 3 tensor:
        #   - gru: last output of the sequence [b_s, vec_dim]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        encoder_1 = Bidirectional(GRU(units, return_sequences = True, return_state = False, 
                                      dropout=dropout, recurrent_dropout=dropout))
        gru = encoder_1(encoder) 

        # We apply here the attention
        # Attention layer computed by hand
        xFirst = Lambda(lambda q: q[:, int(nTime/2)]) (gru)   #[b_s, vec_dim] take the central element of the sequence
        query = Dense(units * 2, activation = activation) (xFirst) # Project the element to a dense layer, this allows the network to learn 

        #dot product attention
        attScores = Dot(axes=[1,2])([query, gru]) 
        attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

        #rescale sequence
        attVector = Dot(axes=[1,1])([attScores, gru]) #[b_s, vec_dim]          

        attVector = Dropout(dropout) (attVector)
        
        code = Dense(32, activation='relu') (attVector)

        decoder_state_fh = Dense(units, activation='relu') (code)
        decoder_state_bh = Dense(units, activation='relu') (code)

        decoder_states = [decoder_state_fh, decoder_state_bh]
    else:
        # First LSTM layer
        encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = False)) (encoder) # [b_s, seq_len, vec_dim]
        # The bidirectional LSTM layer. This layer return the state of the bidirectional LSTM. 
        # The output is a 5 tensor:
        #   - lstm: last output of the sequence [b_s, vec_dim]
        #   - forward_h: the last forward state h [b_s, (nTime/2)]
        #   - forward_c: the last forward state c [b_s, (nTime/2)]
        #   - backward_h: the last backward state h [b_s, (nTime/2)]
        #   - backward_c: the last backward state c [b_s, (nTime/2)]
        encoder_1 = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True))
        lstm, forward_h, forward_c, backward_h, backward_c =  encoder_1(encoder) 

        # We discard `lstm` and only keep the states.
        code_fh = Dense(16, activation='relu') (forward_h)
        code_fc = Dense(16, activation='relu') (forward_c)
        code_bh = Dense(16, activation='relu') (backward_h)
        code_bc = Dense(16, activation='relu') (backward_c)

        decoder_state_fh = Dense(int(nTime/2), activation='relu') (code_fh)
        decoder_state_fc = Dense(int(nTime/2), activation='relu') (code_fc)
        decoder_state_bh = Dense(int(nTime/2), activation='relu') (code_bh)
        decoder_state_bc = Dense(int(nTime/2), activation='relu') (code_bc)

        decoder_states = [decoder_state_fh, decoder_state_fc, decoder_state_bh, decoder_state_bc]

    # Decoder Part
    decoderInputs = Input((nTime, nMel), name='input_2')

    # Set up the decoder, using `decoder_states` as initial state.
    # We set up our decoder to return full output sequences, the state is not important now
    if use_GRU:
        decoder = Bidirectional(GRU(units, return_sequences = True, return_state = True, 
                                    dropout=dropout, recurrent_dropout=dropout, 
                                    kernel_regularizer = tf.keras.regularizers.l2(0.001), 
                                    activity_regularizer = tf.keras.regularizers.l2(0.001),
                                    bias_regularizer = tf.keras.regularizers.l2(0.001)))
        decoderOutputs, _, _ = decoder(decoderInputs, initial_state = decoder_states)
    else:
        decoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = True))
        decoderOutputs, _, _, _, _ = decoder(decoderInputs, initial_state = decoder_states)
    
    decoderOutputs = Dropout(dropout) (decoderOutputs)
    decoderDense = Dense(nMel, activation='relu')
    decoderOutputs = decoderDense(decoderOutputs)

    # Add a dimension for the conv layers
    decoderOutputs = Lambda(lambda q: tf.expand_dims(q, -1), name='add_dim') (decoderOutputs)

    # Now two 2D convolution transpose to recontruct the original signal
    decoderOutputs = Conv2DTranspose(10, (5,1) , activation='relu', padding='same') (decoderOutputs)
    decoderOutputs = BatchNormalization() (decoderOutputs)
    decoderOutputs = Conv2DTranspose(1, (5,1) , activation='relu', padding='same') (decoderOutputs)
    decoderOutputs = BatchNormalization() (decoderOutputs)

    model = tf.keras.Model([encoderInputs, decoderInputs], decoderOutputs)

    return model

# Model with the attention layer 
def NewAttentionModel(nCategories, nTime, nMel, unit, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    # Two 2D convolutional layer to extract features  
    x = Conv2D(10, (5,1) , activation=activation, padding='same',
               kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable = False) (x)
    x = Conv2D(1, (3,1) , activation=activation, padding='same', 
               kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               bias_regularizer = tf.keras.regularizers.l2(0.001)) (x)
    x = BatchNormalization(trainable = False) (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    # Two bidirectional GRU layer were the output is the complete sequence 
    x = Bidirectional(GRU(unit, return_sequences = True, 
                          dropout=dropout, recurrent_dropout=dropout, 
                          kernel_regularizer = tf.keras.regularizers.l2(0.001), 
                          activity_regularizer = tf.keras.regularizers.l2(0.001),
                          bias_regularizer = tf.keras.regularizers.l2(0.001))) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(GRU(unit, return_sequences = True, 
                          dropout=dropout, recurrent_dropout=dropout, 
                          kernel_regularizer = tf.keras.regularizers.l2(0.001), 
                          activity_regularizer = tf.keras.regularizers.l2(0.001),
                          bias_regularizer = tf.keras.regularizers.l2(0.001))) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:, 0]) (x)                  #[b_s, vec_dim] take the central element of the sequence
    xSecond = Lambda(lambda q: q[:, int(nTime/2)]) (x)
    queryFirst = Dense(unit*2, activation = activation) (xFirst)   # Project the element to a dense layer, this allows the network to learn 
    querySecond = Dense(unit*2, activation = activation) (xSecond)
    
    #dot product attention
    attScoresFirst = Dot(axes=[1,2])([queryFirst, x]) 
    attScoresFirst = Softmax(name='attSoftmaxFirst')(attScoresFirst) #[b_s, seq_len]
    attScoresSecond = Dot(axes=[1,2])([querySecond, x]) 
    attScoresSecond = Softmax(name='attSoftmaxSecond')(attScoresSecond) #[b_s, seq_len]
    
    #rescale sequence
    attVectorFirst = Dot(axes=[1,1])([attScoresFirst, x]) #[b_s, vec_dim]
    attVectorSecond = Dot(axes=[1,1])([attScoresSecond, x]) #[b_s, vec_dim]
    attVectorDiff = tf.keras.layers.Subtract() ([attVectorFirst, attVectorSecond])
    x = Concatenate() ([attVectorFirst, attVectorSecond, attVectorDiff])
    
    x = Dropout(dropout) (x)
    
    # Two dense layer 
    x = Dense(128, activation = activation)(x)
    x = Dropout(dropout) (x)
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

"""
print("\nAttention Model\n")
AttModel = AttentionModelSimplified11(35, 100, 80, 32, use_GRU = True)
AttModel.summary()

print("\nRNN Autoencoder Model\n")
AE = AE(10, 28, 28, use_GRU = True)
AE.summary()
AE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

print("\nRNN Autoencoder_Encoder Model\n")
AE_e = AE_Encoder(10, 28, 28, use_GRU = True)
AE_e.summary()

print("\nCNN+RNN Autoencoder Model\n")
Autoencoder, Encoder = Seq2SeqModel(10, 125, 80, use_GRU = True, dropout = 0.1)
Autoencoder.summary()
Encoder.summary()

print("\nCNN+RNN Autoencoder_Encoder Model\n")
Autoencoder = Seq2SeqModel_Encoder(10, 125, 80, use_GRU = True)
Autoencoder.summary()

print("\nAttention AE Model\n")
AttModel = AttentionAEModel(21, 99, 39, use_GRU = True)
AttModel.summary()

print("\n New Attention Model\n")
AttModel = NewAttentionModel(21, 99, 39, 40)
AttModel.summary()
"""