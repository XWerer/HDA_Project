# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax, RepeatVector, Multiply
from tensorflow.keras.layers import Add, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Layer
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU
from tensorflow.keras import initializers as initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import optimizers

import numpy as np

# Model with the attention layer 
def AttentionModel(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

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

def CNN(nCategories, nTime, nMel, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1)) # it's the dimension after the extraction of the mel coefficients

    x = Conv2D(32, (30,5) , activation=activation) (inputs) 
    x = MaxPooling2D((2, 3)) (x)
    x = BatchNormalization() (x)
    x = Conv2D(32, (10,2) , activation=activation) (x)
    x = MaxPooling2D((1, 2)) (x)
    x = BatchNormalization() (x)
    
    x = Flatten()(x)
    
    #x = Dropout(dropout) (x)
    # Two dense layer 
    x = Dense(128, activation = activation)(x)
    
    x = Dense(64, activation = activation)(x)
    
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model


# Model with 1 cnn, 1 rnn e a simple attention by douglas 
def RNNAtt(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    # 2D convolutional layer to extract features  
    x = Conv2D(1, (7,7) , activation=activation, padding='same',
               kernel_regularizer = None, 
               bias_regularizer = None) (inputs)
               #kernel_regularizer = tf.keras.regularizers.l2(0.001), 
               #bias_regularizer = tf.keras.regularizers.l2(0.001)) (inputs)
    x = BatchNormalization(trainable=False) (x)

    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    if use_GRU:
        # One bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]

    else:
        # One bidirectional LSTM layer were the output is the complete sequence 
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

#Original douglas model
def AttRNNSpeechModel(nCategories, samplingrate = 16000, inputLength = 16000):
    
    inputs = Input((100, 80, 1))

    x = Conv2D(10, (5,1) , activation='relu', padding='same') (inputs)
    x = BatchNormalization() (x)
    x = Conv2D(1, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)

    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    x = Bidirectional(LSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(LSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]

    xFirst = Lambda(lambda q: q[:,50]) (x) #[b_s, vec_dim]
    query = Dense(128) (xFirst)

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]

    x = Dense(64, activation = 'relu')(attVector)
    x = Dense(32)(x)

    output = Dense(nCategories, activation = 'softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def PCNNRNNStatefulAttDouglas(nCategories, nTime, nMel, unit = 32, use_GRU = False, dropout = 0.0, activation = 'relu', maxPooling = True):
    
    inputs = Input((nTime, nMel, 1)) # it's the dimension after the extraction of the mel coefficients

    # Two 2D convolutional layer to extract features  
    x = Conv2D(30, (20,6) , activation=activation) (inputs) 
    if(maxPooling):
        x = MaxPooling2D((2, 1)) (x)
    x = BatchNormalization() (x)
    x = Conv2D(12, (10,3) , activation=activation, padding = 'same') (x)
    x = Conv2D(9, (5,2) , activation=activation) (x)
    x = Conv2D(7, (4,1) , activation=activation, padding = 'same') (x)
    #x = Conv2D(3, (4,1) , activation=activation) (x)
    x = Conv2D(1, (4,1) , activation=activation) (x)
    #x = MaxPooling2D((3, 3)) (x)
    x = BatchNormalization() (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x)
    
    x = Bidirectional(GRU(unit, return_sequences = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              kernel_regularizer = None, 
                              activity_regularizer = None,
                              bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              #kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                              #activity_regularizer = tf.keras.regularizers.l2(0.1),
                              #bias_regularizer = tf.keras.regularizers.l2(0.1))) (x) # [b_s, seq_len, vec_dim]
    #x = Flatten()(x)
    xFirst = Lambda(lambda q: q[:, 15, :]) (x)   #[b_s, vec_dim] take the central element of the sequence
    query = Dense(64, name = 'query') (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]
    #attScores = Dropout(dropout) (attScores)
    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    #x = Dropout(dropout) (x)
    # Two dense layer 
    x = Dense(64, activation = activation)(attVector)
    
    #x = Dense(64)(x)
    #x = Dense(48)(x)
    
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model


# Model with the attention layer 
def CNNRNNDouglasAtt2(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients
    
    x = Conv2D(4, (7,7) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (inputs)
    x = BatchNormalization() (x)
    x = Dropout(0.1) (x)
    x = Conv2D(3, (6,6) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    #x = MaxPooling2D((1, 2)) (x)
    x = Dropout(0.0) (x) #before 0.1
    x = Conv2D(2, (5,5) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    x = Dropout(0.1) (x)
    x = Conv2D(1, (3,3) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    x = Dropout(0.1) (x) # before 0.2
    
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001), 
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x) # [b_s, seq_len, vec_dim]
        states = [forward_h, backward_h]
        x = Dropout(0.1) (x) #before 0.2
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001), 
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x) 
                          
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:,50]) (x) #[b_s, vec_dim]
    query = Dense(unit*2) (xFirst)

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]
    
    #attVector = Dropout(dropout) (attVector)
    x = Dropout(dropout) (attVector)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(0.1) (x)
    x = Dense(48, activation = activation)(x)
    x = Dropout(0.0) (x) #before 0.1
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model cnn + 2 rnn passing the state and attention by douglas
def CNNRNNAttDouglas(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients
    
    x = Conv2D(4, (7,7) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (inputs)
    x = BatchNormalization() (x)
    x = Dropout(0.1) (x)
    x = Conv2D(3, (6,6) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    #x = MaxPooling2D((1, 2)) (x)
    x = Dropout(0.0) (x) #before 0.1
    x = Conv2D(2, (5,5) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    x = Dropout(0.1) (x)
    x = Conv2D(1, (3,3) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    x = Dropout(0.1) (x) # before 0.2
    
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001), 
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x) # [b_s, seq_len, vec_dim]
        states = [forward_h, backward_h]
        x = Dropout(0.1) (x) #before 0.2
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001), 
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.001),
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x, initial_state = states) 
                          
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:,50]) (x) #[b_s, vec_dim]
    query = Dense(unit*2) (xFirst)

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]
    
    #attVector = Dropout(dropout) (attVector)
    x = Dropout(dropout) (attVector)
    # Two dense layer 
    x = Dense(64, activation = activation)(x)
    x = Dropout(0.1) (x)
    x = Dense(48, activation = activation)(x)
    x = Dropout(0.0) (x) #before 0.1
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with cnn + 2 rnn passing state + attention with state
def CNNRNNAttState(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients
    
    x = Conv2D(4, (7,7) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (inputs)
    x = BatchNormalization() (x)
    x = Dropout(0.05) (x) #prima 0.1
    x = Conv2D(3, (6,6) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    #x = MaxPooling2D((1, 2)) (x)
    x = Dropout(0.0) (x) #before 0.1
    x = Conv2D(2, (5,5) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    x = Dropout(0.05) (x) #prima 0.1
    x = Conv2D(1, (3,3) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    x = Dropout(0.05) (x) # before 0.2 - prima 0.1
    
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x) # [b_s, seq_len, vec_dim]
        states = [forward_h, backward_h]
        x = Dropout(0.05) (x) #before 0.2 - prima 0.1
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0),  #before 0.001
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x, initial_state = states) 
                          
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
    x = Dropout(0.05) (x) #prima 0.1
    x = Dense(48, activation = activation)(x)
    x = Dropout(0.0) (x) #before 0.1
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model with the attention layer 
def MultyAttention(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients
    
    x = Conv2D(4, (7,7) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (inputs)
    x = BatchNormalization() (x)
    x = Dropout(0.05) (x) #prima 0.1
    x = Conv2D(3, (6,6) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    #x = MaxPooling2D((1, 2)) (x)
    x = Dropout(0.0) (x) #before 0.1
    x = Conv2D(2, (5,5) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    x = Dropout(0.05) (x) #prima 0.1
    x = Conv2D(1, (3,3) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    x = Dropout(0.05) (x) # before 0.2 - prima 0.1
    
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x) # [b_s, seq_len, vec_dim]
        states = [forward_h, backward_h]
        x = Dropout(0.0) (x) #before 0.2 - prima 0.1
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0),  #before 0.001
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0), #before 0.001
                              kernel_initializer=tf.compat.v1.initializers.he_normal(seed=None),
                              recurrent_initializer=tf.compat.v1.initializers.he_normal(seed=None))) (x, initial_state = states) 
                          
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(unit, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Concatenate() ([forward_h, backward_h])
    xSecond = Lambda(lambda q: q[:,50]) (x) #[b_s, vec_dim]
    queryFirst = Dense(unit*2, activation='relu') (xFirst)
    querySecond = Dense(unit*2, activation='relu') (xSecond)

    #dot product attention
    attScoresFirst = Dot(axes=[1,2])([queryFirst, x]) 
    attScoresFirst = Softmax(name='attSoftmaxFirst')(attScoresFirst) #[b_s, seq_len]
    attScoresSecond = Dot(axes=[1,2])([querySecond, x]) 
    attScoresSecond = Softmax(name='attSoftmaxSecond')(attScoresSecond) #[b_s, seq_len]

    #rescale sequence
    attVectorFirst = Dot(axes=[1,1])([attScoresFirst, x]) #[b_s, vec_dim] 
    attVectorSecond = Dot(axes=[1,1])([attScoresSecond, x]) #[b_s, vec_dim] 
    
    attVector = Concatenate() ([attVectorFirst, attVectorSecond])
    x = Dropout(dropout) (attVector)
    # Two dense layer 
    x = Dense(128, activation = activation)(x)
    x = Dropout(0.0) (x) #prima 0.1
    x = Dense(64, activation = activation)(x)
    x = Dropout(0.0) (x) #before 0.1
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

def MultyAttentionDouglas(nCategories, nTime, nMel, dropout = 0.0):
    
    inputs = Input((100, 80, 1))

    x = Conv2D(10, (5,1) , activation='relu', padding='same') (inputs)
    x = BatchNormalization() (x)
    x = Conv2D(1, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)

    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    x = Bidirectional(LSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x, forward_h, backward_h, forward_b, backward_b, = Bidirectional(LSTM(64, return_sequences = True, return_state = True)) (x) # [b_s, seq_len, vec_dim]

    xFirst = Concatenate() ([forward_h, backward_h, forward_b, backward_b])
    xSecond = Lambda(lambda q: q[:,50]) (x) #[b_s, vec_dim]
    queryFirst = Dense(128) (xFirst)
    querySecond = Dense(128) (xSecond)

    #dot product attention
    attScoresFirst = Dot(axes=[1,2])([queryFirst, x]) 
    attScoresFirst = Softmax(name='attSoftmaxFirst')(attScoresFirst) #[b_s, seq_len]
    attScoresSecond = Dot(axes=[1,2])([querySecond, x]) 
    attScoresSecond = Softmax(name='attSoftmaxSecond')(attScoresSecond) #[b_s, seq_len]

    #rescale sequence
    attVectorFirst = Dot(axes=[1,1])([attScoresFirst, x]) #[b_s, vec_dim] 
    attVectorSecond = Dot(axes=[1,1])([attScoresSecond, x]) #[b_s, vec_dim] 
    
    attVector = Concatenate() ([attVectorFirst, attVectorSecond])
    x = Dropout(dropout) (attVector)
    
    # Two dense layer 
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(dropout) (x) 
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(dropout) (x) 
    
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model


# Model with the attention layer (best model 13/07/19 - train 72 - validation 82)
def MaxPooling(nCategories, nTime, nMel, unit, use_GRU = True, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients
    
    x = Conv2D(4, (7,7) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (inputs)
    x = BatchNormalization() (x)
    x = Dropout(0.2) (x)
    x = Conv2D(3, (6,6) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    x = MaxPooling2D((1, 2)) (x)
    x = Dropout(0.2) (x)
    x = Conv2D(2, (5,5) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    x = Dropout(0.2) (x)
    x = Conv2D(1, (3,3) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    x = Dropout(0.25) (x)
    
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x, forward_h, backward_h = Bidirectional(GRU(unit, return_sequences = True, return_state = True, 
                              dropout=dropout, recurrent_dropout=dropout, 
                              #kernel_regularizer = None, 
                              #activity_regularizer = None,
                              #bias_regularizer = None)) (x) # [b_s, seq_len, vec_dim]
                              kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.01), 
                              activity_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.01),
                              bias_regularizer = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.01),
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

def NewAttentionModelOurs(nCategories, nTime, nMel, unit, dropout = 0.0, activation = 'relu'):
    
    inputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coefficients

    # Two 2D convolutional layer to extract features  
    x = Conv2D(4, (7,7) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (inputs)
    x = BatchNormalization() (x)
    x = Dropout(0.05) (x) #prima 0.1
    x = Conv2D(3, (6,6) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    #x = MaxPooling2D((1, 2)) (x)
    x = Dropout(0.0) (x) #before 0.1
    x = Conv2D(2, (5,5) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)
    x = Dropout(0.05) (x) #prima 0.1
    x = Conv2D(1, (3,3) , activation='relu', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (x)
    x = BatchNormalization() (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)
    # Two bidirectional GRU layer were the output is the complete sequence 
    x = Bidirectional(GRU(unit, return_sequences = True, 
                          dropout=dropout, recurrent_dropout=dropout, 
                          kernel_regularizer = tf.keras.regularizers.l2(0.00), #before 0.001
                          activity_regularizer = tf.keras.regularizers.l2(0.00),
                          bias_regularizer = tf.keras.regularizers.l2(0.00))) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(GRU(unit, return_sequences = True, 
                          dropout=dropout, recurrent_dropout=dropout, 
                          kernel_regularizer = tf.keras.regularizers.l2(0.00), 
                          activity_regularizer = tf.keras.regularizers.l2(0.00),
                          bias_regularizer = tf.keras.regularizers.l2(0.00))) (x) # [b_s, seq_len, vec_dim]
    
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
    
    #x = Dropout(dropout) (x)
    
    # Two dense layer 
    x = Dense(128, activation = activation)(x)
    #x = Dropout(dropout) (x)
    x = Dense(64, activation = activation)(x)
    #x = Dropout(dropout) (x)
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Model CNN/RNN Encoder-Decoder
def Seq2SeqModel(nCategories, nTime, nMel, units = 64, use_GRU = False, dropout = 0.0):
    #Encoder 
    encoderInputs = Input((nTime, nMel, 1), name='input_1') # it's the dimension after the extraction of the mel coeficient
    
    # CNN
    encoder = Conv2D(4, (7,7) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (encoderInputs)
    encoder = BatchNormalization() (encoder)
    encoder = Dropout(0.2) (encoder)
    encoder = Conv2D(3, (6,6) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (encoder)
    encoder = BatchNormalization() (encoder)
    encoder = Dropout(0.2) (encoder)
    encoder = Conv2D(2, (5,5) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (encoder)
    encoder = BatchNormalization() (encoder)
    encoder = Dropout(0.2) (encoder)
    encoder = Conv2D(1, (3,3) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (encoder)
    encoder = BatchNormalization() (encoder)

    encoder = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (encoder) #keras.backend.squeeze(encoder, axis)
    
    encoder = Dropout(0.2) (encoder)
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
        code_fh = Dense(64, activation='tanh') (forward_h)
        #backward_h = Dropout(dropout) (backward_h)
        code_bh = Dense(64, activation='tanh') (backward_h)
        
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
    code = Dropout(0.1) (code)
    #Classifier part 
    classifier = Dense(64, activation='relu', name='classifier_input') (code)
    classifier = Dropout(0.25) (classifier)
    classifier = Dense(48, activation='relu') (classifier)
    classifier_output = Dense(nCategories, activation='softmax', name='classification') (classifier)
    
    # Decoder Part
    # Set the decoder input with the feature extracted by the cnn in the encoder with a column of zero 
    #decoderInputs = Input((nTime, nMel), name='input_2')
    #decoderInputs = Lambda(lambda q: tf.slice(q, [0, nTime-1, 0], [-1, 1, nMel]),
    #                       name='remove_last_column') (encoder)
    decoderInputs = Lambda(lambda q: q[:, 0:nTime-1, :], name='remove_last_column') (encoder)
    decoderInputs = Lambda(lambda q: tf.concat([tf.zeros([tf.shape(q)[0], 1, nMel]), q], 1), name='add_zeros_column') (decoderInputs)
    decoderInputs = Dropout(0.2) (decoderInputs)
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
    
    decoderOutputs = Dropout(0.2) (decoderOutputs)
    decoderOutputs = Dense(nMel, activation='tanh') (decoderOutputs)

    # Add a dimension for the conv layers
    decoderOutputs = Lambda(lambda q: tf.expand_dims(q, -1), name='add_dim') (decoderOutputs)

    # Now CNN to recontruct the original signal
    decoder = Conv2D(4, (7,7) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoderInputs)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (decoderOutputs)
    decoder = BatchNormalization() (decoder)
    decoder = Dropout(0.2) (decoder)
    decoder = Conv2D(3, (6,6) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (decoder)
    decoder = BatchNormalization() (decoder)
    decoder = Dropout(0.2) (decoder)
    decoder = Conv2D(2, (5,5) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (decoder)
    decoder = BatchNormalization() (decoder)
    decoder = Dropout(0.2) (decoder)
    decoder = Conv2D(1, (3,3) , activation='tanh', padding='same', 
                     #kernel_regularizer = tf.keras.regularizers.l2(0.01), 
                     #bias_regularizer = tf.keras.regularizers.l2(0.01)) (encoder)
                     kernel_regularizer = None, 
                     bias_regularizer = None) (decoder)

    autoencoder = tf.keras.Model(encoderInputs, decoder)
    classifier = tf.keras.Model(encoderInputs, classifier_output)

    return autoencoder, classifier