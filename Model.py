# TensorFlow and tf.keras
import tensorflow as tf
#from tensorflow import keras
from keras.models import Model, load_model

from keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax
from keras.layers import Add, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dense, Bidirectional, LSTM, GRU, CuDNNLSTM
#from keras.layers import Attention, CuDNNLSTM
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers

#kapre for Mel Coefficient 
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D
from keras import regularizers

import numpy as np

#print(tf.__version__)
#print(tf.keras.__version__)

# Model with the attention layer 
def SimpleModel(nCategories, nTime, nMel, use_GRU = False, dropout = 0.0, activation = 'relu'):
    
    #inputs = Input((nTime, nMel, 1)) # it's the dimension after the extraction of the mel coefficients

    inputs = Input((16000,))

    """ We need to drop out this part and compute by hand the mel coefficient
        because this part user keras without tensorflow and there is a bug that
        create problem """
    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 16000),
                             padding='same', sr=16000, n_mels=80,
                             fmin=40.0, fmax=16000/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)
    """
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
    """
    x = Permute((2,1,3)) (x)

    # Two 2D convolutional layer to extract features  
    x = Conv2D(64, (5,5) , activation=activation) (x)
    x = MaxPooling2D((3, 3)) (x)
    x = BatchNormalization() (x)
    x = Conv2D(32, (3,3) , activation=activation) (x)
    x = MaxPooling2D((3, 3)) (x)
    x = BatchNormalization() (x)

    x = Flatten() (x)
    x = Dropout(dropout) (x)
    # Two dense layer 
    x = Dense(128, activation = activation)(x)
    
    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model


# Model with the attention layer 
def AttentionModel(nCategories, nTime, nMel, use_GRU = False, dropout = 0.0, activation = 'relu'):
    
    #inputs = Input((nTime, nMel, 1)) # it's the dimension after the extraction of the mel coefficients

    inputs = Input((16000,))

    """ We need to drop out this part and compute by hand the mel coefficient
        because this part user keras without tensorflow and there is a bug that
        create problem """
    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 16000),
                             padding='same', sr=16000, n_mels=80,
                             fmin=40.0, fmax=16000/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)
    """
    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs
    """
    x = Permute((2,1,3)) (x)

    # Two 2D convolutional layer to extract features  
    x = Conv2D(10, (5,1) , activation=activation, padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(1, (3,1) , activation=activation, padding='same') (x)
    x = BatchNormalization() (x)

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    """ If we have GPU
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    """
    if use_GRU:
        # Two bidirectional GRU layer were the output is the complete sequence 
        x = Bidirectional(GRU(nMel, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(GRU(nMel, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    else:
        # Two bidirectional LSTM layer were the output is the complete sequence 
        x = Bidirectional(LSTM(nMel, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(nMel, return_sequences = True, dropout=dropout, recurrent_dropout=dropout)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:, int(nTime/2)]) (x)   #[b_s, vec_dim] take the central element of the sequence
    query = Dense(nMel*2, activation = activation) (xFirst) # Project the element to a dense layer, this allows the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    # Now use the Attention layer (not find when compile, i don't know why)
    # attVector = Attention()([query, x])

    # Two dense layer 
    x = Dense(64, activation = activation)(attVector)
    x = Dropout(dropout) (x)
    x = Dense(32, activation = activation)(x)
    
    # Normalizzation to prevent nan 
    #x = BatchNormalization() (x)
    
    # subrract the max to avoid nan
    #x = Lambda(lambda q: q - K.max(q), name='subtract_max') (x)

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
def Seq2SeqModel(nCategories, nTime, nMel, use_GRU = False):
    #Encoder 
    encoderInputs = Input((nMel, nTime, 1)) # it's the dimension after the extraction of the mel coeficient

    encoder = Permute((2,1,3)) (encoderInputs) # Two swap the time with the mel coefficient 

    # Two convolutional layer to extract feature
    encoder = Conv2D(10, (5,1) , activation='relu', padding='same') (encoder)
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
        code_fh = Dense(16, activation='relu') (forward_h)
        code_bh = Dense(16, activation='relu') (backward_h)

        decoder_state_fh = Dense(int(nTime/2), activation='relu') (code_fh)
        decoder_state_bh = Dense(int(nTime/2), activation='relu') (code_bh)

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
    decoderInputs = Input((nTime, nMel))

    # Set up the decoder, using `decoder_states` as initial state.
    # We set up our decoder to return full output sequences, the state is not important now
    if use_GRU:
        decoder = Bidirectional(GRU(int(nTime/2), return_sequences = True, return_state = True))
        decoderOutputs, _, _ = decoder(decoderInputs, initial_state = decoder_states)
    else:
        decoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = True))
        decoderOutputs, _, _, _, _ = decoder(decoderInputs, initial_state = decoder_states)
    
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

# Model CNN/RNN Encoder-Decoder
def Seq2SeqModel_Encoder(nCategories, nTime, nMel, use_GRU = False):
    #Encoder 
    encoderInputs = Input((nMel, nTime, 1)) # it's the dimension after the extraction of the mel coeficient

    encoder = Permute((2,1,3)) (encoderInputs) # Two swap the time with the mel coefficient 

    # Two convolutional layer to extract feature
    encoder = Conv2D(10, (5,1) , activation='relu', padding='same') (encoder)
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

"""
print("\nAttention Model\n")
AttModel = AttentionModel(10, 28, 28, use_GRU = True)
AttModel.summary()

print("\nRNN Autoencoder Model\n")
AE = AE(10, 28, 28, use_GRU = True)
AE.summary()
AE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

print("\nRNN Autoencoder_Encoder Model\n")
AE_e = AE_Encoder(10, 28, 28, use_GRU = True)
AE_e.summary()

print("\nCNN+RNN Autoencoder Model\n")
Autoencoder = Seq2SeqModel(10, 125, 80, use_GRU = True)
Autoencoder.summary()

print("\nCNN+RNN Autoencoder_Encoder Model\n")
Autoencoder = Seq2SeqModel_Encoder(10, 125, 80, use_GRU = True)
Autoencoder.summary()
"""