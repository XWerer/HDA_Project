# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax
from tensorflow.keras.layers import Add, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, Reshape, MaxPooling2D, Dense, Bidirectional, LSTM
#from tensorflow.keras.layers import Attention, CuDNNLSTM
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import optimizers

#kapre for Mel Coefficient 
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

import numpy as np

#print(tf.__version__)
#print(tf.keras.__version__)

# Model with the attention layer 
def AttentionModel(nCategories, nTime, nMel):
    
    inputs = Input((nMel, nTime, 1)) # it's the dimension after the extraction of the mel coeficient

    #inputs = Input((samplingrate,))

    """ We need to drop out this part and compute by hand the mel coefficient
        because this part user keras without tensorflow and there is a bug that
        create problem 
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
    """
    x = Permute((2,1,3)) (inputs)

    # Two 2D convolutional layer to extrac features  
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
    # Two bidirectional LSTM layer were the output is the complete sequence 
    x = Bidirectional(LSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(LSTM(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim]
    
    # Attention layer computed by hand
    xFirst = Lambda(lambda q: q[:,int(nTime/2)]) (x) #[b_s, vec_dim] take the central element of the sequence
    query = Dense(128) (xFirst)            # Project the element to a dense layer this allow the network to learn 

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]          

    # Now use the Attention layer (not find when compile, i don't know why)
    # attVector = Attention()([query, x])

    # Two dense layer 
    x = Dense(64, activation = 'relu')(attVector)
    x = Dense(32)(x)

    output = Dense(nCategories, activation = 'softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    
    return model

# LSTM Autoencoder 
def LSTMAutoencoder(nCategories, nTime, nMel):
    # Encoder part
    # Input layer 
    encoderInputs = Input((nTime, nMel)) # it's the dimension after the extraction of the mel coeficient

    # The bidirectional LSTM layer. This layer return the state of the bidirectional lstm. 
    # The output is a 5 tensor:
    #   - lstm: last output of the sequence [b_s, vec_dim]
    #   - forward_h: the last forward state h [b_s, 64]
    #   - forward_c: the last forward state c [b_s, 64]
    #   - backward_h: the last backward state h [b_s, 64]
    #   - backward_c: the last backward state c [b_s, 64]
    encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True, dropout = 0.3))
    lstm, forward_h, forward_c, backward_h, backward_c =  encoder(encoderInputs) 

    # We discard `encoder_outputs` and only keep the states.
    #encoder_states = [state_h, state_c]
    encoder_states = [forward_h, forward_c, backward_h, backward_c]

    code_fh = Dense(16, activation='relu') (forward_h)
    code_fc = Dense(16, activation='relu') (forward_c)
    code_bh = Dense(16, activation='relu') (backward_h)
    code_bc = Dense(16, activation='relu') (backward_c)

    decoder_state_fh = Dense(int(nTime/2), activation='relu') (code_fh)
    decoder_state_fc = Dense(int(nTime/2), activation='relu') (code_fc)
    decoder_state_bh = Dense(int(nTime/2), activation='relu') (code_bh)
    decoder_state_bc = Dense(int(nTime/2), activation='relu') (code_bc)

    decoder_states = [decoder_state_fh, decoder_state_fc, decoder_state_bh, decoder_state_bc]
    
    # Decoder Part (only one bidirectional lstm and a dense layer)
    # Set up the decoder, using `encoder_states` as initial state.
    #decoder_inputs = Input(shape = (None, 64))
    decoderInputs = Input((nTime, nMel))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = True, dropout = 0.3))
    decoderOutputs, _, _, _, _ = decoder_lstm(decoderInputs, initial_state = decoder_states)
    decoderDense = Dense(nMel, activation='relu')
    decoderOutputs = decoderDense(decoderOutputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoderInputs, decoderInputs], decoderOutputs)
    
    return model

# LSTM Autoencoder encoder only
def LSTMAutoencoder_Encoder(nCategories, nTime, nMel):
    # Encoder only
    # Input layer 
    encoderInputs = Input((nTime, nMel)) # it's the dimension after the extraction of the mel coeficient
    
    # The bidirectional LSTM layer. This layer return the state of the bidirectional lstm. 
    # The output is a 5 tensor:
    #   - lstm: last output of the sequence [b_s, vec_dim]
    #   - forward_h: the last forward state h [b_s, 64]
    #   - forward_c: the last forward state c [b_s, 64]
    #   - backward_h: the last backward state h [b_s, 64]
    #   - backward_c: the last backward state c [b_s, 64]
    encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True, dropout = 0.3))
    lstm, forward_h, forward_c, backward_h, backward_c =  encoder(encoderInputs) 
    # We discard `encoder_outputs` and only keep the states.
    #encoder_states = [state_h, state_c]
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

# Model CNN/RNN Encoder-Decoder
def Seq2SeqModel(nCategories, nTime, nMel):
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

    # First bidirectional layer that return only the full sequence that is the input of the next bidirectiona lstm layer
    encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = False)) (encoder) # [b_s, seq_len, vec_dim]
    # Second bidirectional LSTM layer. This layer return the state of the bidirectional lstm. 
    # The output is a 5 tensor:
    #   - lstm: last output of the sequence [b_s, vec_dim]
    #   - forward_h: the last forward state h [b_s, 64]
    #   - forward_c: the last forward state c [b_s, 64]
    #   - backward_h: the last backward state h [b_s, 64]
    #   - backward_c: the last backward state c [b_s, 64]
    lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True)) (encoder) 

    # We discard `encoder_outputs` and only keep the states.
    #encoder_states = [state_h, state_c]
    encoder_states = [forward_h, forward_c, backward_h, backward_c]

    code_fh = Dense(16, activation='relu') (forward_h)
    code_fc = Dense(16, activation='relu') (forward_c)
    code_bh = Dense(16, activation='relu') (backward_h)
    code_bc = Dense(16, activation='relu') (backward_c)

    decoder_state_fh = Dense(int(nTime/2), activation='relu') (code_fh)
    decoder_state_fc = Dense(int(nTime/2), activation='relu') (code_fc)
    decoder_state_bh = Dense(int(nTime/2), activation='relu') (code_bh)
    decoder_state_bc = Dense(int(nTime/2), activation='relu') (code_bc)

    decoder_states = [decoder_state_fh, decoder_state_fc, decoder_state_bh, decoder_state_bc]

    # Decoder Part (only one bidirectional lstm and a dense layer)
    # Set up the decoder, using `encoder_states` as initial state.
    #decoder_inputs = Input(shape = (None, 64))
    decoderInputs = Input((nTime, nMel))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = True))
    decoderOutputs, _, _, _, _ = decoder_lstm(decoderInputs, initial_state = decoder_states)

    # Dense projection to adjust the dimension
    decoderOutputs = Dense(nMel, activation='relu') (decoderOutputs)

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
def Seq2SeqModel_Encoder(nCategories, nTime, nMel):
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

    # First bidirectional layer that return only the full sequence that is the input of the next bidirectiona lstm layer
    encoder = Bidirectional(LSTM(int(nTime/2), return_sequences = True, return_state = False)) (encoder) # [b_s, seq_len, vec_dim]
    # Second bidirectional LSTM layer. This layer return the state of the bidirectional lstm. 
    # The output is a 5 tensor:
    #   - lstm: last output of the sequence [b_s, vec_dim]
    #   - forward_h: the last forward state h [b_s, 64]
    #   - forward_c: the last forward state c [b_s, 64]
    #   - backward_h: the last backward state h [b_s, 64]
    #   - backward_c: the last backward state c [b_s, 64]
    lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(int(nTime/2), return_sequences = False, return_state = True)) (encoder) 

    # We discard `encoder_outputs` and only keep the states.
    #encoder_states = [state_h, state_c]
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

print("\nAttention Model\n")
AttModel = AttentionModel(10, 28, 28)
AttModel.summary()

print("\nRNN Autoencoder Model\n")
LSTM_AE = LSTMAutoencoder(16, 28, 28)
LSTM_AE.summary()

print("\nRNN Autoencoder_Encoder Model\n")
LSTM_E = LSTMAutoencoder_Encoder(16, 28, 28)
LSTM_E.summary()

print("\nCNN+RNN Autoencoder Model\n")
Autoencoder = Seq2SeqModel(16, 125, 80)
Autoencoder.summary()

print("\nCNN+RNN Autoencoder_Encoder Model\n")
Autoencoder = Seq2SeqModel_Encoder(16, 125, 80)
Autoencoder.summary()