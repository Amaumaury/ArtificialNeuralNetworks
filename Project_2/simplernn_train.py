from dataset import Dataset

import keras
from keras.models import Model, load_model
from keras.layers import Input, Masking, TimeDistributed, Dense, Concatenate, Dropout, LSTM, GRU, SimpleRNN, Lambda
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint


# Load data
final_dataset = Dataset.load_from_file('final_dataset')
max_melody_length = final_dataset.get_max_melody_len()
possible_pitches = len(final_dataset.extract_pitches())
possible_durations = len(final_dataset.extract_durations())
hack_dict = {'P': [0] * possible_pitches, 'T': [0] * possible_durations}

# Prepare data
x_pitches_training_array, x_durations_training_array = final_dataset.get_training_arrays()

y_pitches_training_array = x_pitches_training_array[:, 1:]
y_durations_training_array = x_durations_training_array[:, 1:]

x_pitches_training_array = x_pitches_training_array[:, :-1]
x_durations_training_array = x_durations_training_array[:, :-1]

# Copy paste function to build model
def buildModel(dictionaries, batch_length, dropout=0.2, activation='GRU', Hsize=128):
    X = dict()
    H = dict()
    M = dict()
    Y = dict()
    
    X['T'] = Input(shape=(batch_length, len(dictionaries['T'])), name="XT")
    X['P'] = Input(shape=(batch_length, len(dictionaries['P'])), name="XP")
    
    M['T'] = Masking(mask_value=0., name="MT")(X['T'])
    M['P'] = Masking(mask_value=0., name="MP")(X['P'])
    
    H['1'] = Concatenate(name="MergeX")([M['T'], M['P']])
    
    if activation == 'GRU':
        rnn_layer = GRU(Hsize, dropout=dropout, return_sequences=True)(H['1'])
    elif activation == 'LSTM':
        ...
        #Your hidden layer(s) architecture with LSTM (For your own curiosity, not required for the project)
    elif activation == 'RNN':
        rnn_layer = SimpleRNN(Hsize, dropout=dropout, return_sequences=True)(H['1'])
        
    print(rnn_layer)

    Y['T'] = TimeDistributed(Dense(len(dictionaries['T']), activation='softmax'), name='YT')(rnn_layer)
    Y['P'] = TimeDistributed(Dense(len(dictionaries['P']), activation='softmax'), name='YP')(rnn_layer)
    
    model = Model(inputs = [X['T'], X['P']], outputs = [Y['T'], Y['P']])
    opt = Adam() 
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=opt,
        metrics=['accuracy'])
    
    
    return model

# Create model
RNNmodel = buildModel(hack_dict, 
                      batch_length=max_melody_length-1,#Put here the number of notes (timesteps) you have in your Zero-padded matrices
                      activation='GRU')

print(RNNmodel.summary())

# Train
history = RNNmodel.fit(
    x=[x_durations_training_array, x_pitches_training_array],
    y=[y_durations_training_array, y_pitches_training_array],
    epochs=250,
    validation_split=0.2
)

# Save
RNNmodel.save('gru.hdf5')
