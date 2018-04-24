from keras.preprocessing.sequence import pad_sequences
import keras.utils

class Melody:

    def __init__(self, name, representation):
        self.name = name
        self.midi_pitches = representation['P']
        self.midi_durations = representation['T']
        self.integer_pitches = None
        self.integer_durations = None
        self.matrix_pitches = None
        self.matrix_durations = None

    def get_length(self):
        return len(self.midi_pitches)

    def get_name(self):
        return self.name

    def get_integer_representation(self):
        return (self.name, {'P': self.integer_pitches, 'T': self.integer_durations})

    def get_midi_representation(self):
        return (self.name, {'P': self.midi_pitches, 'T': self.midi_durations})

    def get_matrix_representation(self):
        return (self.name, {'P': self.matrix_pitches, 'T': self.matrix_durations})

    def get_std_matrix_representation(self):
        return (self.name, {'P': self.std_matrix_pitches, 'T': self.std_matrix_durations})

    def get_midi_durations(self):
        return self.midi_durations

    def extract_piches(self):
        return set(self.midi_pitches)

    def extract_durations(self):
        return set(self.midi_durations)

    def intersect_midi_durations(self, test_set):
        return set(test_set).intersection(set(self.midi_durations))

    def build_integer_representation(self, id_to_pitches, pitches_to_id, id_to_durations, durations_to_id):
        self.integer_pitches = [pitches_to_id[midi_pitch] for midi_pitch in self.midi_pitches]
        self.integer_durations = [durations_to_id[midi_duration] for midi_duration in self.midi_durations]

    def build_matrix_representation(self, number_of_pitches, number_of_durations):
        self.matrix_pitches = keras.utils.to_categorical(self.integer_pitches, number_of_pitches)
        self.matrix_durations = keras.utils.to_categorical(self.integer_durations, number_of_durations)

    def build_standardized_matrix_representation(self, max_length):
        self.std_matrix_pitches = pad_sequences(self.matrix_pitches.T, maxlen=max_length).T
        self.std_matrix_durations = pad_sequences(self.matrix_durations.T, maxlen=max_length).T
