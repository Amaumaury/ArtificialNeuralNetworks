from keras.preprocessing.sequence import pad_sequences
import keras.utils

from typing import List, Dict, Set, Tuple, Iterable
class Melody:
    def __init__(self, name: str, representation: Dict['str', List[int]]) -> 'Melody':
        self.name = name
        self.midi_pitches = representation['P']
        self.midi_durations = representation['T']
        self.integer_pitches = None
        self.integer_durations = None
        self.matrix_pitches = None
        self.matrix_durations = None

    def __len__(self) -> int:
        return len(self.midi_pitches)

    def get_name(self) -> str:
        return self.name

    def get_integer_representation(self) -> Tuple[str, Dict[str, List[int]]]:
        return (self.name, {'P': self.integer_pitches, 'T': self.integer_durations})

    def get_midi_representation(self) -> Tuple[str, Dict[str, List[int]]]:
        return (self.name, {'P': self.midi_pitches, 'T': self.midi_durations})

    def get_matrix_representation(self):
        if self.matrix_pitches is None or self.matrix_durations is None:
            self.build_matrix_representation()
        # TODO: Add type
        return (self.name, {'P': self.matrix_pitches, 'T': self.matrix_durations})

    def get_std_matrix_representation(self):
        return (self.name, {'P': self.std_matrix_pitches, 'T': self.std_matrix_durations})

    def get_midi_durations(self) -> List[int]:
        return self.midi_durations

    def extract_pitches(self) -> List[int]:
        return set(self.midi_pitches)

    def extract_durations(self) -> Set[int]:
        return set(self.midi_durations)

    def intersect_midi_durations(self, test_set: Iterable[int]) -> Set[int]:
        return set(test_set).intersection(set(self.midi_durations))

    def build_integer_representation(self, id_to_pitches: Dict[int, int], pitches_to_id: Dict[int, int],
         id_to_durations: Dict[int, int], durations_to_id: Dict[int, int]):
        self.integer_pitches = [pitches_to_id[midi_pitch] for midi_pitch in self.midi_pitches]
        self.integer_durations = [durations_to_id[midi_duration] for midi_duration in self.midi_durations]

    def build_matrix_representation(self, number_of_pitches: int, number_of_durations: int):
        self.matrix_pitches = keras.utils.to_categorical(self.integer_pitches, number_of_pitches)
        self.matrix_durations = keras.utils.to_categorical(self.integer_durations, number_of_durations)

    def build_standardized_matrix_representation(self, max_length: int):
        self.std_matrix_pitches = pad_sequences(self.matrix_pitches.T, maxlen=max_length).T
        self.std_matrix_durations = pad_sequences(self.matrix_durations.T, maxlen=max_length).T

