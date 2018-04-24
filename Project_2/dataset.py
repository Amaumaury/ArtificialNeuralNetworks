from melody import Melody
import numpy as np
import music21
from tqdm import tqdm_notebook

datapath = "chorpus/"

class Dataset:

    def __init__(self, data, build_matrix_representation=False):
        self.melodies = [Melody(melody_name, representation) for melody_name, representation in list(data.items())]
        self.id_to_pitches, pitches_to_id = self.__build_mappings(self.extract_piches())
        self.id_to_durations, durations_to_id = self.__build_mappings(self.extract_durations())

        if build_matrix_representation:
            max_length = 0
            for melody in self.melodies:
                melody.build_integer_representation(self.id_to_pitches, pitches_to_id, self.id_to_durations, durations_to_id)
                melody.build_matrix_representation(len(self.id_to_pitches), len(self.id_to_durations))
                max_length = max(max_length, melody.get_length())

            for melody in self.melodies:
                melody.build_standardized_matrix_representation(max_length)

    def get_mappings(self):
        return {'P': self.id_to_pitches, 'T': self.id_to_durations}

    def with_matrix_representation(self):
        return Dataset(self.get_midi_representation(), True)

    def filter(self, condition):
        filtered_melodies = [melody.get_midi_representation() for melody in self.melodies if condition(melody)]
        return Dataset({name: representation for name, representation in filtered_melodies})

    def apply_to_melody(self, map_):
        mapped_melodies = [map_(melody).get_midi_representation() for melody in self.melodies]
        return Dataset({name: representation for name, representation in mapped_melodies})

    def delete_by_name(self, melody_name):
        return self.filter(lambda melody: melody.get_name() != melody_name)

    def get_number_of_melodies(self):
        return len(self.melodies)

    def get_n_random_melodies(self, n, seed=0):
        np.random.seed(seed)
        return [melody for melody in np.random.choice(self.melodies, n)]

    def get_midi_representation(self):
        melodies = [melody.get_midi_representation() for melody in self.melodies]
        return {name: representation for name, representation in melodies}

    def get_integer_representation(self):
        melodies = [melody.get_integer_representation() for melody in self.melodies]
        return {name: representation for name, representation in melodies}

    def get_all_midi_durations(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        return flatten([melody.get_midi_durations() for melody in self.melodies])

    def extract_piches(self):
        return set().union(*[melody.extract_piches() for melody in self.melodies])

    def extract_durations(self):
        return set().union(*[melody.extract_durations() for melody in self.melodies])

    def __build_mappings(self, possible_values):
        inv_map = lambda dict_: {v: k for k, v in dict_.items()}
        ids = range(len(possible_values))
        id_to_value = dict(zip(ids, sorted(possible_values)))
        return id_to_value, inv_map(id_to_value)

    def transposeDataset(self):
        transposed_dataset = {}

        dataset = self.get_midi_representation()

        for label in tqdm_notebook(list(dataset.keys())):
            # sessiontune32822 throws an error when trying to parse the file
            if label != 'sessiontune32822':
                transposed_dataset[label] = {}
                score = music21.converter.parseFile(datapath + label + ".mid")
                key = score.analyze('key')
                if key.mode == "major":
                    i = music21.interval.Interval(key.tonic, music21.pitch.Pitch('C'))
                elif key.mode == "minor":
                    i = music21.interval.Interval(key.tonic, music21.pitch.Pitch('A'))
                i = i.semitones
                transposed_dataset[label]['P'] = [p + i for p in dataset[label]['P']]
                transposed_dataset[label]['T'] = dataset[label]['T']

        return Dataset(transposed_dataset)
