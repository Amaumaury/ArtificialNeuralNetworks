from melody import Melody
import numpy as np
import music21
from tqdm import tqdm_notebook
import json

from typing import List, Dict, Callable, Set, Tuple, Iterator

datapath = "chorpus/"

class Dataset:

    def __init__(self, data: Dict[str, Dict[str, List[int]]], build_matrix_representation=False) -> 'Dataset':
        self.melodies = [Melody(melody_name, representation) for melody_name, representation in list(data.items())]
        self.id_to_pitches, pitches_to_id = Dataset.build_mappings(self.extract_pitches())
        self.id_to_durations, durations_to_id = Dataset.build_mappings(self.extract_durations())

        for melody in self.melodies:
            melody.build_integer_representation(self.id_to_pitches, pitches_to_id, self.id_to_durations, durations_to_id)

        if build_matrix_representation:
            max_length = 0
            for melody in self.melodies:
                melody.build_matrix_representation(len(self.id_to_pitches), len(self.id_to_durations))
                max_length = max(max_length, len(melody))

            for melody in self.melodies:
                melody.build_standardized_matrix_representation(max_length)

    def matrix_to_integer_representation(self, melody):
        pitches = melody['P']
        durations = melody['T']

        pitches = [np.argwhere(vec_pitch == 1)[0][0] for vec_pitch in pitches if 1 in vec_pitch]
        durations = [np.argwhere(vec_duration == 1)[0][0] for vec_duration in durations if 1 in vec_duration]

        return pitches, durations

    def get_training_arrays(self):
        pitches_training_array, durations_training_array = zip(*[melody.get_feeding_representation() for melody in self.melodies])
        return np.array(pitches_training_array), np.array(durations_training_array)

    def get_melodies_length(self):
        return {melody.get_name(): len(melody) for melody in self.melodies}

    def get_max_melody_len(self):
        return max([len(melody) for melody in self.melodies])

    def get_mappings(self) -> Dict[str, Dict[int, int]]:
        return {'P': self.id_to_pitches, 'T': self.id_to_durations}

    def with_matrix_representation(self):
        return Dataset(self.get_midi_representation(), build_matrix_representation=True)

    def filter(self, condition: Callable[[Melody], bool]) -> 'Dataset':
        melodies = map(lambda melody: melody.get_midi_representation(), filter(condition, self.melodies))
        return Dataset({name: representation for name, representation in melodies})

    def apply_to_melody(self, map_: Callable[[Melody], Melody]) -> 'Dataset':
        mapped_melodies = [map_(melody).get_midi_representation() for melody in self.melodies]
        return Dataset({name: representation for name, representation in mapped_melodies})

    def delete_by_name(self, melody_name: str) -> 'Dataset':
        return self.filter(lambda melody: melody.get_name() != melody_name)

    def __len__(self) -> int:
        return len(self.melodies)

    def __iter__(self) -> Iterator[Melody]:
        return iter(self.melodies)

    def contains(self, melody_name):
        return sum([melody.get_name() == melody_name for melody in self.melodies]) > 0

    def get_n_random_melodies(self, n: int, seed=0) -> List[Melody]:
        np.random.seed(seed)
        return [melody for melody in np.random.choice(self.melodies, n)]

    def get_midi_representation(self) -> Dict[str, Dict[str, List[int]]]:
        melodies = [melody.get_midi_representation() for melody in self.melodies]
        return {name: representation for name, representation in melodies}

    def get_integer_representation(self):
        melodies = [melody.get_integer_representation() for melody in self.melodies]
        return {name: representation for name, representation in melodies}

    def extract_pitches(self) -> Set[int]:
        return set().union(*[melody.extract_pitches() for melody in self.melodies])

    def get_all_midi_durations(self) -> List[int]:
        flatten = lambda l: [item for sublist in l for item in sublist]
        return flatten([melody.get_midi_durations() for melody in self.melodies])

    def get_all_midi_pitches(self) -> List[int]:
        flatten = lambda l: [item for sublist in l for item in sublist]
        return flatten([melody.get_midi_pitches() for melody in self.melodies])

    def extract_durations(self) -> Set[int]:
        return set().union(*[melody.extract_durations() for melody in self.melodies])

    def transposeDataset(self) -> 'Dataset':
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

    def write_to_file(self, file_name):
        melodies = [melody.get_midi_representation() for melody in self.melodies]
        melodies = {name: representation for name, representation in melodies}

        with open(file_name, 'w') as file:
            file.write(json.dumps(melodies))
            file.close()

    @staticmethod
    def load_from_file(file_name):
        tmp_dataset = None

        with open(file_name, 'r') as file:
            tmp_dataset = Dataset(json.load(file), build_matrix_representation=True)

        return tmp_dataset

    @staticmethod
    def build_mappings(possible_values: Set[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
        inv_map = lambda dict_: {v: k for k, v in dict_.items()}
        ids = range(len(possible_values))
        id_to_value = dict(zip(ids, sorted(possible_values)))
        return id_to_value, inv_map(id_to_value)
