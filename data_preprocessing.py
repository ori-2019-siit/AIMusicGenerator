from music21 import note, chord
from joblib import Parallel, delayed
from keras.utils import np_utils
from pixel_cnn_related.midi_to_img import open_midi, get_input_paths
from instruments_loader import load_from_bin
import numpy as np
import pickle
import os


def extract_line(song, instruments_list):
    part_stream = song.parts.stream()

    for instrument in instruments_list:
        for part in part_stream:
            if instrument == part.partName:
                return part

    # some midi files have no instrument names, so sample the first part
    return part_stream[0]


def process_midi_file(path, instruments_list):
    notes = []
    try:
        song = open_midi(path)
        print(path)
        instrument_line = extract_line(song, instruments_list)
        for music_element in instrument_line:
            if isinstance(music_element, note.Note):
                notes.append(str(music_element.pitch))
            elif isinstance(music_element, chord.Chord):
                notes.append('.'.join(str(n) for n in music_element.pitches))
        return notes
    except:
        return []


def get_notes(paths, instruments_list):
    results = Parallel(n_jobs=-1)(delayed(process_midi_file)(p, instruments_list) for p in paths)
    return results


def create_input_and_output(notes_sequence_length, pitches, notes):
    input, output = [], []

    for i in range(0, len(notes) - notes_sequence_length, 1):
        input_sequence = notes[i:i + notes_sequence_length]
        output_of_sequence = notes[i + notes_sequence_length]
        input.append([pitches.index(char) for char in input_sequence])
        output.append(pitches.index(output_of_sequence))

    num_of_sequences = len(input)
    input = np.reshape(input, (num_of_sequences, notes_sequence_length, 1))
    input /= len(pitches)
    output = np_utils.to_categorical(output)

    return input, output


if __name__ == '__main__':
    notes_sequence_length = 100
    instruments = load_from_bin(os.path.join("instruments", "instruments_lstm.bin"))
    paths = get_input_paths("lstm_midi")
    batch_size = 32
    batches = [paths[i * batch_size:(i + 1) * batch_size] for i in range((len(paths) + batch_size - 1) // batch_size)]
    counter = 0

    # load all notes and chords from midi files
    for batch in batches:
        results = get_notes(batch, instruments)
        results = [item for sublist in results for item in sublist]
        pickle.dump(results, open(os.path.join("notes", str(counter) + ".p"), "wb"))
        counter += 1

    # merge all in one list
    notes = []
    paths = get_input_paths("notes")
    for path in paths:
        temp = pickle.load(open(path, "rb"))
        notes += temp
    pickle.dump(notes, open("notes.p", "wb"))
    pitches = sorted(set(notes))
    pickle.dump(pitches, open("notes/pitches.p", "wb"))
    input, output = create_input_and_output(notes_sequence_length, pitches, notes)
    pickle.dump(input, open("notes/input_binary.p", "wb"))
    pickle.dump(output, open("notes/output_binary.p", "wb"))
