from music21 import note, chord
from joblib import Parallel, delayed
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pickle
from AIMusicGenerator.pixel_cnn_related.midi_to_img import open_midi, load_from_bin, get_input_paths


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
    input = input / len(pitches)
    output = np_utils.to_categorical(output)

    print(input)
    print(output)

    return input, output


def create_model(input, total_pitches):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(input.shape[1], input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(total_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def train_network(model, input, output):
    weights_path = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(input, output, epochs=200, batch_size=64, callbacks=callbacks_list)


if __name__ == '__main__':
    notes_sequence_length = 100
    instruments = load_from_bin(os.path.join("instruments", "instruments_lstm.bin"))

    notes = pickle.load(open(os.path.join("notes", "notes.p"), "rb"))
    pitches = sorted(set(notes))

    input, output = create_input_and_output(notes_sequence_length, pitches, notes)
    print(input)
    print(output)
    # Radule TODO: sacuvaj input i output sa pickle i stavi da se trenira
    """
    model = create_model(input, len(pitches))
    train_network(model, input, output)
    """
