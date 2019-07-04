from music21 import note, chord
from instruments_loader import load_from_bin
from midi_to_img import open_midi
from joblib import Parallel, delayed
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import pickle
import os
import time


def extract_instrument_line(midi_file, instruments_list):
    part_stream = midi_file.parts.stream()

    for instrument in instruments_list:
        for part in part_stream:
            if instrument == part.partName:
                return part.recurse()


def process_midi_file(path, instruments_list):
    notes = []
    try:
        song = open_midi(path)
        instrument_line = extract_instrument_line(song, instruments_list)
        for music_element in instrument_line:
            if isinstance(music_element, note.Note):
                notes.append(str(music_element.pitch))
            elif isinstance(music_element, chord.Chord):
                notes.append('.'.join(str(n) for n in music_element.pitches))
        return notes
    except Exception as e:
        return []


def get_notes(paths, instruments_list):
    results = Parallel(n_jobs=-1)(delayed(process_midi_file)(p, instruments_list) for p in paths)
    return results


def map_notes_to_int(pitches):
    mapped_notes = {}
    counter = 0
    for music_element in pitches:
        mapped_notes[music_element] = counter
        counter += 1
    return mapped_notes


def create_input_and_output(notes_sequence_length, mapped_notes, total_pitches, notes):
    input, output = [], []
    for i in range(0, len(notes) - notes_sequence_length, 1):
        input_sequence = notes[i:i + notes_sequence_length]
        output_of_sequence = notes[i + notes_sequence_length]
        for n in input_sequence:
            input.append(mapped_notes[n])
        output.append(mapped_notes[output_of_sequence])

    num_of_sequences = len(output)
    input[:] = [elem / total_pitches for elem in input]
    input = np.reshape(input, (num_of_sequences, notes_sequence_length, 1))
    output = np_utils.to_categorical(output)

    return input, output


def get_input_paths(directory):
    paths = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            paths.append(os.path.join(root, name))

    return paths


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
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(total_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train_network(model, input, output):
    weights_path = "weights/weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(input, output, epochs=200, batch_size=64, callbacks=callbacks_list)


if __name__ == '__main__':
    notes_sequence_length = 100
    instruments = load_from_bin()
    paths = get_input_paths("lstm_midi")
    batch_size = 29
    batches = [paths[i * batch_size:(i + 1) * batch_size] for i in range((len(paths) + batch_size - 1) // batch_size)]
    counter = 0
    start_time = time.time()
    # load all notes and chords from midi files
    for batch in batches:
        results = get_notes(batch, instruments)
        results = [item for sublist in results for item in sublist]
        pickle.dump(results, open(os.path.join("notes", str(counter) + ".p"), "wb"))
        counter += 1

    print("Time elapsed: {} seconds".format(time.time() - start_time))

    # merge all in one list
    notes = []
    paths = get_input_paths("notes")
    for path in paths:
        temp = pickle.load(open(path, "rb"))
        notes += temp
    pickle.dump(notes, open("notes.p", "wb"))

    pitches = set(notes)
    # map pitches to int
    mapped_notes = map_notes_to_int(pitches)
    # create input and output for network
    input, output = create_input_and_output(notes_sequence_length, mapped_notes, len(pitches), notes)
    pickle.dump(input, open("input_binary.p", "wb"))
    pickle.dump(output, open("output_binary.p", "wb"))
    # create model and train network
    model = create_model(input, len(pitches))
    train_network(model, input, output)
