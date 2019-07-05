from music21 import note, chord, instrument, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import pickle
import random
import numpy as np


def map_int_to_notes(pitches):
    mapped_int = {}
    counter = 0
    for music_element in pitches:
        mapped_int[counter] = music_element
        counter += 1
    return mapped_int


def map_notes_to_int(pitches):
    mapped_notes = {}
    counter = 0
    for music_element in pitches:
        mapped_notes[music_element] = counter
        counter += 1
    return mapped_notes


def create_model(input, total_pitches, weights):
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
    model.load_weights(weights)

    return model


def create_list_of_sequences(notes, mapped_notes, notes_sequence_length, total_pitches):
    list_of_sequences = []
    output = []
    for i in range(0, len(notes) - notes_sequence_length, 1):
        input_sequence = notes[i:i + notes_sequence_length]
        output_of_sequence = notes[i + notes_sequence_length]
        list_of_sequences.append([mapped_notes[elem] for elem in input_sequence])
        output.append(mapped_notes[output_of_sequence])

    num_of_sequences = len(list_of_sequences)

    norm_input = np.reshape(list_of_sequences, (num_of_sequences, notes_sequence_length, 1))
    norm_input = norm_input / total_pitches

    return list_of_sequences, norm_input


def generate_sequence(model, mapped_int, input, total_pitches, num_of_notes):
    random_sequence = random.randint(0, len(input) - 1)
    sequence = input[random_sequence]
    generated_sequence = []
    for note in range(num_of_notes):
        input_sequence = np.reshape(sequence, (1, len(sequence), 1))
        input_sequence = input_sequence / total_pitches
        output = model.predict(input_sequence, verbose=0)
        note_as_int = np.argmax(output)
        output_note = mapped_int[note_as_int]
        generated_sequence.append(output_note)
        sequence.append(note_as_int)
        sequence = sequence[1:len(sequence)]
    return generated_sequence


def generate_midi(generated_sequence):
    generated_notes = []
    offset = 0
    for music_element in generated_sequence:
        if '.' in music_element:
            notes = music_element.split('.')
            notes_from_chord = []
            for n in notes:
                gen_note = note.Note(n)
                notes_from_chord.append(gen_note)
            gen_chord = chord.Chord(notes_from_chord)
            gen_chord.offset = offset
            generated_notes.append(gen_chord)
        else:
            gen_note = note.Note(music_element)
            gen_note.offset = offset
            generated_notes.append(gen_note)
        offset += 0.5

    midi_stream = stream.Stream(generated_notes)
    midi_stream.write('midi', fp='generated_music.mid')


if __name__ == '__main__':
    notes_sequence_length = 100
    input = pickle.load(open("input_binary.p", "rb"))
    notes = pickle.load(open("notes.p", "rb"))
    pitches = pickle.load(open("pitches.p", "rb"))
    weights = "weights-18-6.1185.hdf5"
    mapped_int = map_int_to_notes(pitches)
    mapped_notes = map_notes_to_int(pitches)
    list_of_sequences, norm_inp = create_list_of_sequences(notes, mapped_notes, notes_sequence_length, len(pitches))
    model = create_model(norm_inp, len(pitches), weights)
    generated_sequence = generate_sequence(model, mapped_int, list_of_sequences, len(pitches), 500)
    generate_midi(generated_sequence)
