from music21 import midi, note, chord
import pickle
import fractions
import numpy as np
from PIL import Image
import os
from joblib import Parallel, delayed
import time

def load_from_bin():
    return pickle.load(open(os.path.join("..\\instruments", "instruments_lstm.bin"), "rb"))

def open_midi(midi_path, remove_drums=True):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()

    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return midi.translate.midiFileToStream(mf)


def extract_line(midi_file, instruments_list):
    part_stream = midi_file.parts.stream()

    for instrument in instruments_list:
        for part in part_stream:
            if instrument == part.partName:
                pitches = extract_notes(part)
                return pitches

    # some midi files have no instrument names, so sample the first part
    pitches = extract_notes(part_stream[0])
    return pitches


def extract_notes(part):
    pitches = []

    for nt in part.flat.notes:
        if isinstance(nt, note.Note):
            duration = nt.duration.quarterLength
            if isinstance(duration, fractions.Fraction):
                duration = round(float(duration), 1)
            pitches.append((max(0.0, nt.pitch.ps), duration))
        elif isinstance(nt, chord.Chord):
            akord = []
            duration = nt.duration.quarterLength
            if isinstance(duration, fractions.Fraction):
                duration = round(float(duration), 1)
            for pitch in nt.pitches:
                akord.append(((max(0.0, pitch.ps)), duration))
            pitches.append(akord)

    return pitches


def make_image(pitches, name):
    counter = 0
    width, height = 256, 256
    data = np.zeros((height, width, 3), dtype=np.uint8)
    start = 0
    offset = 10
    channel = 0
    images = []

    for element in pitches:
        notes = []
        if isinstance(element, list):
            # take first note from chord and it's duration, calculate number of sequential pixels needed
            pixels = int(element[0][1] * offset)
            for note in element:
                notes.append(int(note[0]))
        else:
            # equivalent for single note
            pixels = int(element[1] * offset)
            notes.append(int(element[0]))

        # check if it can fit on image
        if not start + pixels < width:
            channel += 1
            start = 0
            # we have filled every channel and need to return a value
            if channel == 3:
                img = Image.fromarray(data, "RGB")
                img = img.rotate(90)
                images.append(img)
                data = np.zeros((height, width, 3), dtype=np.uint8)
                start = 0
                channel = 0
                counter += 1

        for note in notes:
            ps = data[note][start:start + pixels + 1]
            for p in ps:
                p[channel] = 255
        start = start + pixels + 1

    return images


def do_work(path, instruments):
    try:
        name = path[path.rfind("\\") + 1:path.rfind(".")]
        print(name)
        song = open_midi(path)
        pitches = extract_line(song, instruments)
        return make_image(pitches, name)
    except Exception as e:
        print(str(e))
        return None


def get_input_paths(path):
    paths = []

    for root, dirs, files in os.walk(path):
        for name in files:
            paths.append(os.path.join(root, name))

    return paths


def make_all_images(paths, num_cores, instruments):
    results = Parallel(n_jobs=num_cores)(delayed(do_work)(p, instruments) for p in paths)
    return results

if __name__ == '__main__':
    instruments = load_from_bin()
    paths = get_input_paths("..\\lstm_midi")

    batch_size = 32
    counter = 0

    batches = [paths[i * batch_size:(i + 1) * batch_size] for i in range((len(paths) + batch_size - 1) // batch_size )]

    start_time = time.time()
    for batch in batches:
        results = make_all_images(batch, -1, instruments)

        for song_images in results:
            if not song_images is None:
                for part in song_images:
                    name = str(counter) + ".png"
                    part.save(os.path.join("..\\generated_images", name))
                    counter += 1

    print("Time elapsed: {} seconds".format(time.time() - start_time))