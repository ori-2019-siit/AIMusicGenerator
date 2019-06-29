from music21 import midi, note, chord
from AIMusicGenerator.instruments_loader import load_from_bin
import fractions
import numpy as np
from PIL import Image

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
                pitches, notes_chords, total_duration = extract_notes(part)
                return pitches, notes_chords, total_duration


def extract_notes(part):
    pitches = []
    notes_chords = []

    total_duration = 0
    for nt in part.flat.notes:
        if isinstance(nt, note.Note):
            duration = nt.duration.quarterLength
            if isinstance(duration, fractions.Fraction):
                duration = round(float(duration), 2)
            total_duration += duration
            pitches.append((max(0.0, nt.pitch.ps), duration))
        elif isinstance(nt, chord.Chord):
            akord = []
            duration = nt.duration.quarterLength
            if isinstance(duration, fractions.Fraction):
                duration = round(float(duration), 1)
            total_duration += duration
            for pitch in nt.pitches:
                akord.append(((max(0.0, pitch.ps)), duration))
            pitches.append(akord)
        notes_chords.append(nt)

    return pitches, notes_chords, total_duration

def make_image(pitches):
    width, height = 300, 89
    data = np.zeros((height, width, 3), dtype=np.uint8)
    start = 0
    offset = 10
    channel = 0

    for element in pitches:
        notes = []
        if isinstance(element, list):
            # take first note from chord and it's duration, calculate number of sequential pixels needed
            pixels = int(element[0][1]*offset)
            for note in element:
                notes.append(int(note[0]))
        else:
            # equivalent for single note
            pixels = int(element[1]*offset)
            notes.append(int(element[0]))

        # check if it can fit on image
        if start + pixels >= width:
            channel += 1
            start = 0
            # we have filled every channel and need to return a value
            if channel == 3:
                return data

        for note in notes:
            ps = data[note][start:start + pixels + 1]
            for p in ps:
                p[channel] = 255
        start = start + pixels + 1

if __name__ == '__main__':
    path = "midi_songs\Kemal Malovcic\kemal malovcic - seti se.mid"
    instruments = load_from_bin()
    pesma = open_midi(path)
    pitches, notes_chords, total_duration = extract_line(pesma, instruments)
    img_array = make_image(pitches)
    img = Image.fromarray(img_array, "RGB")
    img.save("setiSe.png")