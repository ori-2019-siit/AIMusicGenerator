from music21 import note, chord, stream
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path)
    img.load()
    img = img.rotate(-90)
    return np.asarray(img, dtype=np.uint8)


def make_pitches(path):
    data = load_image(path)
    offset = 10
    music = stream.Stream()

    height, width, channels = data.shape

    for channel in range(channels):
        notes = {}  # timestamp 0: [n1, n2]  n = (pitch, duration)
        active_pitches = {p: None for p in range(height)}  # for every pitch, timestamp key pitch : timestamp
        for w in range(width):
            column = data[:, w]

            for pitch in range(height):
                if column[pitch][channel] == 255:
                    if w not in notes.keys():
                        notes[w] = {}
                    if active_pitches[pitch] is None:
                        notes[w][pitch] = 1
                        active_pitches[pitch] = w
                    else:
                        notes[active_pitches[pitch]][pitch] += 1
                else:
                    active_pitches[pitch] = None

        for timestamp in sorted(notes.keys()):
            if len(notes[timestamp]) == 1:
                n = note.Note()
                n.pitch.ps = list(notes[timestamp].keys())[0] + 46
                n.duration.quarterLength = list(notes[timestamp].values())[0] / offset
                music.append(n)
            else:
                durations = {}

                for pitch, duration in notes[timestamp].items():
                    if duration not in durations.keys():
                        durations[duration] = [pitch+46]
                    else:
                        durations[duration].append(pitch+46)

                for duration, pitches in durations.items():
                    if len(pitches) == 1:
                        n = note.Note()
                        n.pitch.ps = pitches[0] + 46
                        n.duration.quarterLength = duration/offset
                        music.append(n)
                    else:
                        c = chord.Chord(pitches)
                        c.duration.quarterLength = duration/offset
                        music.append(c)

    return music


def save_midi(stream, path):
    name = path[path.rfind("/") + 1:path.rfind(".")]
    stream.write("midi", "../generated_midis/{}.mid".format(name))


def image_to_midi(path):
    save_midi(make_pitches(path), path)

if __name__ == '__main__':
    image_to_midi("../generated_images/train/806.png")