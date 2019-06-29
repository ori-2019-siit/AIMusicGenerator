from music21 import converter, instrument
import glob
import operator

music_lines = {}
counter = 0
for file in glob.glob("midi_songs/*/*.mid"):
    try:
        midi = converter.parse(file)
    except:
        continue
    notes_to_parse = None
    try:
        parts = instrument.partitionByInstrument(midi)
        for p in parts:
            if str(p.getInstrument(True, True)) in music_lines:
                music_lines[str(p.getInstrument(True, True))] += 1
            else:
                music_lines[str(p.getInstrument(True, True))] = 1
    except:
        pass
    counter += 1
    if counter % 100 == 0:
        f = open("instruments.txt", "w")
        f.write(str(counter) + "\n")
        for m in music_lines:
            f.write(m + ":" + str(music_lines[m]) + "\n")
        f.close()

sorted_instruments = sorted(music_lines.items(), key=operator.itemgetter(1), reverse=True)
f = open("instruments.txt", "w")
for i in sorted_instruments:
    f.write(i[0] + " : " + str(i[1]) + "\n")
f.close()
