from joblib import Parallel, delayed
import pickle
import operator

from AIMusicGenerator.pixel_cnn_related.midi_to_img import get_input_paths, open_midi


def instrument_frequencies(midi_file):
    part_stream = midi_file.parts.stream()

    instruments = []
    excluded_instruments = ["Electric Bass", "Brass", "Acoustic Bass", "Fretless Bass", "Timpani"]

    for part in part_stream:
        if not part.partName is None and not part.partName in instruments and not part.partName in excluded_instruments:
            instruments.append(part.partName)

    return instruments


def do_work(path):
    try:
        name = path[path.rfind("\\") + 1:path.rfind(".")]
        print(name)
        song = open_midi(path)
        return instrument_frequencies(song)
    except:
        return None


def get_all_frequencies(paths):
    return Parallel(n_jobs=-1)(delayed(do_work)(p) for p in paths)


def save_frequencies(instruments):
    sorted_instruments_for_txt = sorted(instruments.items(), key=operator.itemgetter(1), reverse=True)
    sorted_instruments_for_bin = [item[0] for item in sorted_instruments_for_txt]

    with open("instruments/instruments_lstm.txt", "w") as txt_file:
        for item in sorted_instruments_for_txt:
            txt_file.write("{} : {}\n".format(item[0], item[1]))

    with open("instruments/instruments_lstm.bin", "wb") as bin_file:
        pickle.dump(sorted_instruments_for_bin, bin_file)


if __name__ == '__main__':
    paths = get_input_paths("lstm_midi")
    results = get_all_frequencies(paths)
    big_dict = {}

    for song in results:
        for instrument in song:
            if instrument in big_dict.keys():
                big_dict[instrument] += 1
            else:
                big_dict[instrument] = 1

    save_frequencies(big_dict)
