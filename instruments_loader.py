import pickle
import os


def load_from_bin(path):
    return pickle.load(open(path, "rb"))


def txt_to_bin():
    txt_file = open(os.path.join("instruments", "instruments_lstm.txt"), "r")
    sorted_instruments = parse_text_file(txt_file)
    pickle.dump(sorted_instruments, open(os.path.join("instruments", "instruments_lstm.bin"), "wb"))


def parse_text_file(f):
    sorted_instruments = []
    for line in f:
        sorted_instruments.append(line.split()[0])

    return sorted_instruments
