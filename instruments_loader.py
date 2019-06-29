import pickle

def load_from_bin():
    return pickle.load(open("instruments_binary.p", "rb"))

def txt_to_bin():
    txt_file = open("instruments.txt", "r")
    sorted_instruments = parse_text_file(txt_file)
    pickle.dump(sorted_instruments, open("instruments_binary.p", "wb"))

def parse_text_file(f):
    sorted_instruments = []
    for line in f:
        sorted_instruments.append(line.split()[0])

    return sorted_instruments