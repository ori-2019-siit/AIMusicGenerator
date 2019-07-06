from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import pickle


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
    pitches = pickle.load(open("notes/pitches.p", "rb"))
    input = pickle.load(open("notes/input_binary.p", "rb"))
    output = pickle.load(open("notes/output_binary.p", "rb"))
    model = create_model(input, len(pitches))
    train_network(model, input, output)
