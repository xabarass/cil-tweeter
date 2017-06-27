from keras.models import Sequential, model_from_json
from keras.layers import Dense, MaxPooling1D
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Activation, Input, Bidirectional

class SingleLSTM:
    def __init__(self, params={}):
        print("Creating LSTM neural network")
        self.train_params = {"lstm_units": 200, "dropout": 0.5, "activation": "sigmoid"}
        self.train_params.update(params)

    def get_model(self, embedding_layer):
        model = Sequential()
        model.add(embedding_layer)
        model.add(LSTM(self.train_params["lstm_units"]))
        model.add(Dropout(self.train_params["dropout"]))
        model.add(Dense(1, activation=self.train_params["activation"]))

        return model

class DoubleConv:
    def __init__(self, params={}):
        self.train_params={}
        self.train_params.update(params)

    def get_model(self, embedding_layer):
        model = Sequential()
        model.add(embedding_layer)
        model.add(Convolution1D(100,
                                5,
                                padding='valid',
                                activation='relu'))
        model.add(MaxPooling1D())
        model.add(Convolution1D(200,
                                3,
                                padding='valid',
                                activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model