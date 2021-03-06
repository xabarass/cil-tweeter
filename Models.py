from keras.models import Sequential, Model
from keras.layers import Dense, MaxPooling1D
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Activation, Input, Bidirectional
from keras.layers.merge import Concatenate

class SingleLSTM:
    def __init__(self, params={}):
        print("Creating LSTM neural network factory")
        self.train_params = {"lstm_units": 200, "dropout": 0.5, "activation": "sigmoid"}
        self.train_params.update(params)

    def get_model(self, embedding_layer):
        model = Sequential()
        model.add(embedding_layer)
        model.add(LSTM(self.train_params["lstm_units"], implementation=2))
        model.add(Dropout(self.train_params["dropout"]))
        model.add(Dense(1, activation=self.train_params["activation"]))

        return model

class BidirectionalLSTM:
    def __init__(self, params={}):
        print("Creating bidirectional LSTM neural network factory")
        self.train_params = {"lstm_units": 200, "dropout": 0.5, "activation": "sigmoid"}
        self.train_params.update(params)

    def get_model(self, embedding_layer):
        forward_input = Input(shape=(embedding_layer.input_length,),dtype='int64', name='forward_input')
        emb_forward_input = embedding_layer(forward_input)
        lstmout_forward_input = LSTM(self.train_params["lstm_units"], implementation=2)(emb_forward_input)

        backward_input = Input(shape=(embedding_layer.input_length,),dtype='int64', name='backward_input')
        emb_backward_input = embedding_layer(backward_input)
        lstmout_backward_input = LSTM(self.train_params["lstm_units"], implementation=2)(emb_backward_input)

        lstmout_both_dir = Concatenate()([lstmout_forward_input,lstmout_backward_input])
        dropout_both_dir = Dropout(self.train_params["dropout"])(lstmout_both_dir)
        probability = Dense(1, activation=self.train_params["activation"])(dropout_both_dir)
        model = Model(inputs=[forward_input,backward_input], outputs=[probability])

        return model


class DoubleConv:
    def __init__(self, params={}):
        print("Creating DoubleConv neural network factory")
        self.train_params={}
        self.train_params.update(params)

    def get_model(self, embedding_layer):
        model = Sequential()
        model.add(embedding_layer)
        model.add(Convolution1D(200,
                                5,
                                padding='valid',
                                activation='relu'))
        model.add(MaxPooling1D())
        model.add(Convolution1D(100,
                                3,
                                padding='valid',
                                activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model


class LSTMConvEnsemble:
    boosting_iterate = 0

    def __init__(self, params={}):
        print("Creating LSTM DoubleConv Ensemble neural network factory")
        self.train_params={}
        self.train_params.update(params)
        self.lstm_builder = SingleLSTM()
        self.conv_builder = DoubleConv()

    def get_model(self, embedding_layer):
        if LSTMConvEnsemble.boosting_iterate == 0:
            print("Creating LSTM recurrent neural network")
            model = self.lstm_builder.get_model(embedding_layer=embedding_layer)
        else:
            print("Creating convolutional neural network")
            model = self.conv_builder.get_model(embedding_layer=embedding_layer)
        LSTMConvEnsemble.boosting_iterate += 1
        return model

class CharacterEmbeddingConv:
    def __init__(self):
        print("Creating character convolutional neural network factory")
        pass

    def get_model(self, embedding_layer):
        print("Creating character convolutional neural network")
        model = Sequential()
        model.add(embedding_layer)
        model.add(Convolution1D(1024,
                                10,
                                padding='valid',
                                activation='relu'))
        model.add(MaxPooling1D())
        model.add(Convolution1D(2048,
                                5,
                                padding='valid',
                                activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model
