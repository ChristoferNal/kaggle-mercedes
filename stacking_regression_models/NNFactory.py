from keras.layers import InputLayer, GaussianNoise, Conv1D, Reshape, Dropout, LSTM, Bidirectional, regularizers
from keras.models import Sequential
from keras.optimizers import Adadelta

from MetricR2 import r2_keras


def get_bidirectional(num_of_features, neurons_conv=18, neurons=18, neurons2=20, noise=0.3, dropout=0.15, lr=1.05, rho=0.96):
    model = Sequential()
    model.add(InputLayer(input_shape=(num_of_features,)))

    model.add(Reshape((1, num_of_features)))

    model.add(Conv1D(neurons_conv, 1, activation="linear", input_shape=(1, num_of_features), padding="same", strides=1))

    model.add(GaussianNoise(noise))
    # keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0) 58.77% (+/- 3.81%)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 57.32% (+/- 3.70%)
    # keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 57.91% (+/- 3.34%)
    # keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004) 57.83% (+/- 3.99%)

    optimizer = Adadelta(lr=lr, rho=rho, epsilon=1e-08, decay=0.0)
    model.add(Bidirectional(
        LSTM(neurons, stateful=False, activation="tanh", consume_less='gpu',
             unroll=True,
             return_sequences=True), batch_input_shape=(None, 1, num_of_features)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(
        LSTM(neurons2, stateful=False, activation='tanh', consume_less='gpu', unroll=True,
             batch_input_shape=(None, 1, neurons),
             return_sequences=True)))
    model.add(
        LSTM(1, stateful=False, activation='linear', consume_less='gpu',
             unroll=True, batch_input_shape=(None, 1, 18)))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r2_keras])
    print(model.get_config())
    print("Trained model: bidirectional")
    return model


def get_bidirectional_no_conv(num_of_features, neurons=18, neurons2=20, dropout=0.15, lr=1.05, rho=0.96):
    model = Sequential()
    model.add(InputLayer(input_shape=(num_of_features,)))

    model.add(Reshape((1, num_of_features)))  # reshape into 4D tensor (samples, 1, maxlen, 256)

    optimizer = Adadelta(lr=lr, rho=rho, epsilon=1e-08, decay=0.0)
    model.add(Bidirectional(
        LSTM(neurons, stateful=False, activation="tanh", consume_less='gpu',
             unroll=True,
             return_sequences=True), batch_input_shape=(None, 1, num_of_features)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(
        LSTM(neurons2, stateful=False, activation='tanh', consume_less='gpu', unroll=True,
             batch_input_shape=(None, 1, neurons),
             return_sequences=True)))
    model.add(
        LSTM(1, stateful=False, activation='linear', consume_less='gpu',
             unroll=True, batch_input_shape=(None, 1, 18)))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r2_keras])
    print(model.get_config())
    print("Trained model: bidirectional")
    return model


def get_deep_bidirectional(num_of_features, reg1=0.01, reg2=0.01, neurons_conv=80, neurons=19, neurons2=20, noise=0.3, dropout=0.15,
                           lr=1.05, rho=0.96):
    model = Sequential()
    model.add(InputLayer(input_shape=(num_of_features,)))

    model.add(Reshape((1, num_of_features)))  # reshape into 4D tensor (samples, 1, maxlen, 256)

    model.add(Conv1D(neurons_conv, 1, activation="relu", input_shape=(1, num_of_features), padding="same", strides=1))

    #
    model.add(GaussianNoise(noise))
    # keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0) 58.77% (+/- 3.81%)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 57.32% (+/- 3.70%)
    # keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 57.91% (+/- 3.34%)
    # keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004) 57.83% (+/- 3.99%)

    optimizer = Adadelta(lr=lr, rho=rho, epsilon=1e-08, decay=0.0)
    model.add(Bidirectional(
        LSTM(neurons, stateful=False, activation="tanh", consume_less='gpu',
             unroll=True, recurrent_regularizer=regularizers.l1_l2(reg1, reg2),
             return_sequences=True, go_backwards=True), batch_input_shape=(None, 1, num_of_features)))

    model.add(Dropout(dropout))
    model.add(Bidirectional(
        LSTM(neurons2, stateful=False, activation='tanh', consume_less='gpu', unroll=True,
             batch_input_shape=(None, 1, neurons), recurrent_regularizer=regularizers.l1_l2(reg1, reg2),
             return_sequences=True, go_backwards=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(
        LSTM(neurons2, stateful=False, activation='tanh', consume_less='gpu', unroll=True,
             batch_input_shape=(None, 1, neurons), recurrent_regularizer=regularizers.l1_l2(reg1, reg2),
             return_sequences=True, go_backwards=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(
        LSTM(neurons2, stateful=False, activation='linear', consume_less='gpu', unroll=True,
             batch_input_shape=(None, 1, neurons), recurrent_regularizer=regularizers.l1_l2(reg1, reg2),
             return_sequences=True, go_backwards=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(
        LSTM(neurons2, stateful=False, activation='relu', consume_less='gpu', unroll=True,
             batch_input_shape=(None, 1, neurons), recurrent_regularizer=regularizers.l1_l2(reg1, reg2),
             return_sequences=True, go_backwards=True)))

    model.add(
        LSTM(1, stateful=False, activation='linear', consume_less='gpu',
             recurrent_regularizer=regularizers.l1_l2(reg1, reg2),
             unroll=True, batch_input_shape=(None, 1, 18), go_backwards=True))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r2_keras])
    print(model.get_config())
    print("Trained model: bidirectional")
    return model

