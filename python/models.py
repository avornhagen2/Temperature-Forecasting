
from tensorflow.python.keras.layers import LSTM, Dense, Conv1D
from tensorflow.python.keras.models import Sequential


# Allows for the creation and easy manipulation of a semi convolutional LSTM
# Pretty self explanitory, adds layers to the desired amount with the given parameters in a list. Fills out layers based
# on default if a full list is not given
# also assures that the output dense layer matches the size of the input to ensure prediction dimensions
def createCLSTM(filters=32, kernel_size=2, input_shape=(10, 2) ,num_conv_layers=1, lstmDims=50,  num_LSTM_layers=1, denseNodes=0, num_dense_layers=1, predictionDims=2):



    model = Sequential()
    if num_conv_layers > 1:
        for layNum in range(0, num_conv_layers):
            if isinstance(filters, list) and layNum < len(filters):
                model.add(Conv1D(filters=filters[layNum], kernel_size=kernel_size, strides=1, padding="causal",
                                 activation="relu",
                                 input_shape=input_shape))
            else:
                model.add(Conv1D(filters=32, kernel_size=kernel_size, strides=1, padding="causal",
                                 activation="relu",
                                 input_shape=input_shape))
    else:
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding="causal", activation="relu",
                         input_shape=input_shape))



    if num_LSTM_layers > 1:
        for layNum in range(0, num_LSTM_layers):
            if layNum == num_LSTM_layers - 1:
                try:
                    model.add(LSTM(units=lstmDims[layNum], activation='relu'))
                except:
                    model.add(LSTM(units=50, activation='relu'))
            elif isinstance(lstmDims, list) and layNum < len(lstmDims):
                model.add(LSTM(units=lstmDims[layNum], activation='relu', return_sequences=True))
            else:
                model.add(LSTM(units=50, activation='relu', return_sequences=True))


    else:
        model.add(LSTM(units=lstmDims, activation='relu'))



    if num_dense_layers > 1:
        for layNum in range(0, num_dense_layers):
            if layNum == num_dense_layers - 1:
                model.add(Dense(input_shape[1]))
            elif isinstance(denseNodes, list) and layNum < len(denseNodes):
                model.add(Dense(denseNodes[layNum]))
            else:
                model.add(Dense(predictionDims))
    else:
        model.add(Dense(predictionDims))

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model



def createLSTM(input_shape=(10, 2), lstmDims=50,  num_LSTM_layers=1, denseNodes=0, num_dense_layers=1, predictionDims=2):



    model = Sequential()


    if num_LSTM_layers > 1:
        for layNum in range(0, num_LSTM_layers):
            if layNum == 0:
                model.add \
                    (LSTM(units=lstmDims[layNum], activation='relu', input_shape=input_shape, return_sequences=True))

            elif layNum == num_LSTM_layers - 1:
                try:
                    model.add(LSTM(units=lstmDims[layNum], activation='relu'))
                except:
                    model.add(LSTM(units=50, activation='relu'))
            elif isinstance(lstmDims, list) and layNum < len(lstmDims):
                model.add(LSTM(units=lstmDims[layNum], activation='relu', return_sequences=True))
            else:
                model.add(LSTM(units=50, activation='relu', return_sequences=True))


    else:
        model.add(LSTM(units=lstmDims, activation='relu', input_shape=input_shape))



    if num_dense_layers > 1:
        for layNum in range(0, num_dense_layers):
            if layNum == num_dense_layers - 1:
                model.add(Dense(input_shape[1]))
            elif isinstance(denseNodes, list) and layNum < len(denseNodes):
                model.add(Dense(denseNodes[layNum]))
            else:
                model.add(Dense(predictionDims))
    else:
        model.add(Dense(predictionDims))

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model
