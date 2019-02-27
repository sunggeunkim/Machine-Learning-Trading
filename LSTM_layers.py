from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import regularizers

class LSTM_keras:

    def __init__(self, input_shape, nodes = [50, 50], l2 = 0.001):
        # Initialising the RNN
        self.clf = Sequential()

        for i in range(len(nodes)):
            if i == 0:
                self.clf.add(LSTM(units = nodes[i], 
                   return_sequences = True, 
                   input_shape = input_shape, 
                   kernel_regularizer=regularizers.l2(l2)))
            elif i == len(nodes) - 1:
                self.clf.add(LSTM(units = nodes[i], 
                   return_sequences = False, 
                   kernel_regularizer=regularizers.l2(l2)))
            else:
                self.clf.add(LSTM(units = nodes[i], 
                   return_sequences = True, 
                   kernel_regularizer=regularizers.l2(l2)))


        # Adding the output layer
        self.clf.add(Dense(units = 1, activation="sigmoid"))

        # Compiling the RNN
        self.clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, val_portion = 0.2, n_epochs = 100, n_batch = 32):
        # Fitting the RNN to the Training set
        return self.clf.fit(X_train, y_train, validation_split = val_portion, shuffle = False, epochs = n_epochs, batch_size = n_batch)

    def predict(self, X_test):
        return self.clf.predict(X_test)
