import numpy as np
import pandas as pd
import pandas_datareader as pdr

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM

msft = pdr.DataReader('MSFT', 'iex', start = '2013-01-01', end = '2018-04-30')
msft = msft.rename(index = {i: pd.datetime(int(i[:4]), int(i[5:7]), int(i[8:10])) for i in msft.index.values.tolist()})

window = 50
data = []
for i in range(msft.shape[0] - window - 1):
    data.append(np.array(msft.iloc[i:i+window+1,3] / msft.iloc[i,3]))

train_size = int(0.9 * len(data))

train = np.array(data[:train_size])
test = np.array(data[train_size:])
np.random.seed(2018)
np.random.shuffle(train)
x_train = train[:,:-1]
y_train = train[:,-1]
x_test = test[:,:-1]
y_test = test[:,-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # dimension
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

def build_model(layers):
    model = Sequential()

    model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop")
    return model

model = build_model([1, 50, 100, 1])

model.fit(x_train, y_train, batch_size=512, epochs=10, validation_split=0.1, verbose=1)

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, np.sqrt(train_score)))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (test_score, np.sqrt(test_score)))

y_pred = model.predict(x_test)

import matplotlib.pyplot as plt

plt.plot(y_pred / np.mean(y_pred), color='red', label='prediction')
plt.plot(y_test / np.mean(y_test), color='blue', label='y_test')
plt.legend(loc='lower left')
plt.show()

def predict_sequences(model, data, win_size, pred_len):
    #Predict sequence of "pred_len" steps before shifting prediction run forward by "window_size" steps
    pred_seqs = []
    for i in range(int(len(data)/pred_len)):
        curr_frame = data[i*pred_len]
        predicted = []
        for j in range(pred_len):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [win_size-1], predicted[-1], axis=0)
        pred_seqs.append(predicted)
    return pred_seqs

pred_seqs = predict_sequences(model, x_test, 50, 20)

plt.plot(y_test, label='y_test')
for i, data in enumerate(pred_seqs):
    padding = [None for p in range(i * 20)]
    plt.plot(padding + data, label='Prediction')
plt.legend()
plt.show()