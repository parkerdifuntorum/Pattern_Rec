from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('nba_data3.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
print("print shape of values", dataset.values.shape)
print(dataset.values[0,0])
print(values[0:2])

pyplot.scatter(values[:,0],values[:,8], c='Red', label = "FG Attempted")
pyplot.scatter(values[:,0],values[:,2], c='Yellow', label = "Minutes Played")
pyplot.scatter(values[:,0],values[:,3], c='Green', label = "True Shooting %")
pyplot.scatter(values[:,0],values[:,4], c='Orange', label = "FT Attempted")
pyplot.scatter(values[:,0],values[:,5], c='Blue', label = "Offensive Rebounds")
pyplot.scatter(values[:,0],values[:,6], c='Purple', label = "Steals")
pyplot.scatter(values[:,0],values[:,7], c = "Pink", label = "Turnovers")
pyplot.legend()
pyplot.title("Raw Features")
pyplot.show()

pyplot.plot(values[:,8], c='Red', label = "FG Attempted")
pyplot.plot(values[:,2], c='Yellow', label = "Minutes Played")
pyplot.plot(values[:,3], c='Green', label = "True Shooting %")
pyplot.plot(values[:,4], c='Orange', label = "FT Attempted")
pyplot.plot(values[:,5], c='Blue', label = "Offensive Rebounds")
pyplot.plot(values[:,6], c='Purple', label = "Steals")
pyplot.plot(values[:,7], c = "Pink", label = "Turnovers")
pyplot.legend()
pyplot.title("Raw Features")
pyplot.xlabel("Games")
pyplot.show()


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#scaled =
pyplot.plot(scaled[:,8], c='Red', label = "FG Attempted")
pyplot.plot(scaled[:,2], c='Yellow', label = "Minutes Played")
pyplot.plot(scaled[:,3], c='Green', label = "True Shooting %")
pyplot.plot(scaled[:,4], c='Orange', label = "FT Attempted")
pyplot.plot(scaled[:,5], c='Blue', label = "Offensive Rebounds")
pyplot.plot(scaled[:,6], c='Purple', label = "Steals")
pyplot.plot(scaled[:,7], c = "Pink", label = "Turnovers")
pyplot.legend()
pyplot.title("Scaled Features")
pyplot.show()

pyplot.scatter(scaled[:,0],scaled[:,8], c='Red', label = "FG Attempted")
pyplot.scatter(scaled[:,0],scaled[:,2], c='Yellow', label = "Minutes Played")
pyplot.scatter(scaled[:,0],scaled[:,3], c='Green', label = "True Shooting %")
pyplot.scatter(scaled[:,0],scaled[:,4], c='Orange', label = "FT Attempted")
pyplot.scatter(scaled[:,0],scaled[:,5], c='Blue', label = "Offensive Rebounds")
pyplot.scatter(scaled[:,0],scaled[:,6], c='Purple', label = "Steals")
pyplot.scatter(scaled[:,0],scaled[:,7], c = "Pink", label = "Turnovers")
pyplot.legend()
pyplot.title("Scaled Features")
pyplot.show()

pyplot.plot(scaled)
pyplot.show()




# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[11, 12, 13, 14, 15, 16, 17, 18, 19]], axis=1, inplace=True)
print(reframed.head())


# split into train and test sets
values = reframed.values
n_train_hours = 50
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(train)
train = pca.transform(train)
print(pca.explained_variance_)

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
pyplot.plot(train_y)
pyplot.plot(test_y)
pyplot.title("Training and Testing data")
pyplot.show()



#print('first testx', test_X)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['acc'])
# fit network
history = model.fit(train_X, train_y, epochs=150, batch_size=15, validation_data=(test_X, test_y), verbose=0,
                    shuffle=False)

# plot accuracy
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('Model accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Test'], loc='upper left')
pyplot.show()

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print(inv_y)
print(inv_yhat)


pyplot.plot(inv_y, color='orange', label='actual')
pyplot.plot(inv_yhat, color='blue', label='predicted')
pyplot.legend(['Actual', 'Predicted'])
pyplot.title('The Greek Freak - Points Per Game')
pyplot.xlabel('Games')
pyplot.ylabel('Points')
pyplot.show()

