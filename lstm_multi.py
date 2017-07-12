trainx, trainy, testx, testy = reshape_data(train, test, 30)
print(trainx.shape)
import keras
print(keras.__version__)


"""
(3151, 30, 11)
2.0.2
"""

from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Lambda, AveragePooling2D
from keras.layers import Reshape, Dense,  Flatten, LSTM, Activation
model = Sequential()
model.add(LSTM(40,input_shape = trainx.shape[1:], return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(80,return_sequences=False))
model.add(Dropout(0.1))
#model.add(Dense(40))
model.add(Dense(20))
#model.add(LSTM(80,return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(output_dim=1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="rmsprop")

