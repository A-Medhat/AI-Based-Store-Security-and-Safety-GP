#
#
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense, TimeDistributed
from train_test_split import X_train,X_test,y_train,y_test
from tensorflow.python.client import device_lib
# Initialize the RNN to handle sequences
model = Sequential()


model.add(GRU(units=64, return_sequences=True, input_shape=(145, X_train.shape[2])))
model.add(Dropout(0.4))
model.add(GRU(units=16, return_sequences=True))
model.add(Dropout(0.4))
model.add(TimeDistributed(Dense(units=8, activation='relu')))
model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=16)
# Save the model
model.save('shoplifting_new_method0.h5')



