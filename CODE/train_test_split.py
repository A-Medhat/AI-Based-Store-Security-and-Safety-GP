
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('merged2.csv')

# Assume the data is in sequential order
X = df.iloc[:, :-1].values  # features
y = df.iloc[:, -1].values   # labels

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape to sequences
# This changes X to shape (number_of_videos, frames_per_video, features_per_frame)
X_seq = X_scaled.reshape((-1, 145, X.shape[1]))
y_seq = y.reshape((-1, 145, 1))
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
