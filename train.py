import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import numpy as np
import pandas as pd
import os
import sys

dataset = pd.read_csv('./lettuce_dataset_updated.csv', encoding="ISO-8859-1")

# Przygotowanie danych

ph_level = dataset['pH Level'].values.tolist()
temp_F = dataset['Temperature (F)'].values.tolist()
humid = dataset['Humidity'].values.tolist()
days = dataset['Growth Days'].values.tolist()
plant_id = dataset['Plant_ID'].values.tolist()

X = []
Y = []

id = plant_id[0]
temp_sum = 0
humid_sum = 0
ph_level_sum = 0
day = 1

for i in range(0, len(plant_id)):
    if plant_id[i] == id:
        temp_sum += temp_F[i]
        humid_sum += humid[i]
        ph_level_sum += ph_level[i]
        day = days[i]
    else:
        temp = []
        temp.append(temp_sum/day)
        temp.append(humid_sum/day)
        temp.append(ph_level_sum/day)
        X.append(temp)
        Y.append(day)
        temp_sum = 0
        humid_sum = 0
        ph_level_sum = 0
        day = 1
        id = plant_id[i]

scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(X))

encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(np.array(Y).reshape(-1,1))

y_onehot_dense = y_onehot.todense()
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot_dense, test_size=0.4, random_state=42)

print(y_train.shape[0], y_test.shape[0], X_train.shape[0], X_test.shape[0])

num_epochs = 500
dropout_layer_value = 0.5

if len(sys.argv) == 3:
    num_epochs = int(sys.argv[1])
    dropout_layer_value = float(sys.argv[2])

model = Sequential([
    Dense(8, activation='relu', input_dim=3, kernel_regularizer=regularizers.l2(0.04)),
    Dropout(dropout_layer_value),
    Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.04)),
    Dropout(dropout_layer_value),
    Dense(4, activation='softmax', kernel_regularizer=regularizers.l2(0.04)),
])

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Dokładność testowa: {test_accuracy:.2%}")

model.evaluate(X_test, y_test)[1]

model.save('./model.keras') 
