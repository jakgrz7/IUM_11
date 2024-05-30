import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split

model = keras.models.load_model('./model.keras')

dataset = pd.read_csv('./lettuce_dataset_updated.csv', encoding='ISO-8859-1')

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

X = scaler.fit_transform(X)
X = np.array(X)
Y = np.array(Y)

encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(Y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.4, random_state=42)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Dokładność testowa: {test_accuracy:.2%}")