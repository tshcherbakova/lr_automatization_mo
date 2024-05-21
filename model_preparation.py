import os
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

def load_data(folder):
    data = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as file:
            samples = [line.strip().split(',') for line in file.readlines()]
            timestamps, temperature = zip(*[(int(sample[0]), float(sample[1])) for sample in samples])
            data.append((timestamps, temperature))
    return data

def prepare_data(data):
    X = np.array([timestamps for timestamps, _ in data])
    y = np.array([temperature for _, temperature in data])
    return X, y

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Загрузка данных из папки "train"
train_data = load_data('preprocessed_train')

# Подготовка данных для обучения
X_train, y_train = prepare_data(train_data)

# Обучение модели
model = train_model(X_train, y_train)

# Сохранение модели
joblib.dump(model, 'trained_model.pkl')
