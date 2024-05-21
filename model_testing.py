import os
import numpy as np
from sklearn.metrics import mean_squared_error
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

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Загрузка данных из папки "test"
test_data = load_data('preprocessed_test')

# Подготовка данных для тестирования
X_test, y_test = prepare_data(test_data)

# Загрузка обученной модели
model = load_model('trained_model.pkl')

# Оценка модели
mse = evaluate_model(model, X_test, y_test)
print("Mean Squared Error on Test Data:", mse)
