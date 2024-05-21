import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(folder):
    data = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as file:
            samples = [line.strip().split(',') for line in file.readlines()]
            timestamps, temperature = zip(*[(int(sample[0]), float(sample[1])) for sample in samples])
            data.append((timestamps, temperature))
    return data

def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = []
    for timestamps, temperature in data:
        temperature_array = np.array(temperature).reshape(-1, 1)  # Преобразуем вектор температур в столбец
        scaled_temperature = scaler.fit_transform(temperature_array).flatten()
        scaled_data.append((timestamps, scaled_temperature))
    return scaled_data

def save_data(data, folder):
    if not os.path.exists(folder): 
        os.makedirs(folder)
    for i, (timestamps, temperature) in enumerate(data):
        filepath = os.path.join(folder, f'preprocessed_data_{i}.csv')
        with open(filepath, 'w') as file:
            for timestamp, temp in zip(timestamps, temperature):
                file.write(f"{timestamp},{temp}\n")

# Загрузка данных из папки "train" и "test"
train_data = load_data('train')
test_data = load_data('test')

# Предобработка данных
preprocessed_train_data = preprocess_data(train_data)
preprocessed_test_data = preprocess_data(test_data)

# Сохранение предобработанных данных
save_data(preprocessed_train_data, 'preprocessed_train')
save_data(preprocessed_test_data, 'preprocessed_test')
