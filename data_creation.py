import os
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples, noise_level=0.1, anomaly_prob=0.1):
    # Генерируем временные метки
    timestamps = np.arange(n_samples)
    
    # Генерируем изменение температуры с некоторым шумом
    temperature = np.sin(timestamps * 2 * np.pi / 365) + np.random.normal(scale=noise_level, size=n_samples)
    
    # Добавляем аномалии
    anomalies = np.random.rand(n_samples) < anomaly_prob
    temperature[anomalies] += np.random.normal(scale=5, size=np.sum(anomalies))
    
    return timestamps, temperature

def save_data(timestamps, temperature, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as file:
        for timestamp, temp in zip(timestamps, temperature):
            file.write(f"{timestamp},{temp}\n")

def visualize_data(timestamps, temperature):
    plt.plot(timestamps, temperature)
    plt.xlabel('Day')
    plt.ylabel('Temperature')
    plt.title('Daily Temperature Variation')
    plt.show()

# Параметры данных
n_samples = 365
noise_level = 0.1
anomaly_prob = 0.1

# Создаем и сохраняем данные для тренировки
for i in range(5):
    timestamps, temperature = generate_data(n_samples, noise_level, anomaly_prob)
    save_data(timestamps, temperature, 'train', f'data_train_{i}.csv')
    visualize_data(timestamps, temperature)

# Создаем и сохраняем данные для тестирования
for i in range(5):
    timestamps, temperature = generate_data(n_samples, noise_level, anomaly_prob)
    save_data(timestamps, temperature, 'test', f'data_test_{i}.csv')
    visualize_data(timestamps, temperature)
