#!/bin/bash
# Запуск скрипта для создания данных
echo "Running data_creation.py..."
python data_creation.py
echo "Data created."

# Запуск скрипта для предобработки данных
echo "Running model_preprocessing.py..."
python model_preprocessing.py
echo "Preprocessing completed."

# Запуск скрипта для подготовки и обучения модели
echo "Running model_preparation.py..."
python model_preparation.py
echo "Model preparation and training completed."

# Запуск скрипта для тестирования модели
echo "Running model_testing.py..."
python model_testing.py
echo "Model testing completed."
