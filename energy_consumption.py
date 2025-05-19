#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import psutil
import os
from tqdm import tqdm

# Настройка стилей для графиков
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def measure_energy_consumption(model, data_loader, device, model_name, iterations=100):
    """
    Измеряет энергопотребление модели во время инференса.
    
    Args:
        model: Модель для оценки
        data_loader: Загрузчик данных для инференса
        device: Устройство для инференса
        model_name: Название модели
        iterations: Количество итераций для измерения
        
    Returns:
        dict: Словарь с результатами измерений
    """
    print(f"Измерение энергопотребления для модели {model_name}...")
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Получаем первый батч для разогрева
    for batch in data_loader:
        if model_name == 'LSTM':
            text, text_lengths = batch.text
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            break
        else:  # BERT-lite
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            break
    
    # Разогрев модели
    print("Разогрев модели...")
    with torch.no_grad():
        for _ in range(5):
            if model_name == 'LSTM':
                _ = model(text, text_lengths)
            else:  # BERT-lite
                _ = model(input_ids, attention_mask=attention_mask)
    
    # Измерение потребления CPU
    process = psutil.Process(os.getpid())
    
    # Подготовка массивов для хранения результатов
    inference_times = []
    cpu_usages = []
    memory_usages = []
    
    # Измерение инференса
    print(f"Выполнение {iterations} итераций инференса...")
    for _ in tqdm(range(iterations)):
        # Замер CPU и памяти
        cpu_usage_start = process.cpu_percent(interval=0.1)
        memory_info_start = process.memory_info()
        
        # Замер времени
        start_time = time.time()
        
        # Инференс
        with torch.no_grad():
            if model_name == 'LSTM':
                _ = model(text, text_lengths)
            else:  # BERT-lite
                _ = model(input_ids, attention_mask=attention_mask)
        
        # Расчет времени
        inference_time = time.time() - start_time
        
        # Замер CPU и памяти после инференса
        cpu_usage_end = process.cpu_percent(interval=0.1)
        memory_info_end = process.memory_info()
        
        # Сохранение результатов
        inference_times.append(inference_time)
        cpu_usages.append((cpu_usage_start + cpu_usage_end) / 2)  # Среднее значение
        memory_usages.append((memory_info_end.rss - memory_info_start.rss) / 1024 / 1024)  # MB
    
    # Расчет средних значений
    avg_inference_time = np.mean(inference_times)
    avg_cpu_usage = np.mean(cpu_usages)
    avg_memory_usage = np.mean(memory_usages)
    
    # Оценка энергопотребления (аппроксимация)
    # Предполагаем, что CPU потребляет ~100W при 100% загрузке
    # и GPU потребляет ~250W при полной загрузке (для NVIDIA RTX)
    if device.type == 'cuda':
        energy_consumption = avg_inference_time * (avg_cpu_usage / 100 * 100 + 250) / 3600  # кВт*ч на инференс
    else:
        energy_consumption = avg_inference_time * (avg_cpu_usage / 100 * 100) / 3600  # кВт*ч на инференс
    
    # Экологический след - приблизительно 0.48 кг CO2 на 1 кВт*ч
    carbon_footprint = energy_consumption * 0.48  # кг CO2
    
    # Результаты
    results = {
        'model_name': model_name,
        'inference_time': avg_inference_time,
        'cpu_usage': avg_cpu_usage,
        'memory_usage': avg_memory_usage,
        'energy_consumption': energy_consumption,
        'carbon_footprint': carbon_footprint
    }
    
    print(f"Результаты для {model_name}:")
    print(f"  Среднее время инференса: {avg_inference_time:.6f} с")
    print(f"  Среднее использование CPU: {avg_cpu_usage:.2f}%")
    print(f"  Среднее использование памяти: {avg_memory_usage:.2f} MB")
    print(f"  Энергопотребление: {energy_consumption * 1e6:.2f} мкВт*ч на инференс")
    print(f"  Экологический след: {carbon_footprint * 1e6:.2f} мкг CO2 на инференс")
    
    return results

def measure_training_energy(model_name, train_function, train_data, val_data, test_data, params):
    """
    Измеряет энергопотребление во время обучения модели.
    
    Args:
        model_name: Название модели
        train_function: Функция для обучения модели
        train_data: Тренировочные данные
        val_data: Валидационные данные
        test_data: Тестовые данные
        params: Параметры обучения
        
    Returns:
        dict: Словарь с результатами измерений
    """
    print(f"Измерение энергопотребления при обучении модели {model_name}...")
    
    # Подготовка для измерения энергопотребления
    process = psutil.Process(os.getpid())
    
    # Замеры перед обучением
    cpu_usage_start = process.cpu_percent(interval=1.0)
    memory_info_start = process.memory_info()
    start_time = time.time()
    
    # Обучение модели
    model_path = train_function(train_data, val_data, test_data, params)
    
    # Замеры после обучения
    end_time = time.time()
    cpu_usage_end = process.cpu_percent(interval=1.0)
    memory_info_end = process.memory_info()
    
    # Расчет результатов
    training_time = end_time - start_time  # в секундах
    avg_cpu_usage = (cpu_usage_start + cpu_usage_end) / 2  # среднее значение
    memory_usage = (memory_info_end.rss - memory_info_start.rss) / 1024 / 1024  # MB
    
    # Оценка энергопотребления
    # Предполагаем, что система потребляет ~200W при обучении на CPU
    # и ~350W при обучении на GPU
    if torch.cuda.is_available():
        energy_consumption = training_time * 350 / 3600  # кВт*ч
    else:
        energy_consumption = training_time * 200 / 3600  # кВт*ч
    
    # Экологический след
    carbon_footprint = energy_consumption * 0.48  # кг CO2
    
    # Результаты
    results = {
        'model_name': model_name,
        'training_time': training_time,
        'training_time_hours': training_time / 3600,
        'avg_cpu_usage': avg_cpu_usage,
        'memory_usage': memory_usage,
        'energy_consumption': energy_consumption,
        'carbon_footprint': carbon_footprint,
        'model_path': model_path
    }
    
    print(f"Результаты обучения для {model_name}:")
    print(f"  Время обучения: {training_time / 3600:.2f} часов")
    print(f"  Среднее использование CPU: {avg_cpu_usage:.2f}%")
    print(f"  Использование памяти: {memory_usage:.2f} MB")
    print(f"  Энергопотребление: {energy_consumption:.2f} кВт*ч")
    print(f"  Экологический след: {carbon_footprint:.2f} кг CO2")
    
    return results

def visualize_energy_comparison(lstm_results, bert_results):
    """
    Визуализирует сравнение энергопотребления моделей.
    
    Args:
        lstm_results: Результаты измерений для LSTM
        bert_results: Результаты измерений для BERT-lite
        
    Returns:
        None
    """
    # Создаем DataFrame для результатов
    df = pd.DataFrame({
        'Модель': ['LSTM', 'BERT-lite'],
        'Время обучения (часы)': [lstm_results['training_time_hours'], bert_results['training_time_hours']],
        'Энергопотребление (кВт*ч)': [lstm_results['energy_consumption'], bert_results['energy_consumption']],
        'Экологический след (кг CO2)': [lstm_results['carbon_footprint'], bert_results['carbon_footprint']]
    })
    
    # Сохраняем результаты в CSV
    df.to_csv('results/energy_consumption_results.csv', index=False)
    
    # Визуализация времени обучения
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    ax = sns.barplot(x='Модель', y='Время обучения (часы)', data=df)
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    plt.title('Время обучения')
    
    plt.subplot(1, 3, 2)
    ax = sns.barplot(x='Модель', y='Энергопотребление (кВт*ч)', data=df)
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    plt.title('Энергопотребление')
    
    plt.subplot(1, 3, 3)
    ax = sns.barplot(x='Модель', y='Экологический след (кг CO2)', data=df)
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    plt.title('Экологический след')
    
    plt.tight_layout()
    plt.savefig('plots/energy_consumption_comparison.png', dpi=300)
    plt.close()
    
    print("Графики сравнения энергопотребления сохранены в plots/energy_consumption_comparison.png")

def run_energy_analysis():
    """
    Запускает анализ энергопотребления для LSTM и BERT-lite моделей.
    
    Returns:
        dict: Словарь с результатами анализа
    """
    from preprocessing import load_data
    
    print("Запуск анализа энергопотребления...")
    
    # Создаем директории для результатов
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Загрузка данных
    train_data, val_data, test_data = load_data()
    
    # Параметры для LSTM
    lstm_params = {
        'embedding_dim': 300,
        'hidden_dim': 256,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.3,
        'batch_size': 64,
        'learning_rate': 0.001,
        'n_epochs': 10,  # Уменьшаем для ускорения анализа
        'patience': 3,
        'use_pretrained': True
    }
    
    # Параметры для BERT-lite
    bert_params = {
        'hidden_size': 512,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 2048,
        'dropout': 0.1,
        'max_length': 128,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'n_epochs': 5,  # Уменьшаем для ускорения анализа
        'patience': 2,
        'use_pretrained': True
    }
    
    # Импортируем функции обучения
    from main import train_lstm_model, train_bert_model
    
    # Измеряем энергопотребление при обучении
    lstm_training_results = measure_training_energy('LSTM', train_lstm_model, train_data, val_data, test_data, lstm_params)
    bert_training_results = measure_training_energy('BERT-lite', train_bert_model, train_data, val_data, test_data, bert_params)
    
    # Загружаем обученные модели для измерения инференса
    from lstm_model import LSTMClassifier
    from bert_lite_model import BERTLiteClassifier
    from preprocessing import create_data_iterators
    from bert_data_preparation import create_bert_dataloaders
    
    # Создаем загрузчики данных
    _, _, test_iterator = create_data_iterators(None, None, test_data, test_only=True)
    _, _, test_dataloader, _ = create_bert_dataloaders(None, None, test_data, test_only=True)
    
    # Загружаем LSTM модель
    # (здесь должен быть код для загрузки конкретной LSTM модели, аналогично тому, что в evaluate.py)
    
    # Загружаем BERT-lite модель
    # (здесь должен быть код для загрузки конкретной BERT-lite модели)
    
    # Визуализируем сравнение
    visualize_energy_comparison(lstm_training_results, bert_training_results)
    
    return {
        'lstm_training': lstm_training_results,
        'bert_training': bert_training_results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze energy consumption of LSTM and BERT-lite models.')
    parser.add_argument('--train', action='store_true', help='Run training energy analysis')
    parser.add_argument('--inference', action='store_true', help='Run inference energy analysis')
    
    args = parser.parse_args()
    
    if args.train or args.inference or not (args.train or args.inference):
        # Если не указано ни одного аргумента, запускаем все
        results = run_energy_analysis()
        
        print("\nАнализ энергопотребления завершен.")
        print("Результаты сохранены в files/energy_consumption_results.csv")
