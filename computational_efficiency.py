#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import psutil
from tqdm import tqdm

# Настройка стилей для графиков
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def measure_inference_time(model, data_loader, device, model_name, num_runs=100):
    """
    Измеряет время инференса модели.
    
    Args:
        model: Модель для оценки
        data_loader: Загрузчик данных для инференса
        device: Устройство для инференса
        model_name: Название модели
        num_runs: Количество запусков для измерения
        
    Returns:
        dict: Словарь с результатами измерений
    """
    print(f"Измерение времени инференса для модели {model_name}...")
    
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
    
    # Измерение инференса
    inference_times = []
    
    print(f"Выполнение {num_runs} запусков инференса...")
    for _ in tqdm(range(num_runs)):
        # Измерение времени инференса
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            if model_name == 'LSTM':
                _ = model(text, text_lengths)
            else:  # BERT-lite
                _ = model(input_ids, attention_mask=attention_mask)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        inference_times.append(end_time - start_time)
    
    # Расчет статистики
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    median_time = np.median(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    # Конвертация в миллисекунды
    mean_time_ms = mean_time * 1000
    std_time_ms = std_time * 1000
    median_time_ms = median_time * 1000
    min_time_ms = min_time * 1000
    max_time_ms = max_time * 1000
    
    # Результаты
    results = {
        'model_name': model_name,
        'mean_time': mean_time,
        'std_time': std_time,
        'median_time': median_time,
        'min_time': min_time,
        'max_time': max_time,
        'mean_time_ms': mean_time_ms,
        'std_time_ms': std_time_ms,
        'median_time_ms': median_time_ms,
        'min_time_ms': min_time_ms,
        'max_time_ms': max_time_ms
    }
    
    print(f"Результаты для {model_name}:")
    print(f"  Среднее время инференса: {mean_time_ms:.2f} мс")
    print(f"  Стандартное отклонение: {std_time_ms:.2f} мс")
    print(f"  Медианное время: {median_time_ms:.2f} мс")
    print(f"  Минимальное время: {min_time_ms:.2f} мс")
    print(f"  Максимальное время: {max_time_ms:.2f} мс")
    
    return results

def measure_memory_usage(model, model_name):
    """
    Измеряет использование памяти моделью.
    
    Args:
        model: Модель для оценки
        model_name: Название модели
        
    Returns:
        dict: Словарь с результатами измерений
    """
    print(f"Измерение использования памяти для модели {model_name}...")
    
    # Получение размера модели
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    # Определение количества параметров
    num_params = sum(p.numel() for p in model.parameters())
    
    # Результаты
    results = {
        'model_name': model_name,
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb,
        'num_params': num_params,
        'num_params_millions': num_params / 1e6
    }
    
    print(f"Результаты для {model_name}:")
    print(f"  Размер параметров: {results['param_size_mb']:.2f} MB")
    print(f"  Размер буферов: {results['buffer_size_mb']:.2f} MB")
    print(f"  Общий размер: {size_all_mb:.2f} MB")
    print(f"  Количество параметров: {num_params:,}")
    print(f"  Количество параметров (млн): {results['num_params_millions']:.2f}")
    
    return results

def estimate_flops(model, input_shape, model_name):
    """
    Оценивает количество FLOPS (операций с плавающей точкой) для модели.
    
    Args:
        model: Модель для оценки
        input_shape: Форма входных данных
        model_name: Название модели
        
    Returns:
        dict: Словарь с результатами оценки
    """
    print(f"Оценка FLOPS для модели {model_name}...")
    
    # Приблизительные оценки FLOPS для LSTM и BERT-lite
    # Эти значения основаны на аналитических формулах и должны быть уточнены
    # в зависимости от конкретной архитектуры модели
    
    if model_name == 'LSTM':
        # Для LSTM с BiLSTM и 2 слоями
        hidden_size = 256  # из параметров модели
        seq_length = input_shape[0]
        batch_size = input_shape[1]
        embedding_dim = 300  # из параметров модели
        
        # FLOPS для одного шага LSTM
        flops_per_lstm_step = 4 * (hidden_size * (hidden_size + embedding_dim + 1) + hidden_size)
        
        # FLOPS для BiLSTM с 2 слоями
        total_flops = batch_size * seq_length * flops_per_lstm_step * 2 * 2  # bidirectional * layers
        
    else:  # BERT-lite
        # Для BERT-lite с 6 слоями и 8 головками внимания
        hidden_size = 512  # из параметров модели
        seq_length = input_shape[0]
        batch_size = input_shape[1]
        num_layers = 6
        num_attention_heads = 8
        intermediate_size = 2048
        
        # FLOPS для одной головки внимания
        head_dim = hidden_size // num_attention_heads
        flops_per_attention_head = 2 * seq_length * head_dim * (2 * seq_length + head_dim)
        
        # FLOPS для механизма внимания в одном слое
        flops_attention = num_attention_heads * flops_per_attention_head
        
        # FLOPS для FFN в одном слое
        flops_ffn = 2 * seq_length * hidden_size * intermediate_size
        
        # FLOPS для одного слоя Transformer
        flops_per_layer = flops_attention + flops_ffn
        
        # Общие FLOPS
        total_flops = batch_size * num_layers * flops_per_layer
    
    # Конвертация в GFLOPS
    total_gflops = total_flops / 1e9
    
    # Результаты
    results = {
        'model_name': model_name,
        'total_flops': total_flops,
        'total_gflops': total_gflops
    }
    
    print(f"Результаты для {model_name}:")
    print(f"  Общее количество FLOPS: {total_flops:,}")
    print(f"  Общее количество GFLOPS: {total_gflops:.2f}")
    
    return results

def visualize_computational_efficiency(lstm_results, bert_results):
    """
    Визуализирует сравнение вычислительной эффективности моделей.
    
    Args:
        lstm_results: Результаты измерений для LSTM
        bert_results: Результаты измерений для BERT-lite
        
    Returns:
        None
    """
    # Создаем директорию для сохранения графиков
    os.makedirs('plots', exist_ok=True)
    
    # Время инференса
    plt.figure(figsize=(10, 6))
    models = ['LSTM', 'BERT-lite']
    inference_times = [lstm_results['inference']['mean_time_ms'], bert_results['inference']['mean_time_ms']]
    
    plt.bar(models, inference_times)
    plt.ylabel('Время инференса (мс)')
    plt.title('Сравнение времени инференса')
    
    # Добавление значений над столбцами
    for i, time in enumerate(inference_times):
        plt.text(i, time + 0.1, f'{time:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/inference_time_comparison.png', dpi=300)
    plt.close()
    
    # Использование памяти
    plt.figure(figsize=(10, 6))
    memory_usages = [lstm_results['memory']['total_size_mb'], bert_results['memory']['total_size_mb']]
    
    plt.bar(models, memory_usages)
    plt.ylabel('Использование памяти (MB)')
    plt.title('Сравнение использования памяти')
    
    # Добавление значений над столбцами
    for i, memory in enumerate(memory_usages):
        plt.text(i, memory + 0.1, f'{memory:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/memory_usage_comparison.png', dpi=300)
    plt.close()
    
    # Количество параметров
    plt.figure(figsize=(10, 6))
    num_params = [lstm_results['memory']['num_params_millions'], bert_results['memory']['num_params_millions']]
    
    plt.bar(models, num_params)
    plt.ylabel('Количество параметров (млн)')
    plt.title('Сравнение количества параметров')
    
    # Добавление значений над столбцами
    for i, params in enumerate(num_params):
        plt.text(i, params + 0.1, f'{params:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/params_comparison.png', dpi=300)
    plt.close()
    
    # FLOPS
    plt.figure(figsize=(10, 6))
    gflops = [lstm_results['flops']['total_gflops'], bert_results['flops']['total_gflops']]
    
    plt.bar(models, gflops)
    plt.ylabel('GFLOPS')
    plt.title('Сравнение вычислительной сложности (GFLOPS)')
    
    # Добавление значений над столбцами
    for i, flops in enumerate(gflops):
        plt.text(i, flops + 0.1, f'{flops:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/flops_comparison.png', dpi=300)
    plt.close()
    
    # Обобщенное сравнение
    # Создаем DataFrame для результатов
    df = pd.DataFrame({
        'Модель': ['LSTM', 'BERT-lite'],
        'Время инференса (мс)': inference_times,
        'Использование памяти (MB)': memory_usages,
        'Количество параметров (млн)': num_params,
        'GFLOPS': gflops
    })
    
    # Сохраняем в CSV
    df.to_csv('results/computational_efficiency_results.csv', index=False)
    
    # Создаем гистограмму с отношением BERT-lite к LSTM
    ratios = {
        'Время инференса': bert_results['inference']['mean_time_ms'] / lstm_results['inference']['mean_time_ms'],
        'Использование памяти': bert_results['memory']['total_size_mb'] / lstm_results['memory']['total_size_mb'],
        'Количество параметров': bert_results['memory']['num_params_millions'] / lstm_results['memory']['num_params_millions'],
        'GFLOPS': bert_results['flops']['total_gflops'] / lstm_results['flops']['total_gflops']
    }
    
    plt.figure(figsize=(12, 6))
    
    metrics = list(ratios.keys())
    ratio_values = list(ratios.values())
    
    plt.bar(metrics, ratio_values)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    plt.ylabel('Отношение BERT-lite к LSTM')
    plt.title('Относительная вычислительная эффективность BERT-lite по сравнению с LSTM')
    
    # Добавление значений над столбцами
    for i, ratio in enumerate(ratio_values):
        plt.text(i, ratio + 0.1, f'{ratio:.2f}x', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/efficiency_ratio_comparison.png', dpi=300)
    plt.close()
    
    print("Графики сравнения вычислительной эффективности сохранены в директории 'plots/'")
    print("Результаты сохранены в файле 'results/computational_efficiency_results.csv'")

def run_computational_efficiency_analysis(lstm_model_path, bert_model_path):
    """
    Запускает анализ вычислительной эффективности для LSTM и BERT-lite моделей.
    
    Args:
        lstm_model_path: Путь к сохраненной LSTM модели
        bert_model_path: Путь к сохраненной BERT-lite модели
        
    Returns:
        dict: Словарь с результатами анализа
    """
    # Создаем директории для результатов
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Устройство для запуска
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загрузка данных и моделей
    from preprocessing import load_data, create_data_iterators
    from bert_data_preparation import create_bert_dataloaders
    from lstm_model import LSTMClassifier
    from bert_lite_model import BERTLiteClassifier
    
    # Загружаем тестовые данные
    _, _, test_data = load_data()
    
    # Создаем итераторы данных для LSTM
    TEXT, LABEL, test_iterator = create_data_iterators(None, None, test_data, test_only=True)
    
    # Создаем загрузчики данных для BERT-lite
    _, _, test_dataloader, label_encoder = create_bert_dataloaders(None, None, test_data, test_only=True)
    
    # Параметры моделей
    # LSTM
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = len(LABEL.vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    # BERT-lite
    from transformers import BertConfig
    
    config = BertConfig(
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        num_labels=len(label_encoder.classes_)
    )
    
    # Инициализация моделей
    lstm_model = LSTMClassifier(
        INPUT_DIM, 
        EMBEDDING_DIM, 
        HIDDEN_DIM, 
        OUTPUT_DIM, 
        N_LAYERS, 
        BIDIRECTIONAL, 
        DROPOUT, 
        PAD_IDX
    )
    
    bert_model = BERTLiteClassifier(config)
    
    # Загрузка весов
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=device))
    
    # Перемещение моделей на устройство
    lstm_model = lstm_model.to(device)
    bert_model = bert_model.to(device)
    
    # Измерение производительности
    # Время инференса
    lstm_inference_results = measure_inference_time(lstm_model, test_iterator, device, 'LSTM')
    bert_inference_results = measure_inference_time(bert_model, test_dataloader, device, 'BERT-lite')
    
    # Использование памяти
    lstm_memory_results = measure_memory_usage(lstm_model, 'LSTM')
    bert_memory_results = measure_memory_usage(bert_model, 'BERT-lite')
    
    # Оценка FLOPS
    # Получаем размер входных данных
    for batch in test_iterator:
        text, text_lengths = batch.text
        lstm_input_shape = text.shape
        break
    
    for batch in test_dataloader:
        input_ids = batch['input_ids']
        bert_input_shape = input_ids.shape
        break
    
    lstm_flops_results = estimate_flops(lstm_model, lstm_input_shape, 'LSTM')
    bert_flops_results = estimate_flops(bert_model, bert_input_shape, 'BERT-lite')
    
    # Организация результатов
    lstm_results = {
        'inference': lstm_inference_results,
        'memory': lstm_memory_results,
        'flops': lstm_flops_results
    }
    
    bert_results = {
        'inference': bert_inference_results,
        'memory': bert_memory_results,
        'flops': bert_flops_results
    }
    
    # Визуализация сравнения
    visualize_computational_efficiency(lstm_results, bert_results)
    
    return {
        'lstm': lstm_results,
        'bert': bert_results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze computational efficiency of LSTM and BERT-lite models.')
    parser.add_argument('--lstm_model', type=str, required=True, help='Path to LSTM model')
    parser.add_argument('--bert_model', type=str, required=True, help='Path to BERT-lite model')
    
    args = parser.parse_args()
    
    results = run_computational_efficiency_analysis(args.lstm_model, args.bert_model)
