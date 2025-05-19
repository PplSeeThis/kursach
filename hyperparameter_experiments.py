#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# Импорт нашей модели и функций для обработки данных
from lstm_model import LSTMClassifier
from preprocessing import preprocess_text, tokenize_text, create_vocab, create_data_iterators

# Настройка стилей для графиков
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Функция для обучения LSTM модели с заданными гиперпараметрами
def train_lstm_with_params(train_data, val_data, params, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Обучает LSTM модель с заданными гиперпараметрами и возвращает результаты.
    
    Args:
        train_data: Тренировочные данные
        val_data: Валидационные данные
        params: Словарь с гиперпараметрами модели
        device: Устройство для обучения (cuda или cpu)
    
    Returns:
        dict: Словарь с результатами обучения (f1-score, accuracy, loss)
    """
    print(f"Training LSTM with params: {params}")
    
    # Создание словаря и итераторов данных
    TEXT, LABEL, train_iterator, val_iterator = create_data_iterators(
        train_data, val_data, batch_size=params['batch_size']
    )
    
    # Настройка модели
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = params['embedding_dim']
    HIDDEN_DIM = params['hidden_dim']
    OUTPUT_DIM = len(LABEL.vocab)
    N_LAYERS = params['n_layers']
    BIDIRECTIONAL = params['bidirectional']
    DROPOUT = params['dropout']
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    # Инициализация модели
    model = LSTMClassifier(
        INPUT_DIM, 
        EMBEDDING_DIM, 
        HIDDEN_DIM, 
        OUTPUT_DIM, 
        N_LAYERS, 
        BIDIRECTIONAL, 
        DROPOUT, 
        PAD_IDX
    )
    
    # Инициализация предтренированных embedding'ов, если они есть
    if params.get('use_pretrained', False):
        pretrained_embeddings = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    # Перемещение модели на устройство
    model = model.to(device)
    
    # Настройка оптимизатора и функции потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    
    # Обучение модели
    best_val_f1 = 0
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    for epoch in range(params['n_epochs']):
        # Обучение на эпоху
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_iterator, desc=f"Epoch {epoch+1}/{params['n_epochs']}", leave=False):
            # Получаем данные
            text, text_lengths = batch.text
            labels = batch.label
            
            # Перемещаем на устройство
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            labels = labels.to(device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            predictions = model(text, text_lengths).squeeze(1)
            
            # Вычисляем потери
            loss = criterion(predictions, labels)
            
            # Обратный проход
            loss.backward()
            
            # Оптимизация
            optimizer.step()
            
            # Накапливаем потери
            epoch_loss += loss.item()
            
            # Получаем предсказания
            preds = predictions.argmax(dim=1).cpu().numpy()
            labs = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
        
        # Вычисляем средние потери и f1-score для обучения
        train_loss = epoch_loss / len(train_iterator)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        
        # Валидация модели
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_iterator:
                # Получаем данные
                text, text_lengths = batch.text
                labels = batch.label
                
                # Перемещаем на устройство
                text = text.to(device)
                text_lengths = text_lengths.to(device)
                labels = labels.to(device)
                
                # Прямой проход
                predictions = model(text, text_lengths).squeeze(1)
                
                # Вычисляем потери
                loss = criterion(predictions, labels)
                
                # Накапливаем потери
                val_loss += loss.item()
                
                # Получаем предсказания
                preds = predictions.argmax(dim=1).cpu().numpy()
                labs = labels.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labs)
        
        # Вычисляем средние потери и f1-score для валидации
        val_loss = val_loss / len(val_iterator)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        # Сохраняем лучшую модель
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_lstm_model.pt')
        
        print(f"Epoch {epoch+1}/{params['n_epochs']}: Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Раннее останавливание
        if len(val_f1s) > params['patience'] and all(val_f1s[-params['patience']:] < best_val_f1):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load('best_lstm_model.pt'))
    
    # Тестируем модель на валидационном наборе
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_iterator:
            text, text_lengths = batch.text
            labels = batch.label
            
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            
            predictions = model(text, text_lengths).squeeze(1)
            preds = predictions.argmax(dim=1).cpu().numpy()
            labs = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
    
    # Вычисляем финальный f1-score
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Удаляем временный файл модели
    if os.path.exists('best_lstm_model.pt'):
        os.remove('best_lstm_model.pt')
    
    # Возвращаем результаты
    return {
        'f1_score': val_f1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'params': params
    }

# Эксперименты с разными размерностями скрытого состояния
def experiment_hidden_dim(train_data, val_data, base_params):
    """
    Эксперимент с разными размерностями скрытого состояния LSTM.
    
    Args:
        train_data: Тренировочные данные
        val_data: Валидационные данные
        base_params: Базовые гиперпараметры
        
    Returns:
        list: Список результатов экспериментов
    """
    results = []
    
    # Размерности для эксперимента
    hidden_dims = [128, 256, 512]
    
    for hidden_dim in hidden_dims:
        # Копируем базовые параметры и меняем размерность
        params = base_params.copy()
        params['hidden_dim'] = hidden_dim
        
        # Обучаем модель с новыми параметрами
        result = train_lstm_with_params(train_data, val_data, params)
        results.append(result)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    for result in results:
        hidden_dim = result['params']['hidden_dim']
        f1 = result['f1_score']
        plt.bar(str(hidden_dim), f1, label=f"Hidden dim: {hidden_dim}, F1: {f1:.4f}")
    
    plt.xlabel('Hidden Dimension')
    plt.ylabel('F1-score')
    plt.title('Влияние размерности скрытого состояния на F1-score')
    plt.ylim(0.7, 0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_hidden_dim.png', dpi=300)
    plt.close()
    
    return results

# Эксперименты с разным количеством слоев
def experiment_num_layers(train_data, val_data, base_params):
    """
    Эксперимент с разным количеством слоев LSTM.
    
    Args:
        train_data: Тренировочные данные
        val_data: Валидационные данные
        base_params: Базовые гиперпараметры
        
    Returns:
        list: Список результатов экспериментов
    """
    results = []
    
    # Количество слоев для эксперимента
    num_layers = [1, 2, 3]
    
    for n_layers in num_layers:
        # Копируем базовые параметры и меняем количество слоев
        params = base_params.copy()
        params['n_layers'] = n_layers
        
        # Обучаем модель с новыми параметрами
        result = train_lstm_with_params(train_data, val_data, params)
        results.append(result)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    for result in results:
        n_layers = result['params']['n_layers']
        f1 = result['f1_score']
        plt.bar(str(n_layers), f1, label=f"Layers: {n_layers}, F1: {f1:.4f}")
    
    plt.xlabel('Number of Layers')
    plt.ylabel('F1-score')
    plt.title('Влияние количества слоев на F1-score')
    plt.ylim(0.7, 0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_num_layers.png', dpi=300)
    plt.close()
    
    return results

# Эксперименты с однонаправленным и двунаправленным LSTM
def experiment_bidirectional(train_data, val_data, base_params):
    """
    Эксперимент с однонаправленным и двунаправленным LSTM.
    
    Args:
        train_data: Тренировочные данные
        val_data: Валидационные данные
        base_params: Базовые гиперпараметры
        
    Returns:
        list: Список результатов экспериментов
    """
    results = []
    
    # Направления для эксперимента
    bidirectional_options = [False, True]
    
    for bidirectional in bidirectional_options:
        # Копируем базовые параметры и меняем направление
        params = base_params.copy()
        params['bidirectional'] = bidirectional
        
        # Обучаем модель с новыми параметрами
        result = train_lstm_with_params(train_data, val_data, params)
        results.append(result)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    labels = ['LSTM (однонаправленный)', 'BiLSTM (двунаправленный)']
    f1_scores = [result['f1_score'] for result in results]
    
    plt.bar(labels, f1_scores)
    
    for i, f1 in enumerate(f1_scores):
        plt.text(i, f1 + 0.005, f"{f1:.4f}", ha='center')
    
    plt.xlabel('Тип модели')
    plt.ylabel('F1-score')
    plt.title('Сравнение однонаправленного и двунаправленного LSTM')
    plt.ylim(0.7, 0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_bidirectional.png', dpi=300)
    plt.close()
    
    return results

# Создание общего графика для сравнения всех архитектурных параметров
def create_architecture_comparison_plot(hidden_dim_results, num_layers_results, bidirectional_results):
    """
    Создает общий график для сравнения всех архитектурных параметров.
    
    Args:
        hidden_dim_results: Результаты экспериментов с размерностью скрытого состояния
        num_layers_results: Результаты экспериментов с количеством слоев
        bidirectional_results: Результаты экспериментов с направлением LSTM
        
    Returns:
        None
    """
    plt.figure(figsize=(14, 8))
    
    # Данные для графика
    categories = []
    f1_scores = []
    
    # Размерность скрытого состояния
    for result in hidden_dim_results:
        hidden_dim = result['params']['hidden_dim']
        categories.append(f"Hidden dim: {hidden_dim}")
        f1_scores.append(result['f1_score'])
    
    # Количество слоев
    for result in num_layers_results:
        n_layers = result['params']['n_layers']
        categories.append(f"Layers: {n_layers}")
        f1_scores.append(result['f1_score'])
    
    # Направление LSTM
    for i, result in enumerate(bidirectional_results):
        bidirectional = result['params']['bidirectional']
        label = 'BiLSTM' if bidirectional else 'LSTM'
        categories.append(label)
        f1_scores.append(result['f1_score'])
    
    # Создание графика
    bars = plt.bar(categories, f1_scores)
    
    # Добавление значений над столбцами
    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, f1 + 0.005, f"{f1:.4f}", ha='center')
    
    plt.xlabel('Параметры архитектуры')
    plt.ylabel('F1-score')
    plt.title('Влияние архитектурных параметров на производительность LSTM')
    plt.ylim(0.7, 0.8)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_architecture_comparison.png', dpi=300)
    plt.close()

# Эксперименты с гиперпараметрами обучения (learning rate)
def experiment_learning_rate(train_data, val_data, base_params):
    """
    Эксперимент с разными значениями learning rate.
    
    Args:
        train_data: Тренировочные данные
        val_data: Валидационные данные
        base_params: Базовые гиперпараметры
        
    Returns:
        list: Список результатов экспериментов
    """
    results = []
    
    # Значения learning rate для эксперимента
    learning_rates = [0.0001, 0.001, 0.01]
    
    for lr in learning_rates:
        # Копируем базовые параметры и меняем learning rate
        params = base_params.copy()
        params['learning_rate'] = lr
        
        # Обучаем модель с новыми параметрами
        result = train_lstm_with_params(train_data, val_data, params)
        results.append(result)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    for result in results:
        lr = result['params']['learning_rate']
        f1 = result['f1_score']
        plt.bar(str(lr), f1, label=f"LR: {lr}, F1: {f1:.4f}")
    
    plt.xlabel('Learning Rate')
    plt.ylabel('F1-score')
    plt.title('Влияние learning rate на F1-score')
    plt.ylim(0.7, 0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_learning_rate.png', dpi=300)
    plt.close()
    
    # Построение графиков обучения для разных learning rate
    plt.figure(figsize=(12, 8))
    
    for i, result in enumerate(results):
        lr = result['params']['learning_rate']
        train_losses = result['train_losses']
        val_losses = result['val_losses']
        
        plt.subplot(2, 2, i+1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Learning Rate: {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_learning_rate_curves.png', dpi=300)
    plt.close()
    
    return results

# Эксперименты с размером мини-батча
def experiment_batch_size(train_data, val_data, base_params):
    """
    Эксперимент с разными размерами мини-батча.
    
    Args:
        train_data: Тренировочные данные
        val_data: Валидационные данные
        base_params: Базовые гиперпараметры
        
    Returns:
        list: Список результатов экспериментов
    """
    results = []
    
    # Размеры мини-батча для эксперимента
    batch_sizes = [16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        # Копируем базовые параметры и меняем размер мини-батча
        params = base_params.copy()
        params['batch_size'] = batch_size
        
        # Обучаем модель с новыми параметрами
        result = train_lstm_with_params(train_data, val_data, params)
        results.append(result)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    for result in results:
        batch_size = result['params']['batch_size']
        f1 = result['f1_score']
        plt.bar(str(batch_size), f1, label=f"Batch Size: {batch_size}, F1: {f1:.4f}")
    
    plt.xlabel('Batch Size')
    plt.ylabel('F1-score')
    plt.title('Влияние размера мини-батча на F1-score')
    plt.ylim(0.7, 0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_batch_size.png', dpi=300)
    plt.close()
    
    return results

# Создание общего графика для сравнения всех гиперпараметров обучения
def create_learning_params_comparison_plot(learning_rate_results, batch_size_results):
    """
    Создает общий график для сравнения всех гиперпараметров обучения.
    
    Args:
        learning_rate_results: Результаты экспериментов с learning rate
        batch_size_results: Результаты экспериментов с размером мини-батча
        
    Returns:
        None
    """
    plt.figure(figsize=(14, 8))
    
    # Данные для графика
    categories = []
    f1_scores = []
    
    # Learning Rate
    for result in learning_rate_results:
        lr = result['params']['learning_rate']
        categories.append(f"LR: {lr}")
        f1_scores.append(result['f1_score'])
    
    # Batch Size
    for result in batch_size_results:
        batch_size = result['params']['batch_size']
        categories.append(f"Batch: {batch_size}")
        f1_scores.append(result['f1_score'])
    
    # Создание графика
    bars = plt.bar(categories, f1_scores)
    
    # Добавление значений над столбцами
    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, f1 + 0.005, f"{f1:.4f}", ha='center')
    
    plt.xlabel('Гиперпараметры обучения')
    plt.ylabel('F1-score')
    plt.title('Влияние гиперпараметров обучения на производительность LSTM')
    plt.ylim(0.7, 0.8)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_learning_params.png', dpi=300)
    plt.close()

# Основная функция для запуска всех экспериментов с LSTM
def run_lstm_experiments(train_data, val_data):
    """
    Запускает все эксперименты с LSTM моделью.
    
    Args:
        train_data: Тренировочные данные
        val_data: Валидационные данные
        
    Returns:
        dict: Словарь с результатами всех экспериментов
    """
    # Базовые гиперпараметры
    base_params = {
        'embedding_dim': 300,
        'hidden_dim': 256,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.3,
        'batch_size': 64,
        'learning_rate': 0.001,
        'n_epochs': 30,
        'patience': 5,
        'use_pretrained': True
    }
    
    # Запуск экспериментов
    print("Running experiments with hidden dimensions...")
    hidden_dim_results = experiment_hidden_dim(train_data, val_data, base_params)
    
    print("Running experiments with number of layers...")
    num_layers_results = experiment_num_layers(train_data, val_data, base_params)
    
    print("Running experiments with bidirectionality...")
    bidirectional_results = experiment_bidirectional(train_data, val_data, base_params)
    
    print("Creating architecture comparison plot...")
    create_architecture_comparison_plot(hidden_dim_results, num_layers_results, bidirectional_results)
    
    print("Running experiments with learning rate...")
    learning_rate_results = experiment_learning_rate(train_data, val_data, base_params)
    
    print("Running experiments with batch size...")
    batch_size_results = experiment_batch_size(train_data, val_data, base_params)
    
    print("Creating learning parameters comparison plot...")
    create_learning_params_comparison_plot(learning_rate_results, batch_size_results)
    
    # Возвращаем результаты всех экспериментов
    return {
        'hidden_dim_results': hidden_dim_results,
        'num_layers_results': num_layers_results,
        'bidirectional_results': bidirectional_results,
        'learning_rate_results': learning_rate_results,
        'batch_size_results': batch_size_results
    }

# Функция для сохранения результатов экспериментов в CSV файл
def save_results_to_csv(results, filename='lstm_experiment_results.csv'):
    """
    Сохраняет результаты экспериментов в CSV файл.
    
    Args:
        results: Словарь с результатами экспериментов
        filename: Имя файла для сохранения
        
    Returns:
        None
    """
    # Собираем все результаты
    all_results = []
    
    for exp_name, exp_results in results.items():
        for result in exp_results:
            params = result['params'].copy()
            params['experiment'] = exp_name
            params['f1_score'] = result['f1_score']
            all_results.append(params)
    
    # Создаем DataFrame и сохраняем в CSV
    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Пример использования (не запускается при импорте)
if __name__ == "__main__":
    # Загрузка данных
    from preprocessing import load_data
    
    train_data, val_data, test_data = load_data()
    
    # Запуск экспериментов
    results = run_lstm_experiments(train_data, val_data)
    
    # Сохранение результатов
    save_results_to_csv(results)
