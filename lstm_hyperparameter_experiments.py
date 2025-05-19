"""
lstm_hyperparameter_experiments.py
Модуль для експериментів з гіперпараметрами LSTM моделі
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import time
import argparse
from sklearn.metrics import f1_score

# Імпортуємо наші модулі
from preprocessing import load_and_preprocess_data
from lstm_model import create_dataloaders_for_lstm, initialize_lstm_model, train_lstm_model, evaluate_lstm_model

def run_lstm_experiments_hidden_dim(train_df, val_df, test_df, hidden_dims=[128, 256, 512], device='cuda', results_dir='results/lstm'):
    """
    Експерименти з розмірністю прихованого стану LSTM
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        hidden_dims: Список розмірностей прихованого стану
        device: Пристрій для обчислень
        results_dir: Директорія для збереження результатів
    
    Returns:
        DataFrame з результатами експериментів
    """
    # Створення директорії для результатів
    os.makedirs(results_dir, exist_ok=True)
    
    # Підготовка даних для LSTM
    train_dataloader, val_dataloader, test_dataloader, \
    word2idx, idx2word, label2idx, idx2label, vocab_size = create_dataloaders_for_lstm(
        train_df, val_df, test_df
    )
    
    # Список для збереження результатів
    results = []
    
    # Експерименти з різними розмірностями прихованого стану
    for hidden_dim in hidden_dims:
        print(f"\nЕксперимент з hidden_dim={hidden_dim}")
        
        # Створення моделі
        model = initialize_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=hidden_dim,
            output_dim=len(label2idx),
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        # Налаштування оптимізатора та функції втрат
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Навчання моделі
        start_time = time.time()
        model, history = train_lstm_model(
            model, train_dataloader, val_dataloader,
            optimizer, criterion, 20, 5, device
        )
        training_time = time.time() - start_time
        
        # Оцінка моделі на тестовій вибірці
        test_loss, test_f1 = evaluate_lstm_model(model, test_dataloader, criterion, device)
        
        # Збереження результатів
        result = {
            'hidden_dim': hidden_dim,
            'val_f1': history['best_val_f1'],
            'test_f1': test_f1,
            'test_loss': test_loss,
            'training_time': training_time,
            'epochs': len(history['train_losses'])
        }
        results.append(result)
        
        # Збереження історії навчання
        with open(os.path.join(results_dir, f'lstm_hidden_dim_{hidden_dim}_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Результати: val_f1={history['best_val_f1']:.4f}, test_f1={test_f1:.4f}, час навчання={training_time:.2f}с")
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Збереження результатів у CSV
    results_df.to_csv(os.path.join(results_dir, 'lstm_hidden_dim_results.csv'), index=False)
    
    # Візуалізація результатів
    plt.figure(figsize=(10, 5))
    plt.plot(hidden_dims, results_df['test_f1'], 'o-', label='F1-міра')
    plt.xlabel('Розмірність прихованого стану')
    plt.ylabel('F1-міра')
    plt.title('Вплив розмірності прихованого стану на продуктивність LSTM')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'lstm_hidden_dim_results.png'))
    
    return results_df

def run_lstm_experiments_n_layers(train_df, val_df, test_df, n_layers_list=[1, 2, 3], device='cuda', results_dir='results/lstm'):
    """
    Експерименти з кількістю шарів LSTM
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        n_layers_list: Список кількостей шарів
        device: Пристрій для обчислень
        results_dir: Директорія для збереження результатів
    
    Returns:
        DataFrame з результатами експериментів
    """
    # Створення директорії для результатів
    os.makedirs(results_dir, exist_ok=True)
    
    # Підготовка даних для LSTM
    train_dataloader, val_dataloader, test_dataloader, \
    word2idx, idx2word, label2idx, idx2label, vocab_size = create_dataloaders_for_lstm(
        train_df, val_df, test_df
    )
    
    # Список для збереження результатів
    results = []
    
    # Експерименти з різною кількістю шарів
    for n_layers in n_layers_list:
        print(f"\nЕксперимент з n_layers={n_layers}")
        
        # Створення моделі
        model = initialize_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            output_dim=len(label2idx),
            n_layers=n_layers,
            bidirectional=True,
            dropout=0.3
        )
        
        # Налаштування оптимізатора та функції втрат
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Навчання моделі
        start_time = time.time()
        model, history = train_lstm_model(
            model, train_dataloader, val_dataloader,
            optimizer, criterion, 20, 5, device
        )
        training_time = time.time() - start_time
        
        # Оцінка моделі на тестовій вибірці
        test_loss, test_f1 = evaluate_lstm_model(model, test_dataloader, criterion, device)
        
        # Збереження результатів
        result = {
            'n_layers': n_layers,
            'val_f1': history['best_val_f1'],
            'test_f1': test_f1,
            'test_loss': test_loss,
            'training_time': training_time,
            'epochs': len(history['train_losses'])
        }
        results.append(result)
        
        # Збереження історії навчання
        with open(os.path.join(results_dir, f'lstm_n_layers_{n_layers}_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Результати: val_f1={history['best_val_f1']:.4f}, test_f1={test_f1:.4f}, час навчання={training_time:.2f}с")
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Збереження результатів у CSV
    results_df.to_csv(os.path.join(results_dir, 'lstm_n_layers_results.csv'), index=False)
    
    # Візуалізація результатів
    plt.figure(figsize=(10, 5))
    plt.plot(n_layers_list, results_df['test_f1'], 'o-', label='F1-міра')
    plt.xlabel('Кількість шарів')
    plt.ylabel('F1-міра')
    plt.title('Вплив кількості шарів на продуктивність LSTM')
    plt.grid(True)
    plt.xticks(n_layers_list)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'lstm_n_layers_results.png'))
    
    return results_df

def run_lstm_experiments_bidirectional(train_df, val_df, test_df, device='cuda', results_dir='results/lstm'):
    """
    Експерименти з двонаправленістю LSTM
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        device: Пристрій для обчислень
        results_dir: Директорія для збереження результатів
    
    Returns:
        DataFrame з результатами експериментів
    """
    # Створення директорії для результатів
    os.makedirs(results_dir, exist_ok=True)
    
    # Підготовка даних для LSTM
    train_dataloader, val_dataloader, test_dataloader, \
    word2idx, idx2word, label2idx, idx2label, vocab_size = create_dataloaders_for_lstm(
        train_df, val_df, test_df
    )
    
    # Список для збереження результатів
    results = []
    
    # Експерименти з двонаправленістю
    for bidirectional in [False, True]:
        direction_name = 'bidirectional' if bidirectional else 'unidirectional'
        print(f"\nЕксперимент з {direction_name} LSTM")
        
        # Створення моделі
        model = initialize_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            output_dim=len(label2idx),
            n_layers=2,
            bidirectional=bidirectional,
            dropout=0.3
        )
        
        # Налаштування оптимізатора та функції втрат
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Навчання моделі
        start_time = time.time()
        model, history = train_lstm_model(
            model, train_dataloader, val_dataloader,
            optimizer, criterion, 20, 5, device
        )
        training_time = time.time() - start_time
        
        # Оцінка моделі на тестовій вибірці
        test_loss, test_f1 = evaluate_lstm_model(model, test_dataloader, criterion, device)
        
        # Збереження результатів
        result = {
            'bidirectional': bidirectional,
            'direction_name': direction_name,
            'val_f1': history['best_val_f1'],
            'test_f1': test_f1,
            'test_loss': test_loss,
            'training_time': training_time,
            'epochs': len(history['train_losses'])
        }
        results.append(result)
        
        # Збереження історії навчання
        with open(os.path.join(results_dir, f'lstm_{direction_name}_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Результати: val_f1={history['best_val_f1']:.4f}, test_f1={test_f1:.4f}, час навчання={training_time:.2f}с")
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Збереження результатів у CSV
    results_df.to_csv(os.path.join(results_dir, 'lstm_bidirectional_results.csv'), index=False)
    
    # Візуалізація результатів
    plt.figure(figsize=(8, 5))
    plt.bar(results_df['direction_name'], results_df['test_f1'])
    plt.xlabel('Тип LSTM')
    plt.ylabel('F1-міра')
    plt.title('Вплив двонаправленості на продуктивність LSTM')
    plt.grid(axis='y')
    plt.savefig(os.path.join(results_dir, 'lstm_bidirectional_results.png'))
    
    return results_df

def run_lstm_experiments_learning_rate(train_df, val_df, test_df, learning_rates=[0.0001, 0.001, 0.01], device='cuda', results_dir='results/lstm'):
    """
    Експерименти з швидкістю навчання LSTM
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        learning_rates: Список швидкостей навчання
        device: Пристрій для обчислень
        results_dir: Директорія для збереження результатів
    
    Returns:
        DataFrame з результатами експериментів
    """
    # Створення директорії для результатів
    os.makedirs(results_dir, exist_ok=True)
    
    # Підготовка даних для LSTM
    train_dataloader, val_dataloader, test_dataloader, \
    word2idx, idx2word, label2idx, idx2label, vocab_size = create_dataloaders_for_lstm(
        train_df, val_df, test_df
    )
    
    # Список для збереження результатів
    results = []
    
    # Експерименти з різними швидкостями навчання
    for lr in learning_rates:
        print(f"\nЕксперимент з learning_rate={lr}")
        
        # Створення моделі
        model = initialize_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            output_dim=len(label2idx),
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        # Налаштування оптимізатора та функції втрат
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Навчання моделі
        start_time = time.time()
        model, history = train_lstm_model(
            model, train_dataloader, val_dataloader,
            optimizer, criterion, 20, 5, device
        )
        training_time = time.time() - start_time
        
        # Оцінка моделі на тестовій вибірці
        test_loss, test_f1 = evaluate_lstm_model(model, test_dataloader, criterion, device)
        
        # Збереження результатів
        result = {
            'learning_rate': lr,
            'val_f1': history['best_val_f1'],
            'test_f1': test_f1,
            'test_loss': test_loss,
            'training_time': training_time,
            'epochs': len(history['train_losses'])
        }
        results.append(result)
        
        # Збереження історії навчання
        with open(os.path.join(results_dir, f'lstm_lr_{lr}_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Результати: val_f1={history['best_val_f1']:.4f}, test_f1={test_f1:.4f}, час навчання={training_time:.2f}с")
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Збереження результатів у CSV
    results_df.to_csv(os.path.join(results_dir, 'lstm_learning_rate_results.csv'), index=False)
    
    # Візуалізація результатів
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results_df['test_f1'], 'o-', label='F1-міра')
    plt.xscale('log')
    plt.xlabel('Швидкість навчання')
    plt.ylabel('F1-міра')
    plt.title('Вплив швидкості навчання на продуктивність LSTM')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'lstm_learning_rate_results.png'))
    
    return results_df

def run_lstm_experiments_batch_size(train_df, val_df, test_df, batch_sizes=[16, 32, 64, 128], device='cuda', results_dir='results/lstm'):
    """
    Експерименти з розміром міні-батчу LSTM
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        batch_sizes: Список розмірів міні-батчу
        device: Пристрій для обчислень
        results_dir: Директорія для збереження результатів
    
    Returns:
        DataFrame з результатами експериментів
    """
    # Створення директорії для результатів
    os.makedirs(results_dir, exist_ok=True)
    
    # Список для збереження результатів
    results = []
    
    # Експерименти з різними розмірами міні-батчу
    for batch_size in batch_sizes:
        print(f"\nЕксперимент з batch_size={batch_size}")
        
        # Підготовка даних для LSTM
        train_dataloader, val_dataloader, test_dataloader, \
        word2idx, idx2word, label2idx, idx2label, vocab_size = create_dataloaders_for_lstm(
            train_df, val_df, test_df, batch_size=batch_size
        )
        
        # Створення моделі
        model = initialize_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            output_dim=len(label2idx),
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        # Налаштування оптимізатора та функції втрат
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Навчання моделі
        start_time = time.time()
        model, history = train_lstm_model(
            model, train_dataloader, val_dataloader,
            optimizer, criterion, 20, 5, device
        )
        training_time = time.time() - start_time
        
        # Оцінка моделі на тестовій вибірці
        test_loss, test_f1 = evaluate_lstm_model(model, test_dataloader, criterion, device)
        
        # Збереження результатів
        result = {
            'batch_size': batch_size,
            'val_f1': history['best_val_f1'],
            'test_f1': test_f1,
            'test_loss': test_loss,
            'training_time': training_time,
            'epochs': len(history['train_losses'])
        }
        results.append(result)
        
        # Збереження історії навчання
        with open(os.path.join(results_dir, f'lstm_batch_size_{batch_size}_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Результати: val_f1={history['best_val_f1']:.4f}, test_f1={test_f1:.4f}, час навчання={training_time:.2f}с")
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Збереження результатів у CSV
    results_df.to_csv(os.path.join(results_dir, 'lstm_batch_size_results.csv'), index=False)
    
    # Візуалізація результатів
    plt.figure(figsize=(10, 5))
    plt.plot(batch_sizes, results_df['test_f1'], 'o-', label='F1-міра')
    plt.xlabel('Розмір міні-батчу')
    plt.ylabel('F1-міра')
    plt.title('Вплив розміру міні-батчу на продуктивність LSTM')
    plt.grid(True)
    plt.xticks(batch_sizes)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'lstm_batch_size_results.png'))
    
    return results_df

def visualize_all_lstm_experiments(results_dir='results/lstm'):
    """
    Візуалізація результатів всіх експериментів з LSTM
    
    Args:
        results_dir: Директорія з результатами експериментів
    """
    # Створення зведеного графіка для всіх експериментів
    plt.figure(figsize=(15, 12))
    
    # 1. Вплив розмірності прихованого стану
    try:
        hidden_dim_results = pd.read_csv(os.path.join(results_dir, 'lstm_hidden_dim_results.csv'))
        plt.subplot(2, 3, 1)
        plt.plot(hidden_dim_results['hidden_dim'], hidden_dim_results['test_f1'], 'o-')
        plt.xlabel('Розмірність прихованого стану')
        plt.ylabel('F1-міра')
        plt.title('Вплив розмірності прихованого стану')
        plt.grid(True)
    except:
        print("Дані про розмірність прихованого стану відсутні")
    
    # 2. Вплив кількості шарів
    try:
        n_layers_results = pd.read_csv(os.path.join(results_dir, 'lstm_n_layers_results.csv'))
        plt.subplot(2, 3, 2)
        plt.plot(n_layers_results['n_layers'], n_layers_results['test_f1'], 'o-')
        plt.xlabel('Кількість шарів')
        plt.ylabel('F1-міра')
        plt.title('Вплив кількості шарів')
        plt.grid(True)
        plt.xticks(n_layers_results['n_layers'])
    except:
        print("Дані про кількість шарів відсутні")
    
    # 3. Вплив двонаправленості
    try:
        bidirectional_results = pd.read_csv(os.path.join(results_dir, 'lstm_bidirectional_results.csv'))
        plt.subplot(2, 3, 3)
        plt.bar(bidirectional_results['direction_name'], bidirectional_results['test_f1'])
        plt.xlabel('Тип LSTM')
        plt.ylabel('F1-міра')
        plt.title('Вплив двонаправленості')
        plt.grid(axis='y')
    except:
        print("Дані про двонаправленість відсутні")
    
    # 4. Вплив швидкості навчання
    try:
        lr_results = pd.read_csv(os.path.join(results_dir, 'lstm_learning_rate_results.csv'))
        plt.subplot(2, 3, 4)
        plt.plot(lr_results['learning_rate'], lr_results['test_f1'], 'o-')
        plt.xscale('log')
        plt.xlabel('Швидкість навчання')
        plt.ylabel('F1-міра')
        plt.title('Вплив швидкості навчання')
        plt.grid(True)
    except:
        print("Дані про швидкість навчання відсутні")
    
    # 5. Вплив розміру міні-батчу
    try:
        batch_size_results = pd.read_csv(os.path.join(results_dir, 'lstm_batch_size_results.csv'))
        plt.subplot(2, 3, 5)
        plt.plot(batch_size_results['batch_size'], batch_size_results['test_f1'], 'o-')
        plt.xlabel('Розмір міні-батчу')
        plt.ylabel('F1-міра')
        plt.title('Вплив розміру міні-батчу')
        plt.grid(True)
        plt.xticks(batch_size_results['batch_size'])
    except:
        print("Дані про розмір міні-батчу відсутні")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lstm_all_experiments.png'))
    plt.show()

if __name__ == "__main__":
    # Парсинг аргументів командного рядка
    parser = argparse.ArgumentParser(description='Експерименти з гіперпараметрами LSTM моделі')
    parser.add_argument('--data_path', type=str, default='emotions_dataset.csv', help='Шлях до файлу з даними')
    parser.add_argument('--results_dir', type=str, default='results/lstm', help='Директорія для збереження результатів')
    parser.add_argument('--all', action='store_true', help='Запустити всі експерименти')
    parser.add_argument('--hidden_dim', action='store_true', help='Експерименти з розмірністю прихованого стану')
    parser.add_argument('--n_layers', action='store_true', help='Експерименти з кількістю шарів')
    parser.add_argument('--bidirectional', action='store_true', help='Експерименти з двонаправленістю')
    parser.add_argument('--learning_rate', action='store_true', help='Експерименти з швидкістю навчання')
    parser.add_argument('--batch_size', action='store_true', help='Експерименти з розміром міні-батчу')
    parser.add_argument('--visualize', action='store_true', help='Візуалізація результатів')
    args = parser.parse_args()
    
    # Визначення пристрою для обчислень
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Використовується пристрій: {device}")
    
    # Завантаження та препроцесинг даних
    print("Завантаження та препроцесинг даних...")
    train_df, val_df, test_df = load_and_preprocess_data(args.data_path)
    
    # Запуск експериментів
    if args.all or args.hidden_dim:
        print("\nЕксперименти з розмірністю прихованого стану LSTM")
        run_lstm_experiments_hidden_dim(train_df, val_df, test_df, device=device, results_dir=args.results_dir)
    
    if args.all or args.n_layers:
        print("\nЕксперименти з кількістю шарів LSTM")
        run_lstm_experiments_n_layers(train_df, val_df, test_df, device=device, results_dir=args.results_dir)
    
    if args.all or args.bidirectional:
        print("\nЕксперименти з двонаправленістю LSTM")
        run_lstm_experiments_bidirectional(train_df, val_df, test_df, device=device, results_dir=args.results_dir)
    
    if args.all or args.learning_rate:
        print("\nЕксперименти з швидкістю навчання LSTM")
        run_lstm_experiments_learning_rate(train_df, val_df, test_df, device=device, results_dir=args.results_dir)
    
    if args.all or args.batch_size:
        print("\nЕксперименти з розміром міні-батчу LSTM")
        run_lstm_experiments_batch_size(train_df, val_df, test_df, device=device, results_dir=args.results_dir)
    
    if args.all or args.visualize:
        print("\nВізуалізація результатів експериментів з LSTM")
        visualize_all_lstm_experiments(args.results_dir)