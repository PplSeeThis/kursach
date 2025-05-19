"""
hyperparameter_experiments.py
Модуль для експериментів з гіперпараметрами моделей класифікації емоцій
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
from sklearn.metrics import f1_score

# Імпортуємо наші модулі
from lstm_model import initialize_lstm_model, create_dataloaders_for_lstm
from bert_lite_model import create_bert_lite_model, prepare_data_for_bert
from training import train_lstm_model, train_bert_lite_model, evaluate_lstm_model, evaluate_bert_lite_model
from preprocessing import load_and_preprocess_data

def run_lstm_hyperparameter_experiments(train_df, val_df, test_df, experiment_configs, device='cuda', results_dir='results'):
    """
    Проведення експериментів з гіперпараметрами LSTM моделі
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        experiment_configs: Список конфігурацій для експериментів
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
    
    # Проведення експериментів
    for i, config in enumerate(experiment_configs):
        print(f"\nЕксперимент {i+1}/{len(experiment_configs)}:")
        print(f"Конфігурація: {config}")
        
        # Створення моделі з заданою конфігурацією
        model = initialize_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 300),
            hidden_dim=config.get('hidden_dim', 256),
            output_dim=len(label2idx),
            n_layers=config.get('n_layers', 2),
            bidirectional=config.get('bidirectional', True),
            dropout=config.get('dropout', 0.3)
        )
        
        # Налаштування оптимізатора
        optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        # Навчання моделі
        model, history = train_lstm_model(
            model, train_dataloader, val_dataloader,
            optimizer, criterion, 
            n_epochs=config.get('n_epochs', 20), 
            patience=config.get('patience', 5), 
            device=device
        )
        
        # Оцінка моделі на тестовій вибірці
        test_loss, test_f1 = evaluate_lstm_model(model, test_dataloader, criterion, device)
        
        # Збереження результатів
        result = {
            'config': config,
            'best_val_f1': history['best_val_f1'],
            'test_f1': test_f1,
            'test_loss': test_loss,
            'training_time': history['training_time']
        }
        results.append(result)
        
        # Збереження історії навчання
        config_str = f"lstm_hidden{config.get('hidden_dim', 256)}_layers{config.get('n_layers', 2)}"
        config_str += f"_{'bi' if config.get('bidirectional', True) else 'uni'}"
        
        with open(os.path.join(results_dir, f"{config_str}_history.json"), 'w') as f:
            # Конвертація numpy.float32 у float для серіалізації JSON
            history_json = {k: [float(val) for val in v] if isinstance(v, list) else float(v) 
                           for k, v in history.items()}
            json.dump(history_json, f, indent=4)
            
        print(f"Результати: Val F1: {history['best_val_f1']:.4f}, Test F1: {test_f1:.4f}")
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Збереження результатів у CSV
    results_df.to_csv(os.path.join(results_dir, 'lstm_experiments_results.csv'), index=False)
    
    return results_df

def run_bert_lite_hyperparameter_experiments(train_df, val_df, test_df, experiment_configs, device='cuda', results_dir='results'):
    """
    Проведення експериментів з гіперпараметрами BERT-lite моделі
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        experiment_configs: Список конфігурацій для експериментів
        device: Пристрій для обчислень
        results_dir: Директорія для збереження результатів
    
    Returns:
        DataFrame з результатами експериментів
    """
    # Створення директорії для результатів
    os.makedirs(results_dir, exist_ok=True)
    
    # Токенізатор BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Список для збереження результатів
    results = []
    
    # Проведення експериментів
    for i, config in enumerate(experiment_configs):
        print(f"\nЕксперимент {i+1}/{len(experiment_configs)}:")
        print(f"Конфігурація: {config}")
        
        # Підготовка даних для BERT-lite (для кожного експерименту окремо, щоб враховувати різні max_length)
        train_dataloader, val_dataloader, test_dataloader, \
        label2idx, idx2label, num_classes = prepare_data_for_bert(
            train_df, val_df, test_df, tokenizer,
            max_length=config.get('max_length', 128),
            batch_size=config.get('batch_size', 32)
        )
        
        # Створення моделі з заданою конфігурацією
        model = create_bert_lite_model(
            num_classes=num_classes,
            hidden_size=config.get('hidden_size', 512),
            num_hidden_layers=config.get('num_hidden_layers', 6),
            num_attention_heads=config.get('num_attention_heads', 8),
            intermediate_size=config.get('intermediate_size', 2048),
            dropout=config.get('dropout', 0.3),
            device=device
        )
        
        # Налаштування оптимізатора
        optimizer = optim.AdamW(model.parameters(), lr=config.get('learning_rate', 2e-5))
        criterion = nn.CrossEntropyLoss()
        
        # Навчання моделі
        model, history = train_bert_lite_model(
            model, train_dataloader, val_dataloader,
            optimizer, criterion, 
            n_epochs=config.get('n_epochs', 10), 
            patience=config.get('patience', 3), 
            device=device
        )
        
        # Оцінка моделі на тестовій вибірці
        test_loss, test_f1 = evaluate_bert_lite_model(model, test_dataloader, criterion, device)
        
        # Збереження результатів
        result = {
            'config': config,
            'best_val_f1': history['best_val_f1'],
            'test_f1': test_f1,
            'test_loss': test_loss,
            'training_time': history['training_time']
        }
        results.append(result)
        
        # Збереження історії навчання
        config_str = f"bert_hidden{config.get('hidden_size', 512)}_layers{config.get('num_hidden_layers', 6)}"
        config_str += f"_heads{config.get('num_attention_heads', 8)}"
        
        with open(os.path.join(results_dir, f"{config_str}_history.json"), 'w') as f:
            # Конвертація numpy.float32 у float для серіалізації JSON
            history_json = {k: [float(val) for val in v] if isinstance(v, list) else float(v) 
                           for k, v in history.items()}
            json.dump(history_json, f, indent=4)
            
        print(f"Результати: Val F1: {history['best_val_f1']:.4f}, Test F1: {test_f1:.4f}")
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Збереження результатів у CSV
    results_df.to_csv(os.path.join(results_dir, 'bert_lite_experiments_results.csv'), index=False)
    
    return results_df

def visualize_lstm_architecture_experiments(results_df, output_dir='results'):
    """
    Візуалізація результатів експериментів з архітектурними параметрами LSTM
    
    Args:
        results_df: DataFrame з результатами експериментів
        output_dir: Директорія для збереження графіків
    """
    # Створення директорії для збереження графіків
    os.makedirs(output_dir, exist_ok=True)
    
    # Вплив розмірності прихованого стану
    hidden_dims = []
    hidden_dim_f1s = []
    
    for _, row in results_df[results_df['config'].apply(lambda x: x.get('n_layers') == 2 and x.get('bidirectional') == True)].iterrows():
        hidden_dims.append(row['config'].get('hidden_dim'))
        hidden_dim_f1s.append(row['test_f1'])
    
    # Вплив кількості шарів
    n_layers = []
    n_layers_f1s = []
    
    for _, row in results_df[results_df['config'].apply(lambda x: x.get('hidden_dim') == 256 and x.get('bidirectional') == True)].iterrows():
        n_layers.append(row['config'].get('n_layers'))
        n_layers_f1s.append(row['test_f1'])
    
    # Вплив двонаправленості
    directions = []
    directions_f1s = []
    
    for _, row in results_df[results_df['config'].apply(lambda x: x.get('hidden_dim') == 256 and x.get('n_layers') == 2)].iterrows():
        directions.append('BiLSTM' if row['config'].get('bidirectional') else 'LSTM')
        directions_f1s.append(row['test_f1'])
    
    # Створення графіка
    plt.figure(figsize=(15, 5))
    
    # Графік впливу розмірності прихованого стану
    plt.subplot(1, 3, 1)
    plt.bar(range(len(hidden_dims)), hidden_dim_f1s, tick_label=[str(dim) for dim in hidden_dims])
    plt.xlabel('Розмірність прихованого стану')
    plt.ylabel('F1-міра')
    plt.title('Вплив розмірності прихованого стану')
    plt.grid(axis='y')
    
    # Графік впливу кількості шарів
    plt.subplot(1, 3, 2)
    plt.bar(range(len(n_layers)), n_layers_f1s, tick_label=[str(layer) for layer in n_layers])
    plt.xlabel('Кількість шарів')
    plt.ylabel('F1-міра')
    plt.title('Вплив кількості шарів')
    plt.grid(axis='y')
    
    # Графік впливу двонаправленості
    plt.subplot(1, 3, 3)
    plt.bar(range(len(directions)), directions_f1s, tick_label=directions)
    plt.xlabel('Тип LSTM')
    plt.ylabel('F1-міра')
    plt.title('Вплив двонаправленості')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_architecture_comparison.png'))
    plt.show()

def visualize_bert_lite_architecture_experiments(results_df, output_dir='results'):
    """
    Візуалізація результатів експериментів з архітектурними параметрами BERT-lite
    
    Args:
        results_df: DataFrame з результатами експериментів
        output_dir: Директорія для збереження графіків
    """
    # Створення директорії для збереження графіків
    os.makedirs(output_dir, exist_ok=True)
    
    # Вплив кількості шарів
    num_layers = []
    num_layers_f1s = []
    
    for _, row in results_df[results_df['config'].apply(lambda x: x.get('num_attention_heads') == 8 and x.get('hidden_size') == 512)].iterrows():
        num_layers.append(row['config'].get('num_hidden_layers'))
        num_layers_f1s.append(row['test_f1'])
    
    # Вплив кількості головок уваги
    num_heads = []
    num_heads_f1s = []
    
    for _, row in results_df[results_df['config'].apply(lambda x: x.get('num_hidden_layers') == 6 and x.get('hidden_size') == 512)].iterrows():
        num_heads.append(row['config'].get('num_attention_heads'))
        num_heads_f1s.append(row['test_f1'])
    
    # Вплив розмірності прихованого стану
    hidden_sizes = []
    hidden_sizes_f1s = []
    
    for _, row in results_df[results_df['config'].apply(lambda x: x.get('num_hidden_layers') == 6 and x.get('num_attention_heads') == 8)].iterrows():
        hidden_sizes.append(row['config'].get('hidden_size'))
        hidden_sizes_f1s.append(row['test_f1'])
    
    # Створення графіка
    plt.figure(figsize=(15, 5))
    
    # Графік впливу кількості шарів
    plt.subplot(1, 3, 1)
    plt.bar(range(len(num_layers)), num_layers_f1s, tick_label=[str(layer) for layer in num_layers])
    plt.xlabel('Кількість шарів трансформера')
    plt.ylabel('F1-міра')
    plt.title('Вплив кількості шарів')
    plt.grid(axis='y')
    
    # Графік впливу кількості головок уваги
    plt.subplot(1, 3, 2)
    plt.bar(range(len(num_heads)), num_heads_f1s, tick_label=[str(head) for head in num_heads])
    plt.xlabel('Кількість головок уваги')
    plt.ylabel('F1-міра')
    plt.title('Вплив кількості головок уваги')
    plt.grid(axis='y')
    
    # Графік впливу розмірності прихованого стану
    plt.subplot(1, 3, 3)
    plt.bar(range(len(hidden_sizes)), hidden_sizes_f1s, tick_label=[str(size) for size in hidden_sizes])
    plt.xlabel('Розмірність прихованого стану')
    plt.ylabel('F1-міра')
    plt.title('Вплив розмірності прихованого стану')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bert_architecture_comparison.png'))
    plt.show()

def visualize_learning_hyperparameters(results_df, model_type, output_dir='results'):
    """
    Візуалізація результатів експериментів з гіперпараметрами навчання
    
    Args:
        results_df: DataFrame з результатами експериментів
        model_type: Тип моделі ('lstm' або 'bert')
        output_dir: Директорія для збереження графіків
    """
    # Створення директорії для збереження графіків
    os.makedirs(output_dir, exist_ok=True)
    
    # Вплив швидкості навчання
    learning_rates = []
    learning_rates_f1s = []
    
    if model_type == 'lstm':
        base_config_filter = lambda x: x.get('batch_size') == 64
        lr_key = 'learning_rate'
        batch_size_key = 'batch_size'
    else:  # bert
        base_config_filter = lambda x: x.get('batch_size') == 32
        lr_key = 'learning_rate'
        batch_size_key = 'batch_size'
    
    for _, row in results_df[results_df['config'].apply(base_config_filter)].iterrows():
        learning_rates.append(row['config'].get(lr_key))
        learning_rates_f1s.append(row['test_f1'])
    
    # Вплив розміру батчу
    batch_sizes = []
    batch_sizes_f1s = []
    
    if model_type == 'lstm':
        base_config_filter = lambda x: x.get('learning_rate') == 0.001
    else:  # bert
        base_config_filter = lambda x: x.get('learning_rate') == 2e-5
    
    for _, row in results_df[results_df['config'].apply(base_config_filter)].iterrows():
        batch_sizes.append(row['config'].get(batch_size_key))
        batch_sizes_f1s.append(row['test_f1'])
    
    # Створення графіка
    plt.figure(figsize=(12, 5))
    
    # Графік впливу швидкості навчання
    plt.subplot(1, 2, 1)
    plt.bar(range(len(learning_rates)), learning_rates_f1s, tick_label=[str(lr) for lr in learning_rates])
    plt.xlabel('Швидкість навчання')
    plt.ylabel('F1-міра')
    plt.title('Вплив швидкості навчання')
    plt.grid(axis='y')
    
    # Графік впливу розміру батчу
    plt.subplot(1, 2, 2)
    plt.bar(range(len(batch_sizes)), batch_sizes_f1s, tick_label=[str(bs) for bs in batch_sizes])
    plt.xlabel('Розмір міні-батчу')
    plt.ylabel('F1-міра')
    plt.title('Вплив розміру міні-батчу')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_learning_params.png'))
    plt.show()

if __name__ == "__main__":
    # Завантаження та препроцесинг даних
    train_df, val_df, test_df = load_and_preprocess_data('emotions_dataset.csv')
    
    # Визначення пристрою для обчислень
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Конфігурації для експериментів з LSTM
    lstm_architecture_configs = [
        # Експерименти з розмірністю прихованого стану
        {'hidden_dim': 128, 'n_layers': 2, 'bidirectional': True},
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True},
        {'hidden_dim': 512, 'n_layers': 2, 'bidirectional': True},
        
        # Експерименти з кількістю шарів
        {'hidden_dim': 256, 'n_layers': 1, 'bidirectional': True},
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True},
        {'hidden_dim': 256, 'n_layers': 3, 'bidirectional': True},
        
        # Експерименти з двонаправленістю
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': False},
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True},
    ]
    
    lstm_learning_configs = [
        # Експерименти з швидкістю навчання
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True, 'learning_rate': 0.0001, 'batch_size': 64},
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True, 'learning_rate': 0.001, 'batch_size': 64},
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True, 'learning_rate': 0.01, 'batch_size': 64},
        
        # Експерименти з розміром батчу
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True, 'learning_rate': 0.001, 'batch_size': 16},
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True, 'learning_rate': 0.001, 'batch_size': 32},
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True, 'learning_rate': 0.001, 'batch_size': 64},
        {'hidden_dim': 256, 'n_layers': 2, 'bidirectional': True, 'learning_rate': 0.001, 'batch_size': 128},
    ]
    
    # Конфігурації для експериментів з BERT-lite
    bert_architecture_configs = [
        # Експерименти з кількістю шарів
        {'num_hidden_layers': 4, 'num_attention_heads': 8, 'hidden_size': 512},
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512},
        {'num_hidden_layers': 8, 'num_attention_heads': 8, 'hidden_size': 512},
        
        # Експерименти з кількістю головок уваги
        {'num_hidden_layers': 6, 'num_attention_heads': 4, 'hidden_size': 512},
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512},
        {'num_hidden_layers': 6, 'num_attention_heads': 12, 'hidden_size': 512},
        
        # Експерименти з розмірністю прихованого стану
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 384},
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512},
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 768},
    ]
    
    bert_learning_configs = [
        # Експерименти з швидкістю навчання
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512, 'learning_rate': 5e-6, 'batch_size': 32},
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512, 'learning_rate': 2e-5, 'batch_size': 32},
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512, 'learning_rate': 5e-5, 'batch_size': 32},
        
        # Експерименти з розміром батчу
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512, 'learning_rate': 2e-5, 'batch_size': 16},
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512, 'learning_rate': 2e-5, 'batch_size': 32},
        {'num_hidden_layers': 6, 'num_attention_heads': 8, 'hidden_size': 512, 'learning_rate': 2e-5, 'batch_size': 64},
    ]
    
    # Проведення експериментів
    print("Проведення експериментів з архітектурними параметрами LSTM...")
    lstm_architecture_results = run_lstm_hyperparameter_experiments(
        train_df, val_df, test_df, lstm_architecture_configs, device, 'results/lstm_architecture'
    )
    
    print("\nПроведення експериментів з гіперпараметрами навчання LSTM...")
    lstm_learning_results = run_lstm_hyperparameter_experiments(
        train_df, val_df, test_df, lstm_learning_configs, device, 'results/lstm_learning'
    )
    
    print("\nПроведення експериментів з архітектурними параметрами BERT-lite...")
    bert_architecture_results = run_bert_lite_hyperparameter_experiments(
        train_df, val_df, test_df, bert_architecture_configs, device, 'results/bert_architecture'
    )
    
    print("\nПроведення експериментів з гіперпараметрами навчання BERT-lite...")
    bert_learning_results = run_bert_lite_hyperparameter_experiments(
        train_df, val_df, test_df, bert_learning_configs, device, 'results/bert_learning'
    )
    
    # Візуалізація результатів
    print("\nВізуалізація результатів...")
    visualize_lstm_architecture_experiments(lstm_architecture_results, 'results')
    visualize_bert_lite_architecture_experiments(bert_architecture_results, 'results')
    visualize_learning_hyperparameters(lstm_learning_results, 'lstm', 'results')
    visualize_learning_hyperparameters(bert_learning_results, 'bert', 'results')