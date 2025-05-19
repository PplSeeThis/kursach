"""
data_size_comparison.py
Модуль для порівняння моделей на різних обсягах даних
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Імпортуємо наші модулі
from lstm_model import initialize_lstm_model, create_dataloaders_for_lstm
from bert_lite_model import create_bert_lite_model, prepare_data_for_bert
from training import train_lstm_model, train_bert_lite_model, evaluate_lstm_model, evaluate_bert_lite_model
from transformers import BertTokenizer

def compare_models_on_different_data_sizes(train_df, val_df, test_df, fractions=[0.1, 0.25, 0.5, 0.75, 1.0], device='cuda', results_dir='results'):
    """
    Порівняння LSTM та BERT-lite моделей на різних обсягах навчальних даних
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        fractions: Частки навчальних даних
        device: Пристрій для обчислень
        results_dir: Директорія для збереження результатів
    
    Returns:
        Словник з результатами порівняння
    """
    # Створення директорії для результатів
    os.makedirs(results_dir, exist_ok=True)
    
    # Перетворення фракцій у відсотки для графіка
    data_sizes = [fraction * 100 for fraction in fractions]
    
    # Результати
    lstm_f1s = []
    bert_f1s = []
    
    # Для кожної частки даних
    for i, fraction in enumerate(fractions):
        subset_size = int(len(train_df) * fraction)
        print(f"\nТренування на {fraction*100}% даних ({subset_size} зразків)")
        
        # Створення підмножини даних
        subset_indices = np.random.choice(len(train_df), subset_size, replace=False)
        subset_train_df = train_df.iloc[subset_indices]
        
        # Підготовка даних та навчання LSTM моделі
        print("Навчання LSTM моделі...")
        lstm_f1 = train_and_evaluate_lstm(subset_train_df, val_df, test_df, device)
        lstm_f1s.append(lstm_f1)
        
        # Підготовка даних та навчання BERT-lite моделі
        print("Навчання BERT-lite моделі...")
        bert_f1 = train_and_evaluate_bert_lite(subset_train_df, val_df, test_df, device)
        bert_f1s.append(bert_f1)
        
        print(f"LSTM F1: {lstm_f1:.4f}, BERT-lite F1: {bert_f1:.4f}")
        
    # Збереження результатів
    results = {
        'data_sizes': data_sizes,
        'lstm_f1s': lstm_f1s,
        'bert_f1s': bert_f1s
    }
    
    # Створення графіка
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, lstm_f1s, 'o-', label='LSTM')
    plt.plot(data_sizes, bert_f1s, 's-', label='BERT-lite')
    plt.xlabel('Відсоток навчальних даних (%)')
    plt.ylabel('F1-міра')
    plt.title('Залежність F1-міри від обсягу даних')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'data_size_comparison.png'))
    
    # Збереження результатів у CSV
    results_df = pd.DataFrame({
        'data_size': data_sizes,
        'lstm_f1': lstm_f1s,
        'bert_f1': bert_f1s,
        'difference': np.array(bert_f1s) - np.array(lstm_f1s)
    })
    results_df.to_csv(os.path.join(results_dir, 'data_size_comparison.csv'), index=False)
    
    return results

def train_and_evaluate_lstm(train_df, val_df, test_df, device):
    """
    Навчання та оцінка LSTM моделі
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        device: Пристрій для обчислень
        
    Returns:
        F1-міра на тестовій вибірці
    """
    # Підготовка даних для LSTM
    train_dataloader, val_dataloader, test_dataloader, \
    word2idx, idx2word, label2idx, idx2label, vocab_size = create_dataloaders_for_lstm(
        train_df, val_df, test_df
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
    
    # Налаштування оптимізатора
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Навчання моделі
    model, history = train_lstm_model(
        model, train_dataloader, val_dataloader,
        optimizer, criterion, 20, 5, device
    )
    
    # Оцінка моделі на тестовій вибірці
    test_loss, test_f1 = evaluate_lstm_model(model, test_dataloader, criterion, device)
    
    return test_f1

def train_and_evaluate_bert_lite(train_df, val_df, test_df, device):
    """
    Навчання та оцінка BERT-lite моделі
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        device: Пристрій для обчислень
        
    Returns:
        F1-міра на тестовій вибірці
    """
    # Токенізатор BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Підготовка даних для BERT-lite
    train_dataloader, val_dataloader, test_dataloader, \
    label2idx, idx2label, num_classes = prepare_data_for_bert(
        train_df, val_df, test_df, tokenizer
    )
    
    # Створення моделі
    model = create_bert_lite_model(
        num_classes=num_classes,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.3,
        device=device
    )
    
    # Налаштування оптимізатора
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Навчання моделі
    model, history = train_bert_lite_model(
        model, train_dataloader, val_dataloader,
        optimizer, criterion, 10, 3, device
    )
    
    # Оцінка моделі на тестовій вибірці
    test_loss, test_f1 = evaluate_bert_lite_model(model, test_dataloader, criterion, device)
    
    return test_f1

if __name__ == "__main__":
    # Завантаження та препроцесинг даних
    from preprocessing import load_and_preprocess_data
    
    train_df, val_df, test_df = load_and_preprocess_data('emotions_dataset.csv')
    
    # Визначення пристрою для обчислень
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Порівняння моделей на різних обсягах даних
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = compare_models_on_different_data_sizes(train_df, val_df, test_df, fractions, device)
    
    # Виведення результатів
    print("\nРезультати порівняння:")
    for i, size in enumerate(results['data_sizes']):
        print(f"Обсяг даних: {size}%")
        print(f"LSTM F1: {results['lstm_f1s'][i]:.4f}")
        print(f"BERT-lite F1: {results['bert_f1s'][i]:.4f}")
        print(f"Різниця: {results['bert_f1s'][i] - results['lstm_f1s'][i]:.4f}")
        print("")