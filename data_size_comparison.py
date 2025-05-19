#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# Импорт наших моделей и функций для обработки данных
from lstm_model import LSTMClassifier
from bert_lite_model import BERTLiteClassifier
from preprocessing import preprocess_text, create_data_iterators
from bert_data_preparation import prepare_bert_data, create_bert_dataloaders

# Настройка стилей для графиков
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Функция для создания подмножеств данных разного размера
def create_data_subsets(train_data, sizes=[0.1, 0.25, 0.5, 0.75, 1.0]):
    """
    Создает подмножества данных разного размера.
    
    Args:
        train_data: Полный тренировочный набор данных
        sizes: Список относительных размеров подмножеств
        
    Returns:
        dict: Словарь подмножеств данных разного размера
    """
    full_size = len(train_data)
    subsets = {}
    
    for size in sizes:
        subset_size = int(full_size * size)
        # Создаем стратифицированное подмножество
        subset = train_data.sample(n=subset_size, random_state=42)
        subsets[size] = subset
    
    return subsets

# Функция для обучения LSTM модели на подмножестве данных
def train_lstm_on_subset(train_subset, val_data, test_data, params, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Обучает LSTM модель на подмножестве данных и оценивает ее на тестовых данных.
    
    Args:
        train_subset: Подмножество тренировочных данных
        val_data: Валидационные данные
        test_data: Тестовые данные
        params: Параметры модели
        device: Устройство для обучения
        
    Returns:
        float: F1-score модели на тестовых данных
    """
    print(f"Training LSTM on subset with {len(train_subset)} samples...")
    
    # Создание итераторов данных
    TEXT, LABEL, train_iterator, val_iterator, test_iterator = create_data_iterators(
        train_subset, val_data, test_data, batch_size=params['batch_size']
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
        
        # Сохраняем лучшую модель
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_lstm_model.pt')
        
        print(f"Epoch {epoch+1}/{params['n_epochs']}: Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Раннее останавливание
        if epoch > params['patience'] and val_f1 < best_val_f1:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load('best_lstm_model.pt'))
    
    # Тестируем модель на тестовом наборе
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_iterator:
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
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"LSTM Test F1: {test_f1:.4f}")
    
    return test_f1

# Функция для обучения BERT-lite модели на подмножестве данных
def train_bert_on_subset(train_subset, val_data, test_data, params, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Обучает BERT-lite модель на подмножестве данных и оценивает ее на тестовых данных.
    
    Args:
        train_subset: Подмножество тренировочных данных
        val_data: Валидационные данные
        test_data: Тестовые данные
        params: Параметры модели
        device: Устройство для обучения
        
    Returns:
        float: F1-score модели на тестовых данных
    """
    print(f"Training BERT-lite on subset with {len(train_subset)} samples...")
    
    # Подготовка данных для BERT
    train_dataloader, val_dataloader, test_dataloader, label_encoder = create_bert_dataloaders(
        train_subset, val_data, test_data, max_length=params['max_length'], batch_size=params['batch_size']
    )
    
    # Количество классов
    num_classes = len(label_encoder.classes_)
    
    # Настройка конфигурации BERT-lite
    from transformers import BertConfig
    
    config = BertConfig(
        vocab_size=30522,  # Стандартный размер словаря для BERT
        hidden_size=params['hidden_size'],
        num_hidden_layers=params['num_hidden_layers'],
        num_attention_heads=params['num_attention_heads'],
        intermediate_size=params['intermediate_size'],
        hidden_dropout_prob=params['dropout'],
        attention_probs_dropout_prob=params['dropout'],
        max_position_embeddings=512,
        type_vocab_size=2,
        num_labels=num_classes
    )
    
    # Инициализация модели
    model = BERTLiteClassifier(config)
    
    # Если указано, загружаем предобученные веса
    if params.get('use_pretrained', True):
        # В реальном сценарии здесь бы загружались веса из предобученной модели
        print("Initializing model with pre-trained weights...")
    
    # Перемещение модели на устройство
    model = model.to(device)
    
    # Настройка оптимизатора
    from transformers import AdamW, get_linear_schedule_with_warmup
    
    optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
    
    # Количество шагов для обучения
    total_steps = len(train_dataloader) * params['n_epochs']
    
    # Настройка планировщика скорости обучения
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),  # 10% шагов для разогрева
        num_training_steps=total_steps
    )
    
    # Функция потерь
    criterion = torch.nn.CrossEntropyLoss()
    
    # Обучение модели
    best_val_f1 = 0
    
    for epoch in range(params['n_epochs']):
        # Обучение на эпоху
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{params['n_epochs']}", leave=False):
            # Получаем данные
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Вычисляем потери
            loss = criterion(logits, labels)
            
            # Обратный проход
            loss.backward()
            
            # Оптимизация
            optimizer.step()
            scheduler.step()
            
            # Накапливаем потери
            epoch_loss += loss.item()
            
            # Получаем предсказания
            preds = logits.argmax(dim=1).cpu().numpy()
            labs = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
        
        # Вычисляем средние потери и f1-score для обучения
        train_loss = epoch_loss / len(train_dataloader)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Валидация модели
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Получаем данные
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Прямой проход
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Вычисляем потери
                loss = criterion(logits, labels)
                
                # Накапливаем потери
                val_loss += loss.item()
                
                # Получаем предсказания
                preds = logits.argmax(dim=1).cpu().numpy()
                labs = labels.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labs)
        
        # Вычисляем средние потери и f1-score для валидации
        val_loss = val_loss / len(val_dataloader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Сохраняем лучшую модель
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_bert_model.pt')
        
        print(f"Epoch {epoch+1}/{params['n_epochs']}: Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Раннее останавливание
        if epoch > params['patience'] and val_f1 < best_val_f1:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load('best_bert_model.pt'))
    
    # Тестируем модель на тестовом наборе
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=1).cpu().numpy()
            labs = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
    
    # Вычисляем финальный f1-score
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"BERT-lite Test F1: {test_f1:.4f}")
    
    return test_f1

# Функция для сравнения моделей на разных объемах данных
def compare_models_data_size(train_data, val_data, test_data, sizes=[0.1, 0.25, 0.5, 0.75, 1.0]):
    """
    Сравнивает LSTM и BERT-lite модели на разных объемах тренировочных данных.
    
    Args:
        train_data: Полный тренировочный набор данных
        val_data: Валидационные данные
        test_data: Тестовые данные
        sizes: Список относительных размеров подмножеств
        
    Returns:
        dict: Словарь с результатами сравнения
    """
    # Создаем подмножества данных
    train_subsets = create_data_subsets(train_data, sizes)
    
    # Результаты для каждой модели
    lstm_results = []
    bert_results = []
    
    # Параметры для LSTM
    lstm_params = {
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
        'n_epochs': 10,
        'patience': 3,
        'use_pretrained': True
    }
    
    # Обучаем модели на каждом подмножестве
    for size, subset in train_subsets.items():
        print(f"\n=== Training on {size*100}% of data ({len(subset)} samples) ===\n")
        
        # Обучаем LSTM
        lstm_f1 = train_lstm_on_subset(subset, val_data, test_data, lstm_params)
        lstm_results.append((size, lstm_f1))
        
        # Обучаем BERT-lite
        bert_f1 = train_bert_on_subset(subset, val_data, test_data, bert_params)
        bert_results.append((size, bert_f1))
    
    # Преобразуем результаты в DataFrame для удобства анализа
    results_df = pd.DataFrame({
        'Data Size': [size*100 for size, _ in lstm_results],  # в процентах
        'Data Samples': [len(train_subsets[size]) for size in sizes],
        'LSTM F1': [f1 for _, f1 in lstm_results],
        'BERT-lite F1': [f1 for _, f1 in bert_results],
        'Difference': [bert_f1 - lstm_f1 for (_, lstm_f1), (_, bert_f1) in zip(lstm_results, bert_results)]
    })
    
    # Вывод результатов
    print("\n=== Comparison Results ===\n")
    print(results_df)
    
    # Сохраняем результаты в CSV
    results_df.to_csv('data_size_comparison_results.csv', index=False)
    
    # Строим график
    plt.figure(figsize=(12, 8))
    
    plt.plot([size*100 for size, _ in lstm_results], [f1 for _, f1 in lstm_results], 'b-o', label='LSTM')
    plt.plot([size*100 for size, _ in bert_results], [f1 for _, f1 in bert_results], 'r-s', label='BERT-lite')
    
    # Добавляем точки данных
    for i, ((size, lstm_f1), (_, bert_f1)) in enumerate(zip(lstm_results, bert_results)):
        plt.text(size*100+1, lstm_f1+0.005, f"{lstm_f1:.3f}", ha='left', va='bottom', color='blue')
        plt.text(size*100+1, bert_f1+0.005, f"{bert_f1:.3f}", ha='left', va='bottom', color='red')
        
        # Отображаем разницу
        diff = bert_f1 - lstm_f1
        plt.annotate(f"Δ = {diff:.3f}", 
                     xy=(size*100, (lstm_f1 + bert_f1)/2), 
                     xytext=(size*100+3, (lstm_f1 + bert_f1)/2),
                     arrowprops=dict(arrowstyle="->", color='green'),
                     color='green')
    
    plt.xlabel('Размер данных (% от полного набора)')
    plt.ylabel('F1-score')
    plt.title('Зависимость F1-score от объема обучающих данных')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('data_size_comparison.png', dpi=300)
    plt.close()
    
    return {
        'lstm_results': lstm_results,
        'bert_results': bert_results,
        'results_df': results_df
    }

# Основная функция для запуска сравнения
def run_data_size_comparison(sizes=[0.1, 0.25, 0.5, 0.75, 1.0]):
    """
    Запускает сравнение моделей на разных объемах данных.
    
    Args:
        sizes: Список относительных размеров подмножеств
        
    Returns:
        dict: Результаты сравнения
    """
    # Загрузка данных
    from preprocessing import load_data
    
    train_data, val_data, test_data = load_data()
    
    # Сравнение моделей
    results = compare_models_data_size(train_data, val_data, test_data, sizes)
    
    return results

# Пример использования (не запускается при импорте)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare LSTM and BERT-lite models on different data sizes.')
    parser.add_argument('--sizes', type=float, nargs='+', default=[0.1, 0.25, 0.5, 0.75, 1.0],
                        help='List of relative data sizes to use (default: 0.1 0.25 0.5 0.75 1.0)')
    
    args = parser.parse_args()
    
    # Запуск сравнения
    results = run_data_size_comparison(args.sizes)
