#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import torch
from tqdm import tqdm

# Импорт наших моделей и функций для обработки данных
from lstm_model import LSTMClassifier
from bert_lite_model import BERTLiteClassifier
from preprocessing import load_data, create_data_iterators
from bert_data_preparation import create_bert_dataloaders

# Настройка стилей для графиков
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Функция для создания и сохранения матрицы ошибок
def create_confusion_matrix(y_true, y_pred, class_names, model_name, normalize=True):
    """
    Создает и сохраняет матрицу ошибок.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        class_names: Имена классов
        model_name: Название модели для добавления в имя файла
        normalize: Нормализовать матрицу (True) или нет (False)
        
    Returns:
        np.ndarray: Матрица ошибок
    """
    # Создаем матрицу ошибок
    cm = confusion_matrix(y_true, y_pred)
    
    # Нормализуем матрицу, если требуется
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.title(f'Матрица ошибок для модели {model_name}')
    
    # Сохраняем изображение
    plt.tight_layout()
    filename = f"{model_name.lower().replace('-', '_')}_confusion_matrix.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Матрица ошибок сохранена в файл {filename}")
    
    return cm

# Функция для оценки LSTM модели
def evaluate_lstm_model(model_path, test_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Оценивает LSTM модель на тестовых данных и создает матрицу ошибок.
    
    Args:
        model_path: Путь к сохраненной модели
        test_data: Тестовые данные
        device: Устройство для инференса
        
    Returns:
        tuple: (f1_score, classification_report, confusion_matrix)
    """
    print("Оценка LSTM модели...")
    
    # Создание итераторов данных
    TEXT, LABEL, test_iterator = create_data_iterators(None, None, test_data, test_only=True)
    
    # Параметры модели
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = len(LABEL.vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3
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
    
    # Загрузка весов модели
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Получение предсказаний
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_iterator, desc="Инференс"):
            text, text_lengths = batch.text
            labels = batch.label
            
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            
            predictions = model(text, text_lengths).squeeze(1)
            preds = predictions.argmax(dim=1).cpu().numpy()
            labs = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
    
    # Вычисление метрик
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=LABEL.vocab.itos)
    
    # Создание матрицы ошибок
    cm = create_confusion_matrix(all_labels, all_preds, LABEL.vocab.itos, "LSTM")
    
    print(f"LSTM F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return f1, report, cm

# Функция для оценки BERT-lite модели
def evaluate_bert_model(model_path, test_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Оценивает BERT-lite модель на тестовых данных и создает матрицу ошибок.
    
    Args:
        model_path: Путь к сохраненной модели
        test_data: Тестовые данные
        device: Устройство для инференса
        
    Returns:
        tuple: (f1_score, classification_report, confusion_matrix)
    """
    print("Оценка BERT-lite модели...")
    
    # Подготовка данных для BERT
    _, _, test_dataloader, label_encoder = create_bert_dataloaders(None, None, test_data, test_only=True)
    
    # Количество классов
    num_classes = len(label_encoder.classes_)
    
    # Настройка конфигурации BERT-lite
    from transformers import BertConfig
    
    config = BertConfig(
        vocab_size=30522,  # Стандартный размер словаря для BERT
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        num_labels=num_classes
    )
    
    # Инициализация модели
    model = BERTLiteClassifier(config)
    
    # Загрузка весов модели
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Получение предсказаний
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Инференс"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=1).cpu().numpy()
            labs = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
    
    # Вычисление метрик
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
    
    # Создание матрицы ошибок
    cm = create_confusion_matrix(all_labels, all_preds, label_encoder.classes_, "BERT-lite")
    
    print(f"BERT-lite F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return f1, report, cm

# Функция для визуализации сравнения моделей
def visualize_model_comparison(lstm_f1, bert_f1, lstm_report, bert_report):
    """
    Визуализирует сравнение моделей по различным метрикам.
    
    Args:
        lstm_f1: F1-score LSTM модели
        bert_f1: F1-score BERT-lite модели
        lstm_report: Отчет о классификации LSTM модели
        bert_report: Отчет о классификации BERT-lite модели
        
    Returns:
        None
    """
    # Парсинг отчетов классификации
    lstm_report_dict = classification_report_to_dict(lstm_report)
    bert_report_dict = classification_report_to_dict(bert_report)
    
    # Сравнение F1-score по эмоциям
    emotions = list(lstm_report_dict.keys())
    if 'accuracy' in emotions:
        emotions.remove('accuracy')
    if 'macro avg' in emotions:
        emotions.remove('macro avg')
    if 'weighted avg' in emotions:
        emotions.remove('weighted avg')
    
    lstm_f1_per_emotion = [lstm_report_dict[emotion]['f1-score'] for emotion in emotions]
    bert_f1_per_emotion = [bert_report_dict[emotion]['f1-score'] for emotion in emotions]
    
    # Создание графика сравнения F1-score по эмоциям
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(emotions))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, lstm_f1_per_emotion, width, label='LSTM')
    bars2 = plt.bar(x + width/2, bert_f1_per_emotion, width, label='BERT-lite')
    
    plt.xlabel('Эмоция')
    plt.ylabel('F1-score')
    plt.title('Сравнение F1-score моделей для разных эмоций')
    plt.xticks(x, emotions)
    plt.legend()
    
    # Добавление значений над столбцами
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig('emotion_comparison.png', dpi=300)
    plt.close()
    
    # Сравнение общего F1-score
    plt.figure(figsize=(10, 6))
    
    plt.bar(['LSTM', 'BERT-lite'], [lstm_f1, bert_f1])
    plt.xlabel('Модель')
    plt.ylabel('Macro F1-score')
    plt.title('Сравнение общего F1-score моделей')
    
    # Добавление значений над столбцами
    plt.text(0, lstm_f1 + 0.01, f'{lstm_f1:.4f}', ha='center', va='bottom')
    plt.text(1, bert_f1 + 0.01, f'{bert_f1:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    
    # Создание сводной таблицы с результатами
    results_df = pd.DataFrame({
        'LSTM': [lstm_report_dict[emotion]['f1-score'] for emotion in emotions],
        'BERT-lite': [bert_report_dict[emotion]['f1-score'] for emotion in emotions],
        'Difference': [bert_report_dict[emotion]['f1-score'] - lstm_report_dict[emotion]['f1-score'] for emotion in emotions]
    }, index=emotions)
    
    # Добавление общих метрик
    results_df.loc['macro avg'] = [
        lstm_report_dict['macro avg']['f1-score'],
        bert_report_dict['macro avg']['f1-score'],
        bert_report_dict['macro avg']['f1-score'] - lstm_report_dict['macro avg']['f1-score']
    ]
    
    results_df.loc['weighted avg'] = [
        lstm_report_dict['weighted avg']['f1-score'],
        bert_report_dict['weighted avg']['f1-score'],
        bert_report_dict['weighted avg']['f1-score'] - lstm_report_dict['weighted avg']['f1-score']
    ]
    
    # Сохранение результатов в CSV
    results_df.to_csv('model_comparison_results.csv')
    
    print("Результаты сравнения сохранены в model_comparison_results.csv")
    
    return results_df

# Вспомогательная функция для преобразования отчета классификации в словарь
def classification_report_to_dict(report):
    """
    Преобразует строковый отчет классификации в словарь.
    
    Args:
        report: Строковый отчет классификации
        
    Returns:
        dict: Словарь с метриками
    """
    lines = report.split('\n')
    report_dict = {}
    
    for line in lines[2:-3]:  # Пропускаем первые две строки (заголовки) и последние три (пустые)
        if not line:
            continue
        line = line.strip()
        if line:
            line_data = line.split()
            if len(line_data) < 5:  # Пропускаем строки с некорректным форматом
                continue
            
            class_name = line_data[0]
            precision = float(line_data[1])
            recall = float(line_data[2])
            f1 = float(line_data[3])
            support = int(line_data[4])
            
            report_dict[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }
    
    # Добавляем средние значения
    for line in lines[-3:]:  # Последние три строки содержат средние значения
        if not line:
            continue
        line = line.strip()
        if line:
            line_data = line.split()
            
            # Проверяем, содержит ли строка информацию о средних значениях
            if 'accuracy' in line:
                accuracy = float(line_data[1])
                report_dict['accuracy'] = accuracy
                continue
                
            if len(line_data) >= 6:  # Строки с avg
                avg_type = line_data[0] + ' ' + line_data[1]  # 'macro avg' или 'weighted avg'
                precision = float(line_data[2])
                recall = float(line_data[3])
                f1 = float(line_data[4])
                support = int(line_data[5])
                
                report_dict[avg_type] = {
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1,
                    'support': support
                }
    
    return report_dict

# Основная функция для оценки и сравнения моделей
def evaluate_and_compare_models(lstm_model_path, bert_model_path):
    """
    Оценивает и сравнивает LSTM и BERT-lite модели.
    
    Args:
        lstm_model_path: Путь к сохраненной LSTM модели
        bert_model_path: Путь к сохраненной BERT-lite модели
        
    Returns:
        pd.DataFrame: Сводная таблица с результатами сравнения
    """
    # Загрузка данных
    _, _, test_data = load_data()
    
    # Оценка LSTM модели
    lstm_f1, lstm_report, _ = evaluate_lstm_model(lstm_model_path, test_data)
    
    # Оценка BERT-lite модели
    bert_f1, bert_report, _ = evaluate_bert_model(bert_model_path, test_data)
    
    # Визуализация сравнения
    results_df = visualize_model_comparison(lstm_f1, bert_f1, lstm_report, bert_report)
    
    return results_df

# Пример использования (не запускается при импорте)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate and compare LSTM and BERT-lite models.')
    parser.add_argument('--lstm_model', type=str, required=True, help='Path to LSTM model')
    parser.add_argument('--bert_model', type=str, required=True, help='Path to BERT-lite model')
    
    args = parser.parse_args()
    
    # Оценка и сравнение моделей
    results_df = evaluate_and_compare_models(args.lstm_model, args.bert_model)
    
    print("\nСводная таблица с результатами сравнения:")
    print(results_df)
