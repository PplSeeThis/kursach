"""
evaluation.py
Модуль для оцінки та візуалізації результатів моделей класифікації емоцій
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
import json

def evaluate_model_detailed(model, dataloader, idx2label, device, model_type='lstm'):
    """
    Детальна оцінка моделі на вказаному датасеті
    
    Args:
        model: Модель (LSTM або BERT-lite)
        dataloader: Даталоадер з даними
        idx2label: Словник відображення індексів у мітки
        device: Пристрій для обчислень
        model_type: Тип моделі ('lstm' або 'bert')
        
    Returns:
        Словник з детальними метриками
    """
    # Перехід у режим оцінки
    model.eval()
    
    # Збереження всіх прогнозів та міток
    all_preds = []
    all_labels = []
    inference_times = []
    
    # Вимкнення градієнтів
    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'lstm':
                # Розпакування даних для LSTM
                texts, lengths, labels = batch
                texts = texts.to(device)
                lengths = lengths.to(device)
                
                # Вимірювання часу інференсу
                start_time = time.time()
                outputs = model(texts, lengths)
                end_time = time.time()
            else:  # model_type == 'bert'
                # Розпакування даних для BERT
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                # Вимірювання часу інференсу
                start_time = time.time()
                outputs = model(input_ids, attention_mask)
                end_time = time.time()
            
            # Обчислення часу інференсу на зразок
            batch_size = labels.size(0)
            inference_time = (end_time - start_time) / batch_size
            inference_times.append(inference_time)
            
            # Отримання прогнозів
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Конвертація індексів у назви класів для звіту
    label_names = [idx2label[idx] for idx in range(len(idx2label))]
    
    # Обчислення метрик
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)
    
    # Обчислення середнього часу інференсу
    avg_inference_time = np.mean(inference_times) * 1000  # конвертація в мс
    
    # Повернення детальних метрик
    return {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'f1_per_class': {label_names[i]: report[label]['f1-score'] for i, label in enumerate(label_names)},
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': report,
        'y_true': all_labels,
        'y_pred': all_preds,
        'inference_time': avg_inference_time  # в мс
    }

def plot_training_history(history, title='Історія навчання'):
    """
    Візуалізація процесу навчання
    
    Args:
        history: Словник з історією навчання
        title: Заголовок графіка
    """
    plt.figure(figsize=(15, 5))
    
    # Графік втрат
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['val_losses'], label='Validation')
    plt.title('Втрати')
    plt.xlabel('Епоха')
    plt.ylabel('Втрата')
    plt.legend()
    plt.grid(True)
    
    # Графік F1-міри
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1s'], label='Train')
    plt.plot(history['val_f1s'], label='Validation')
    plt.title('F1-міра')
    plt.xlabel('Епоха')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title='Матриця помилок'):
    """
    Візуалізація матриці помилок
    
    Args:
        y_true: Справжні мітки
        y_pred: Передбачені мітки
        class_names: Назви класів
        title: Заголовок графіка
    """
    # Обчислення матриці помилок
    cm = confusion_matrix(y_true, y_pred)
    
    # Нормалізація
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Візуалізація
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Передбачені мітки')
    plt.ylabel('Справжні мітки')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()
    
    # Повернення матриць для подальшого аналізу
    return cm, cm_norm

def plot_f1_per_class(y_true, y_pred, class_names, title='F1-міра за класами'):
    """
    Візуалізація F1-міри для кожного класу
    
    Args:
        y_true: Справжні мітки
        y_pred: Передбачені мітки
        class_names: Назви класів
        title: Заголовок графіка
    """
    # Обчислення F1-міри для кожного класу
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Візуалізація
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, f1_per_class)
    plt.title(title)
    plt.xlabel('Клас')
    plt.ylabel('F1-міра')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()
    
    # Повернення значень для подальшого аналізу
    return f1_per_class

def plot_models_comparison(lstm_metrics, bert_metrics, metric_name, class_names=None, title=None):
    """
    Порівняння метрик різних моделей
    
    Args:
        lstm_metrics: Метрики LSTM моделі
        bert_metrics: Метрики BERT-lite моделі
        metric_name: Назва метрики ('accuracy', 'f1_weighted', 'f1_per_class')
        class_names: Назви класів (потрібно для 'f1_per_class')
        title: Заголовок графіка
    """
    if metric_name == 'f1_per_class':
        if class_names is None:
            raise ValueError("class_names повинні бути вказані для метрики 'f1_per_class'")
        
        # Візуалізація F1-міри для кожного класу
        plt.figure(figsize=(14, 6))
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.bar(x - width/2, lstm_metrics, width, label='LSTM')
        plt.bar(x + width/2, bert_metrics, width, label='BERT-lite')
        
        plt.xlabel('Клас')
        plt.ylabel('F1-міра')
        plt.title(title or 'Порівняння F1-міри за класами')
        plt.xticks(x, class_names, rotation=45)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        if title:
            plt.savefig(f'{title.lower().replace(" ", "_")}.png')
        else:
            plt.savefig('f1_per_class_comparison.png')
        plt.show()
    else:
        # Візуалізація загальних метрик
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        lstm_values = [lstm_metrics['accuracy'], lstm_metrics['precision'], 
                      lstm_metrics['recall'], lstm_metrics['f1']]
        bert_values = [bert_metrics['accuracy'], bert_metrics['precision'], 
                      bert_metrics['recall'], bert_metrics['f1']]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, lstm_values, width, label='LSTM')
        plt.bar(x + width/2, bert_values, width, label='BERT-lite')
        
        plt.xlabel('Метрика')
        plt.ylabel('Значення')
        plt.title(title or 'Порівняння загальних метрик')
        plt.xticks(x, metrics)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        if title:
            plt.savefig(f'{title.lower().replace(" ", "_")}.png')
        else:
            plt.savefig('metrics_comparison.png')
        plt.show()

def plot_data_size_comparison(data_sizes, lstm_f1s, bert_f1s, title='Залежність F1-міри від обсягу даних'):
    """
    Візуалізація залежності F1-міри від обсягу навчальних даних
    
    Args:
        data_sizes: Відсотки обсягу даних
        lstm_f1s: F1-міри для LSTM моделі
        bert_f1s: F1-міри для BERT-lite моделі
        title: Заголовок графіка
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, lstm_f1s, 'o-', label='LSTM')
    plt.plot(data_sizes, bert_f1s, 's-', label='BERT-lite')
    plt.xlabel('Відсоток навчальних даних (%)')
    plt.ylabel('F1-міра')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

def save_model(model, model_path, config_path, metadata):
    """
    Збереження моделі та її конфігурації
    
    Args:
        model: Модель для збереження
        model_path: Шлях для збереження ваг моделі
        config_path: Шлях для збереження конфігурації
        metadata: Метадані моделі (словник)
    """
    # Створення директорії, якщо вона не існує
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Збереження ваг моделі
    torch.save(model.state_dict(), model_path)
    
    # Збереження конфігурації
    with open(config_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Модель збережена в {model_path}")
    print(f"Конфігурація збережена в {config_path}")

def load_lstm_model(model_path, config_path, device='cuda'):
    """
    Завантаження LSTM моделі
    
    Args:
        model_path: Шлях до ваг моделі
        config_path: Шлях до конфігурації
        device: Пристрій для обчислень
        
    Returns:
        Завантажена модель та її метадані
    """
    from lstm_model import LSTMModel
    
    # Завантаження конфігурації
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Створення моделі
    model = LSTMModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        n_layers=config['n_layers'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout'],
        pad_idx=config['pad_idx']
    )
    
    # Завантаження ваг
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Переміщення моделі на пристрій
    model = model.to(device)
    
    return model, config

def load_bert_model(model_path, config_path, device='cuda'):
    """
    Завантаження BERT-lite моделі
    
    Args:
        model_path: Шлях до ваг моделі
        config_path: Шлях до конфігурації
        device: Пристрій для обчислень
        
    Returns:
        Завантажена модель та її метадані
    """
    from bert_lite_model import BERTLiteModel
    
    # Завантаження конфігурації
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Створення моделі
    model = BERTLiteModel(
        num_classes=config['num_classes'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        dropout=config['dropout']
    )
    
    # Завантаження ваг
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Переміщення моделі на пристрій
    model = model.to(device)
    
    return model, config

def predict_emotion(text, model, preprocessing_func, idx2label, device='cuda'):
    """
    Передбачення емоції для вхідного тексту
    
    Args:
        text: Вхідний текст
        model: Модель (LSTM або BERT-lite)
        preprocessing_func: Функція препроцесингу
        idx2label: Словник відображення індексів у мітки
        device: Пристрій для обчислень
        
    Returns:
        Кортеж з передбаченою емоцією та ймовірністю
    """
    # Переведення моделі в режим оцінки
    model.eval()
    
    # Препроцесинг тексту
    inputs = preprocessing_func(text)
    
    # Переміщення входів на пристрій
    inputs = [tensor.to(device) for tensor in inputs]
    
    # Передбачення
    with torch.no_grad():
        if len(inputs) == 2:  # LSTM або BERT
            outputs = model(inputs[0], inputs[1])
        else:
            # Цей блок не повинен виконуватися, але для повноти
            outputs = model(inputs[0])
    
    # Отримання ймовірностей
    probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Отримання індексу найбільш ймовірного класу
    predicted_idx = probabilities.argmax().item()
    
    # Отримання емоції та ймовірності
    emotion = idx2label[predicted_idx]
    probability = probabilities[predicted_idx].item()
    
    return emotion, probability, probabilities.cpu().numpy()

def compare_models_on_different_data_sizes(lstm_model, bert_model, 
                                          train_dataset, val_dataloader, test_dataloader,
                                          fractions=[0.1, 0.25, 0.5, 0.75, 1.0],
                                          device='cuda'):
    """
    Порівняння LSTM та BERT-lite моделей на різних обсягах навчальних даних
    
    Args:
        lstm_model: Модель LSTM
        bert_model: Модель BERT-lite
        train_dataset: Тренувальний датасет
        val_dataloader: Валідаційний даталоадер
        test_dataloader: Тестовий даталоадер
        fractions: Частки навчальних даних
        device: Пристрій для обчислень
        
    Returns:
        Словник з результатами порівняння
    """
    from lstm_model import train_lstm_model, evaluate_lstm_model
    from bert_lite_model import train_bert_lite_model, evaluate_bert_lite_model
    
    # Перетворення фракцій у відсотки для графіка
    data_sizes = [fraction * 100 for fraction in fractions]
    
    # Результати
    lstm_results = []
    bert_results = []
    
    # Створення підмножин даних
    subset_sizes = [int(len(train_dataset) * fraction) for fraction in fractions]
    
    for i, size in enumerate(subset_sizes):
        fraction = fractions[i]
        print(f"\nТренування на {fraction*100}% даних ({size} зразків)")
        
        # Створення підмножини даних
        indices = torch.randperm(len(train_dataset))[:size]
        subset = torch.utils.data.Subset(train_dataset, indices)
        subset_dataloader = torch.utils.data.DataLoader(
            subset, 
            batch_size=32,
            shuffle=True
        )
        
        # Навчання LSTM
        print("Навчання LSTM моделі...")
        lstm_copy = type(lstm_model)(
            vocab_size=lstm_model.embedding.num_embeddings,
            embedding_dim=lstm_model.embedding.embedding_dim,
            hidden_dim=lstm_model.lstm.hidden_size,
            output_dim=lstm_model.fc2.out_features,
            n_layers=lstm_model.lstm.num_layers,
            bidirectional=lstm_model.lstm.bidirectional,
            dropout=lstm_model.dropout.p,
            pad_idx=0
        )
        
        # Копіювання початкових ваг
        lstm_copy.load_state_dict(lstm_model.state_dict())
        
        # Навчання
        optimizer = torch.optim.Adam(lstm_copy.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        lstm_copy, lstm_history = train_lstm_model(
            lstm_copy, subset_dataloader, val_dataloader,
            optimizer, criterion, 20, 5, device
        )
        
        # Оцінка LSTM
        val_loss, val_f1 = evaluate_lstm_model(lstm_copy, test_dataloader, criterion, device)
        
        # Навчання BERT-lite
        print("Навчання BERT-lite моделі...")
        bert_copy = type(bert_model)(
            num_classes=bert_model.classifier.out_features,
            hidden_size=bert_model.config.hidden_size,
            num_hidden_layers=bert_model.config.num_hidden_layers,
            num_attention_heads=bert_model.config.num_attention_heads,
            intermediate_size=bert_model.config.intermediate_size,
            dropout=bert_model.dropout.p
        )
        
        # Копіювання початкових ваг
        bert_copy.load_state_dict(bert_model.state_dict())
        
        # Навчання
        optimizer = torch.optim.AdamW(bert_copy.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        bert_copy, bert_history = train_bert_lite_model(
            bert_copy, subset_dataloader, val_dataloader,
            optimizer, criterion, 10, 3, device
        )
        
        # Оцінка BERT-lite
        bert_val_loss, bert_val_f1 = evaluate_bert_lite_model(bert_copy, test_dataloader, criterion, device)
        
        # Збереження результатів
        lstm_results.append(val_f1)
        bert_results.append(bert_val_f1)
        
        print(f"LSTM F1: {val_f1:.4f}, BERT-lite F1: {bert_val_f1:.4f}")
    
    # Повернення результатів
    return {
        'data_sizes': data_sizes,
        'lstm_f1s': lstm_results,
        'bert_f1s': bert_results
    }