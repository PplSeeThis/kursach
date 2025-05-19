import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_lstm_model(model, train_iterator, val_iterator, optimizer, criterion, n_epochs, device, scheduler=None, patience=5, model_save_path='best_lstm_model.pt'):
    """
    Обучение LSTM модели
    
    Args:
        model: LSTM модель
        train_iterator: даталоадер с тренировочными данными
        val_iterator: даталоадер с валидационными данными
        optimizer: оптимизатор
        criterion: функция потерь
        n_epochs: количество эпох
        device: устройство (cpu или cuda)
        scheduler: планировщик скорости обучения (опционально)
        patience: количество эпох без улучшения для раннего останова
        model_save_path: путь для сохранения лучшей модели
        
    Returns:
        history: словарь с историей обучения
        best_model: лучшая модель
        training_time: время обучения
    """
    # Инициализация переменных для отслеживания прогресса
    best_val_f1 = 0
    epochs_without_improvement = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    # Перемещение модели на указанное устройство
    model = model.to(device)
    
    # Замер времени начала обучения
    start_time = time.time()
    
    # Основной цикл обучения
    for epoch in range(n_epochs):
        # Режим обучения
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        # Прогресс-бар для отслеживания обучения
        train_bar = tqdm(train_iterator, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        
        for batch in train_bar:
            # Извлечение данных из батча
            text = batch['text'].to(device)
            text_lengths = batch['length'].to(device)
            labels = batch['label'].to(device)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход
            predictions = model(text, text_lengths)
            
            # Вычисление потерь
            loss = criterion(predictions, labels)
            
            # Обратное распространение ошибки
            loss.backward()
            
            # Обновление весов
            optimizer.step()
            
            # Накопление потерь
            epoch_loss += loss.item()
            
            # Сохранение предсказаний и истинных меток для расчета метрик
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Обновление прогресс-бара
            train_bar.set_postfix({'loss': loss.item()})
        
        # Вычисление средних потерь и метрик для эпохи
        train_loss = epoch_loss / len(train_iterator)
        train_acc = accuracy_score(all_labels, all_preds)
        _, _, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        
        # Сохранение метрик обучения
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        
        # Валидация
        val_loss, val_acc, val_f1 = evaluate_lstm_model(model, val_iterator, criterion, device)
        
        # Сохранение метрик валидации
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Обновление планировщика скорости обучения
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Вывод результатов эпохи
        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'\tVal Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
        
        # Сохранение лучшей модели
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Ранний останов
        if epochs_without_improvement >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Замер времени окончания обучения
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load(model_save_path))
    
    return history, model, training_time

def evaluate_lstm_model(model, iterator, criterion, device):
    """
    Оценка LSTM модели
    
    Args:
        model: LSTM модель
        iterator: даталоадер с данными для оценки
        criterion: функция потерь
        device: устройство (cpu или cuda)
        
    Returns:
        val_loss: средние потери
        val_acc: точность
        val_f1: F1-мера
    """
    # Режим оценки
    model.eval()
    
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    # Отключение вычисления градиентов для ускорения оценки
    with torch.no_grad():
        # Прогресс-бар для отслеживания оценки
        val_bar = tqdm(iterator, desc='Evaluating')
        
        for batch in val_bar:
            # Извлечение данных из батча
            text = batch['text'].to(device)
            text_lengths = batch['length'].to(device)
            labels = batch['label'].to(device)
            
            # Прямой проход
            predictions = model(text, text_lengths)
            
            # Вычисление потерь
            loss = criterion(predictions, labels)
            
            # Накопление потерь
            epoch_loss += loss.item()
            
            # Сохранение предсказаний и истинных меток для расчета метрик
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Обновление прогресс-бара
            val_bar.set_postfix({'loss': loss.item()})
    
    # Вычисление средних потерь и метрик
    val_loss = epoch_loss / len(iterator)
    val_acc = accuracy_score(all_labels, all_preds)
    _, _, val_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    return val_loss, val_acc, val_f1

def train_bert_model(model, train_iterator, val_iterator, optimizer, criterion, n_epochs, device, scheduler=None, patience=3, model_save_path='best_bert_model.pt'):
    """
    Обучение BERT-lite модели
    
    Args:
        model: BERT-lite модель
        train_iterator: даталоадер с тренировочными данными
        val_iterator: даталоадер с валидационными данными
        optimizer: оптимизатор
        criterion: функция потерь
        n_epochs: количество эпох
        device: устройство (cpu или cuda)
        scheduler: планировщик скорости обучения (опционально)
        patience: количество эпох без улучшения для раннего останова
        model_save_path: путь для сохранения лучшей модели
        
    Returns:
        history: словарь с историей обучения
        best_model: лучшая модель
        training_time: время обучения
    """
    # Инициализация переменных для отслеживания прогресса
    best_val_f1 = 0
    epochs_without_improvement = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    # Перемещение модели на указанное устройство
    model = model.to(device)
    
    # Замер времени начала обучения
    start_time = time.time()
    
    # Основной цикл обучения
    for epoch in range(n_epochs):
        # Режим обучения
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        # Прогресс-бар для отслеживания обучения
        train_bar = tqdm(train_iterator, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        
        for batch in train_bar:
            # Извлечение данных из батча
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход
            predictions = model(input_ids, attention_mask)
            
            # Вычисление потерь
            loss = criterion(predictions, labels)
            
            # Обратное распространение ошибки
            loss.backward()
            
            # Обновление весов
            optimizer.step()
            
            # Накопление потерь
            epoch_loss += loss.item()
            
            # Сохранение предсказаний и истинных меток для расчета метрик
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Обновление прогресс-бара
            train_bar.set_postfix({'loss': loss.item()})
        
        # Вычисление средних потерь и метрик для эпохи
        train_loss = epoch_loss / len(train_iterator)
        train_acc = accuracy_score(all_labels, all_preds)
        _, _, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        
        # Сохранение метрик обучения
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        
        # Валидация
        val_loss, val_acc, val_f1 = evaluate_bert_model(model, val_iterator, criterion, device)
        
        # Сохранение метрик валидации
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Обновление планировщика скорости обучения
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Вывод результатов эпохи
        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'\tVal Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
        
        # Сохранение лучшей модели
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Ранний останов
        if epochs_without_improvement >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Замер времени окончания обучения
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load(model_save_path))
    
    return history, model, training_time

def evaluate_bert_model(model, iterator, criterion, device):
    """
    Оценка BERT-lite модели
    
    Args:
        model: BERT-lite модель
        iterator: даталоадер с данными для оценки
        criterion: функция потерь
        device: устройство (cpu или cuda)
        
    Returns:
        val_loss: средние потери
        val_acc: точность
        val_f1: F1-мера
    """
    # Режим оценки
    model.eval()
    
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    # Отключение вычисления градиентов для ускорения оценки
    with torch.no_grad():
        # Прогресс-бар для отслеживания оценки
        val_bar = tqdm(iterator, desc='Evaluating')
        
        for batch in val_bar:
            # Извлечение данных из батча
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Прямой проход
            predictions = model(input_ids, attention_mask)
            
            # Вычисление потерь
            loss = criterion(predictions, labels)
            
            # Накопление потерь
            epoch_loss += loss.item()
            
            # Сохранение предсказаний и истинных меток для расчета метрик
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Обновление прогресс-бара
            val_bar.set_postfix({'loss': loss.item()})
    
    # Вычисление средних потерь и метрик
    val_loss = epoch_loss / len(iterator)
    val_acc = accuracy_score(all_labels, all_preds)
    _, _, val_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    return val_loss, val_acc, val_f1

def test_model(model, test_iterator, criterion, device, model_type='lstm'):
    """
    Тестирование модели
    
    Args:
        model: модель (LSTM или BERT-lite)
        test_iterator: даталоадер с тестовыми данными
        criterion: функция потерь
        device: устройство (cpu или cuda)
        model_type: тип модели ('lstm' или 'bert')
        
    Returns:
        test_loss: средние потери
        test_acc: точность
        test_f1_macro: F1-мера (macro)
        test_f1_weighted: F1-мера (weighted)
        precision: точность по классам
        recall: полнота по классам
        conf_matrix: матрица ошибок
    """
    # Режим оценки
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Замер времени начала тестирования
    start_time = time.time()
    
    # Отключение вычисления градиентов для ускорения тестирования
    with torch.no_grad():
        # Прогресс-бар для отслеживания тестирования
        test_bar = tqdm(test_iterator, desc='Testing')
        
        for batch in test_bar:
            # Извлечение данных из батча в зависимости от типа модели
            if model_type == 'lstm':
                text = batch['text'].to(device)
                text_lengths = batch['length'].to(device)
                labels = batch['label'].to(device)
                
                # Прямой проход
                predictions = model(text, text_lengths)
            else:  # bert
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Прямой проход
                predictions = model(input_ids, attention_mask)
            
            # Сохранение предсказаний и истинных меток для расчета метрик
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Замер времени окончания тестирования
    inference_time = time.time() - start_time
    inference_time_per_sample = inference_time / len(all_labels)
    
    # Вычисление метрик
    test_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    test_f1_macro = np.mean(f1_score)
    test_f1_weighted = precision_recall_fscore_support(all_labels, all_preds, average='weighted')[2]
    
    # Вычисление матрицы ошибок
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Вывод результатов
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 (Macro): {test_f1_macro:.4f}')
    print(f'Test F1 (Weighted): {test_f1_weighted:.4f}')
    print(f'Inference time per sample: {inference_time_per_sample*1000:.2f} ms')
    
    return test_acc, test_f1_macro, test_f1_weighted, precision, recall, conf_matrix, inference_time_per_sample

def plot_training_curves(history, save_path=None):
    """
    Построение графиков обучения
    
    Args:
        history: словарь с историей обучения
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(16, 6))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix, class_names, model_name, normalize=True, save_path=None):
    """
    Построение матрицы ошибок
    
    Args:
        conf_matrix: матрица ошибок
        class_names: названия классов
        model_name: название модели
        normalize: нормализовать ли значения
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        # Нормализация по строкам (истинные метки)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {model_name}'
    
    # Создание тепловой карты
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_f1_scores_by_emotion(precision, recall, f1, class_names, model_name, save_path=None):
    """
    Построение графика F1-мер для каждой эмоции
    
    Args:
        precision: точность по классам
        recall: полнота по классам
        f1: F1-мера по классам
        class_names: названия классов
        model_name: название модели
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1')
    
    plt.title(f'Metrics by Emotion - {model_name}', fontsize=14)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, class_names, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_models_f1(f1_scores_lstm, f1_scores_bert, class_names, save_path=None):
    """
    Сравнение F1-мер двух моделей для каждой эмоции
    
    Args:
        f1_scores_lstm: F1-мера LSTM модели по классам
        f1_scores_bert: F1-мера BERT-lite модели по классам
        class_names: названия классов
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, f1_scores_lstm, width, label='LSTM')
    plt.bar(x + width/2, f1_scores_bert, width, label='BERT-lite')
    
    plt.title('F1 Scores Comparison by Emotion', fontsize=14)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(x, class_names, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_data_size_comparison(data_sizes, lstm_f1_scores, bert_f1_scores, save_path=None):
    """
    Построение графика зависимости F1-меры от размера обучающей выборки
    
    Args:
        data_sizes: размеры обучающей выборки (в процентах)
        lstm_f1_scores: F1-меры LSTM модели для разных размеров выборки
        bert_f1_scores: F1-меры BERT-lite модели для разных размеров выборки
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(data_sizes, lstm_f1_scores, 'o-', label='LSTM', linewidth=2)
    plt.plot(data_sizes, bert_f1_scores, 's-', label='BERT-lite', linewidth=2)
    
    plt.title('F1 Score vs. Training Data Size', fontsize=14)
    plt.xlabel('Training Data Size (%)', fontsize=12)
    plt.ylabel('F1 Score (Macro)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Добавление подписей значений
    for i, (x, y1, y2) in enumerate(zip(data_sizes, lstm_f1_scores, bert_f1_scores)):
        plt.text(x, y1 - 0.02, f'{y1:.3f}', ha='center', va='top', fontsize=10)
        plt.text(x, y2 + 0.01, f'{y2:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Пример использования (закомментирован)
    pass
