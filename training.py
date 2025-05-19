"""
training.py
Модуль для навчання та оцінки моделей класифікації емоцій
"""

import torch
import numpy as np
import time
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

def train_lstm_model(model, train_dataloader, val_dataloader, optimizer, criterion, 
                    n_epochs, patience, device='cuda'):
    """
    Навчання LSTM моделі
    
    Args:
        model: Модель LSTM
        train_dataloader: Даталоадер з навчальними даними
        val_dataloader: Даталоадер з валідаційними даними
        optimizer: Оптимізатор
        criterion: Функція втрат
        n_epochs: Максимальна кількість епох
        patience: Кількість епох без покращення для раннього зупинення
        device: Пристрій для обчислень
        
    Returns:
        Кортеж з навченою моделлю та історією навчання
    """
    # Переміщення моделі на потрібний пристрій
    model = model.to(device)
    
    # Списки для відстеження метрик
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    # Параметри для раннього зупинення
    best_val_f1 = 0
    epochs_without_improvement = 0
    best_model_state = None
    
    # Час початку навчання
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Навчання
        model.train()
        epoch_loss = 0
        
        all_preds = []
        all_labels = []
        
        for texts, lengths, labels in train_dataloader:
            # Переміщення даних на потрібний пристрій
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            # Обнулення градієнтів
            optimizer.zero_grad()
            
            # Прямий прохід
            predictions = model(texts, lengths)
            
            # Обчислення втрати
            loss = criterion(predictions, labels)
            
            # Зворотне поширення
            loss.backward()
            
            # Оновлення ваг
            optimizer.step()
            
            # Збереження втрати
            epoch_loss += loss.item()
            
            # Отримання прогнозів
            _, predicted = torch.max(predictions, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Обчислення середньої втрати та F1 для епохи
        train_loss = epoch_loss / len(train_dataloader)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        
        # Валідація
        val_loss, val_f1 = evaluate_lstm_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        # Вивід прогресу
        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        
        # Раннє зупинення
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Загальний час навчання
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    # Завантаження найкращої моделі
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'best_val_f1': best_val_f1,
        'training_time': training_time
    }

def evaluate_lstm_model(model, dataloader, criterion, device='cuda'):
    """
    Оцінка LSTM моделі
    
    Args:
        model: Модель LSTM
        dataloader: Даталоадер з даними
        criterion: Функція втрат
        device: Пристрій для обчислень
        
    Returns:
        Кортеж з середньою втратою та F1-мірою
    """
    # Перехід у режим оцінки
    model.eval()
    
    # Метрики
    val_loss = 0
    all_preds = []
    all_labels = []
    
    # Вимкнення градієнтів
    with torch.no_grad():
        for texts, lengths, labels in dataloader:
            # Переміщення даних на потрібний пристрій
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            # Прямий прохід
            predictions = model(texts, lengths)
            
            # Обчислення втрати
            loss = criterion(predictions, labels)
            val_loss += loss.item()
            
            # Отримання прогнозів
            _, predicted = torch.max(predictions, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Обчислення середньої втрати та F1
    avg_loss = val_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1

def train_bert_lite_model(model, train_dataloader, val_dataloader, optimizer, criterion, 
                          n_epochs, patience, device='cuda'):
    """
    Навчання BERT-lite моделі
    
    Args:
        model: Модель BERT-lite
        train_dataloader: Даталоадер з навчальними даними
        val_dataloader: Даталоадер з валідаційними даними
        optimizer: Оптимізатор
        criterion: Функція втрат
        n_epochs: Максимальна кількість епох
        patience: Кількість епох без покращення для раннього зупинення
        device: Пристрій для обчислень
        
    Returns:
        Кортеж з навченою моделлю та історією навчання
    """
    # Переміщення моделі на потрібний пристрій
    model = model.to(device)
    
    # Створення планувальника швидкості навчання
    total_steps = len(train_dataloader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Списки для відстеження метрик
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    # Параметри для раннього зупинення
    best_val_f1 = 0
    epochs_without_improvement = 0
    best_model_state = None
    
    # Час початку навчання
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Навчання
        model.train()
        epoch_loss = 0
        
        all_preds = []
        all_labels = []
        
        for batch in train_dataloader:
            # Розпакування батчу
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Обнулення градієнтів
            optimizer.zero_grad()
            
            # Прямий прохід
            outputs = model(input_ids, attention_mask)
            
            # Обчислення втрати
            loss = criterion(outputs, labels)
            
            # Зворотне поширення
            loss.backward()
            
            # Обмеження градієнтів (gradient clipping)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Оновлення ваг та планувальника
            optimizer.step()
            scheduler.step()
            
            # Збереження втрати
            epoch_loss += loss.item()
            
            # Отримання прогнозів
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Обчислення середньої втрати та F1 для епохи
        train_loss = epoch_loss / len(train_dataloader)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        
        # Валідація
        val_loss, val_f1 = evaluate_bert_lite_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        # Вивід прогресу
        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        
        # Раннє зупинення
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Загальний час навчання
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    # Завантаження найкращої моделі
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'best_val_f1': best_val_f1,
        'training_time': training_time
    }

def evaluate_bert_lite_model(model, dataloader, criterion, device='cuda'):
    """
    Оцінка BERT-lite моделі
    
    Args:
        model: Модель BERT-lite
        dataloader: Даталоадер з даними
        criterion: Функція втрат
        device: Пристрій для обчислень
        
    Returns:
        Кортеж з середньою втратою та F1-мірою
    """
    # Перехід у режим оцінки
    model.eval()
    
    # Метрики
    val_loss = 0
    all_preds = []
    all_labels = []
    
    # Вимкнення градієнтів
    with torch.no_grad():
        for batch in dataloader:
            # Розпакування даних
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Прямий прохід
            outputs = model(input_ids, attention_mask)
            
            # Обчислення втрати
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Отримання прогнозів
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Обчислення середньої втрати та F1
    avg_loss = val_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1

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