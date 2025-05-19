"""
lstm_model.py
Модуль з реалізацією LSTM моделі для класифікації емоцій у тексті
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
from sklearn.metrics import f1_score

class TextDataset(Dataset):
    """
    Клас датасету для текстових даних
    """
    def __init__(self, texts, labels, word2idx, max_len=128):
        """
        Ініціалізація датасету
        
        Args:
            texts: Список текстів
            labels: Список міток
            word2idx: Словник відображення слів у індекси
            max_len: Максимальна довжина послідовності
        """
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        
    def __len__(self):
        """
        Отримання кількості елементів у датасеті
        
        Returns:
            Кількість елементів
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Отримання елемента датасету за індексом
        
        Args:
            idx: Індекс елемента
            
        Returns:
            Словник з токенізованим текстом, довжиною та міткою
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Токенізація та перетворення на індекси
        tokens = text.split()
        indices = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        
        # Обрізання або доповнення
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        
        length = len(indices)
        
        # Доповнення паддінгом
        indices = indices + [self.word2idx['<pad>']] * (self.max_len - len(indices))
        
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_batch(batch):
    """
    Функція для об'єднання елементів у батч
    
    Args:
        batch: Список елементів датасету
        
    Returns:
        Кортеж з тензорами тексту, довжини та міток
    """
    text_list = [item['text'] for item in batch]
    length_list = [item['length'] for item in batch]
    label_list = [item['label'] for item in batch]
    
    text = torch.stack(text_list)
    length = torch.stack(length_list)
    label = torch.stack(label_list)
    
    # Сортування за довжиною у спадному порядку
    length, perm_idx = length.sort(0, descending=True)
    text = text[perm_idx]
    label = label[perm_idx]
    
    return text, length, label

def create_dataloaders_for_lstm(train_df, val_df, test_df, min_freq=2, max_len=128, batch_size=32):
    """
    Створення даталоадерів для LSTM моделі
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        min_freq: Мінімальна частота слова для включення в словник
        max_len: Максимальна довжина послідовності
        batch_size: Розмір батчу
        
    Returns:
        Кортеж з трьома даталоадерами, словниками та розміром словника
    """
    # Створення словника
    from preprocessing import VocabBuilder
    
    vocab_builder = VocabBuilder(min_freq=min_freq)
    
    for text in train_df['processed_text']:
        for word in text.split():
            vocab_builder.add_word(word)
    
    word2idx, idx2word = vocab_builder.build_vocab()
    
    # Перетворення міток на числові індекси
    label2idx = {label: idx for idx, label in enumerate(train_df['emotion'].unique())}
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    train_labels = [label2idx[label] for label in train_df['emotion']]
    val_labels = [label2idx[label] for label in val_df['emotion']]
    test_labels = [label2idx[label] for label in test_df['emotion']]
    
    # Створення датасетів
    train_dataset = TextDataset(train_df['processed_text'].tolist(), train_labels, word2idx, max_len)
    val_dataset = TextDataset(val_df['processed_text'].tolist(), val_labels, word2idx, max_len)
    test_dataset = TextDataset(test_df['processed_text'].tolist(), test_labels, word2idx, max_len)
    
    # Створення даталоадерів
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    return train_dataloader, val_dataloader, test_dataloader, word2idx, idx2word, label2idx, idx2label, len(word2idx)

class LSTMModel(nn.Module):
    """
    LSTM модель для класифікації емоцій
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        """
        Ініціалізація моделі
        
        Args:
            vocab_size: Розмір словника
            embedding_dim: Розмірність вбудовування
            hidden_dim: Розмірність прихованого стану LSTM
            output_dim: Кількість класів емоцій
            n_layers: Кількість шарів LSTM
            bidirectional: Чи використовувати двонаправлений LSTM
            dropout: Ймовірність дропауту
            pad_idx: Індекс паддінгу в словнику
        """
        super().__init__()
        
        # Шар вбудовування слів
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Двонаправлений LSTM
        self.lstm = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          dropout=dropout if n_layers > 1 else 0,
                          batch_first=True)
        
        # Визначення розмірності виходу LSTM (x2 для двонаправленого)
        self.lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Global Max Pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Повнозв'язний шар
        self.fc1 = nn.Linear(self.lstm_output_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Вихідний шар
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, text, text_lengths):
        """
        Прямий прохід через модель
        
        Args:
            text: Тензор з індексами слів [batch_size, seq_len]
            text_lengths: Тензор з реальними довжинами послідовностей [batch_size]
            
        Returns:
            Тензор з прогнозами класів [batch_size, output_dim]
        """
        # Вбудовування слів
        embedded = self.embedding(text)  # [batch_size, seq_len, emb_dim]
        
        # Пакування послідовностей для ефективної обробки
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Розпакування послідовностей
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        # Global Max Pooling
        output = output.permute(0, 2, 1)  # [batch_size, hidden_dim, seq_len]
        pooled = self.global_max_pool(output).squeeze(2)  # [batch_size, hidden_dim]
        
        # Повнозв'язний шар
        dense = self.fc1(pooled)
        relu = self.relu(dense)
        dropout = self.dropout(relu)
        
        # Вихідний шар
        output = self.fc2(dropout)
        
        return output

def initialize_lstm_model(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                          bidirectional, dropout, pad_idx=0):
    """
    Ініціалізація LSTM моделі
    
    Args:
        vocab_size: Розмір словника
        embedding_dim: Розмірність вбудовування
        hidden_dim: Розмірність прихованого стану LSTM
        output_dim: Кількість класів емоцій
        n_layers: Кількість шарів LSTM
        bidirectional: Чи використовувати двонаправлений LSTM
        dropout: Ймовірність дропауту
        pad_idx: Індекс паддінгу в словнику
        
    Returns:
        Ініціалізована LSTM модель
    """
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        pad_idx=pad_idx
    )
    
    # Ініціалізація ваг
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model

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

def preprocess_for_lstm(text, word2idx, max_length=128):
    """
    Препроцесинг тексту для LSTM моделі
    
    Args:
        text: Вхідний текст
        word2idx: Словник відображення слів у індекси
        max_length: Максимальна довжина послідовності
        
    Returns:
        Кортеж з тензором індексів та довжиною
    """
    from preprocessing import preprocess_text
    
    # Препроцесинг тексту
    processed_text = preprocess_text(text)
    
    # Токенізація та перетворення на індекси
    tokens = processed_text.split()
    indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    
    # Обмеження довжини
    if len(indices) > max_length:
        indices = indices[:max_length]
    
    length = len(indices)
    
    # Доповнення паддінгом
    indices = indices + [word2idx['<pad>']] * (max_length - len(indices))
    
    return torch.tensor(indices).unsqueeze(0), torch.tensor([length])