"""
bert_lite_model.py
Модуль з реалізацією BERT-lite моделі для класифікації емоцій у тексті
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
import time
from sklearn.metrics import f1_score

class BERTLiteModel(nn.Module):
    """
    BERT-lite модель для класифікації емоцій
    """
    def __init__(self, num_classes, hidden_size=512, num_hidden_layers=6, 
                 num_attention_heads=8, intermediate_size=2048, dropout=0.3):
        """
        Ініціалізація моделі
        
        Args:
            num_classes: Кількість класів емоцій
            hidden_size: Розмірність прихованого стану
            num_hidden_layers: Кількість шарів трансформера
            num_attention_heads: Кількість головок уваги
            intermediate_size: Розмір проміжного шару FFN
            dropout: Ймовірність дропауту
        """
        super(BERTLiteModel, self).__init__()
        
        # Створення конфігурації для BERT-lite
        self.config = BertConfig(
            vocab_size=119547,  # Розмір словника для bert-base-multilingual-cased
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        
        # Ініціалізація моделі BERT-lite
        self.bert = BertModel(self.config)
        
        # Класифікаційна надбудова
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Прямий прохід через модель
        
        Args:
            input_ids: Тензор з індексами токенів [batch_size, seq_len]
            attention_mask: Тензор з маскою уваги [batch_size, seq_len]
            
        Returns:
            Тензор з прогнозами класів [batch_size, num_classes]
        """
        # Отримання виходу [CLS] токена з BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Класифікаційна надбудова
        pooled_output = self.dropout(pooled_output)
        hidden = self.fc(pooled_output)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.classifier(hidden)
        
        return output

def create_bert_lite_model(num_classes, pretrained_model_name='bert-base-multilingual-cased', device='cuda'):
    """
    Створення та ініціалізація моделі BERT-lite
    
    Args:
        num_classes: Кількість класів емоцій
        pretrained_model_name: Назва предтренованої моделі
        device: Пристрій для обчислень
        
    Returns:
        Ініціалізована BERT-lite модель
    """
    # Створення моделі
    model = BERTLiteModel(num_classes=num_classes)
    
    # Завантаження ваг предтренованої моделі для ініціалізації
    pretrained_model = BertModel.from_pretrained(pretrained_model_name)
    
    # Ініціалізація параметрів BERT-lite з предтренованої моделі
    # Тут ми копіюємо ваги для перших num_hidden_layers шарів
    for i in range(model.config.num_hidden_layers):
        # Копіювання ваг для шару енкодера, якщо індекс менше кількості шарів предтренованої моделі
        if i < len(pretrained_model.encoder.layer):
            model.bert.encoder.layer[i].load_state_dict(
                pretrained_model.encoder.layer[i].state_dict()
            )
    
    # Копіювання ваг для вбудовувань
    model.bert.embeddings.load_state_dict(
        pretrained_model.embeddings.state_dict()
    )
    
    # Переміщення моделі на потрібний пристрій
    model = model.to(device)
    
    return model

def encode_text_for_bert(texts, tokenizer, max_length=128):
    """
    Кодування тексту для BERT
    
    Args:
        texts: Список текстів
        tokenizer: Токенізатор BERT
        max_length: Максимальна довжина послідовності
        
    Returns:
        Тензори з індексами токенів та масками уваги
    """
    # Кодування тексту
    encoded_texts = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Отримання тензорів
    input_ids = encoded_texts['input_ids']
    attention_masks = encoded_texts['attention_mask']
    
    return input_ids, attention_masks

def prepare_data_for_bert(train_df, val_df, test_df, tokenizer, max_length=128, batch_size=32):
    """
    Підготовка даних для BERT-lite
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        tokenizer: Токенізатор BERT
        max_length: Максимальна довжина послідовності
        batch_size: Розмір батчу
        
    Returns:
        Кортеж з даталоадерами, словниками та кількістю класів
    """
    # Перетворення міток на числові індекси
    label2idx = {label: idx for idx, label in enumerate(train_df['emotion'].unique())}
    idx2label = {idx: label for label, idx in label2idx.items()}
    num_classes = len(label2idx)
    
    train_labels = [label2idx[label] for label in train_df['emotion']]
    val_labels = [label2idx[label] for label in val_df['emotion']]
    test_labels = [label2idx[label] for label in test_df['emotion']]
    
    # Кодування тексту
    train_input_ids, train_attention_masks = encode_text_for_bert(train_df['text'].tolist(), tokenizer, max_length)
    val_input_ids, val_attention_masks = encode_text_for_bert(val_df['text'].tolist(), tokenizer, max_length)
    test_input_ids, test_attention_masks = encode_text_for_bert(test_df['text'].tolist(), tokenizer, max_length)
    
    # Перетворення міток у тензори
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    test_labels = torch.tensor(test_labels)
    
    # Створення датасетів
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    
    # Створення даталоадерів
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )
    
    return train_dataloader, val_dataloader, test_dataloader, label2idx, idx2label, num_classes

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

def preprocess_for_bert(text, tokenizer, max_length=128):
    """
    Препроцесинг тексту для BERT-lite моделі
    
    Args:
        text: Вхідний текст
        tokenizer: Токенізатор BERT
        max_length: Максимальна довжина послідовності
        
    Returns:
        Кортеж з тензорами індексів та маски уваги
    """
    # Кодування тексту
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    return encoded_text['input_ids'], encoded_text['attention_mask']