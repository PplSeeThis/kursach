"""
bert_data_preparation.py
Модуль для підготовки даних для BERT-lite моделі
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

def prepare_data_for_bert(train_df, val_df, test_df, tokenizer_name='bert-base-multilingual-cased', 
                          max_length=128, batch_size=32):
    """
    Підготовка даних для BERT-lite моделі
    
    Args:
        train_df: DataFrame з навчальними даними
        val_df: DataFrame з валідаційними даними
        test_df: DataFrame з тестовими даними
        tokenizer_name: Назва токенізатора BERT
        max_length: Максимальна довжина послідовності
        batch_size: Розмір батчу
        
    Returns:
        Кортеж з даталоадерами, словниками та кількістю класів
    """
    # Завантаження токенізатора
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Кодування міток
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['emotion'])
    val_labels = label_encoder.transform(val_df['emotion'])
    test_labels = label_encoder.transform(test_df['emotion'])
    
    # Словники для перетворення індексів в мітки і навпаки
    label2idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    num_classes = len(label2idx)
    
    # Підготовка даних
    train_dataloader = create_bert_dataloader(
        train_df['text'].tolist(), 
        train_labels, 
        tokenizer, 
        max_length, 
        batch_size, 
        is_training=True
    )
    
    val_dataloader = create_bert_dataloader(
        val_df['text'].tolist(), 
        val_labels, 
        tokenizer, 
        max_length, 
        batch_size, 
        is_training=False
    )
    
    test_dataloader = create_bert_dataloader(
        test_df['text'].tolist(), 
        test_labels, 
        tokenizer, 
        max_length, 
        batch_size, 
        is_training=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader, label2idx, idx2label, num_classes

def create_bert_dataloader(texts, labels, tokenizer, max_length, batch_size, is_training=True):
    """
    Створення даталоадера для BERT
    
    Args:
        texts: Список текстів
        labels: Список міток
        tokenizer: Токенізатор BERT
        max_length: Максимальна довжина послідовності
        batch_size: Розмір батчу
        is_training: Чи є це навчальний даталоадер
        
    Returns:
        Даталоадер для BERT
    """
    # Кодування текстів
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,                     
            add_special_tokens=True,  # Додавання [CLS] та [SEP]
            max_length=max_length,    
            padding='max_length',     
            truncation=True,          
            return_attention_mask=True, 
            return_tensors='pt',      
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Конвертація в тензори
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    # Створення датасету
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    # Створення даталоадера
    if is_training:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size
    )
    
    return dataloader

def preprocess_for_bert(text, tokenizer, max_length=128):
    """
    Препроцесинг одного тексту для BERT-lite
    
    Args:
        text: Вхідний текст
        tokenizer: Токенізатор BERT
        max_length: Максимальна довжина послідовності
        
    Returns:
        Тензори з індексами токенів та маскою уваги
    """
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    return encoded_dict['input_ids'], encoded_dict['attention_mask']