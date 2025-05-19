import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class TextDataset(Dataset):
    """Датасет для работы с текстовыми данными для LSTM модели"""
    
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        # Токенизация и преобразование в индексы
        tokens = text.split()  # Предполагается, что текст уже предобработан
        tokens = tokens[:self.max_len]  # Обрезаем до максимальной длины
        
        # Преобразование токенов в индексы
        indexes = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        
        # Дополнение последовательности до максимальной длины
        padding_length = self.max_len - len(indexes)
        if padding_length > 0:
            indexes = indexes + [self.vocab["<PAD>"]] * padding_length
        
        return {
            'text': torch.tensor(indexes, dtype=torch.long),
            'length': torch.tensor(min(len(tokens), self.max_len), dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertDataset(Dataset):
    """Датасет для работы с текстовыми данными для BERT модели"""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        # Токенизация BERT
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def build_vocab(texts, max_vocab_size=50000):
    """
    Построение словаря для LSTM модели
    """
    # Счетчик всех слов в текстах
    word_counts = {}
    for text in tqdm(texts, desc="Building vocabulary"):
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Сортировка по частоте
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Создание словаря
    vocab = {"<PAD>": 0, "<UNK>": 1}  # Специальные токены
    for i, (word, _) in enumerate(sorted_words[:max_vocab_size-2]):
        vocab[word] = i + 2  # +2 из-за <PAD> и <UNK>
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def prepare_data_for_lstm(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64, max_vocab_size=50000, max_len=128):
    """
    Подготовка данных для LSTM модели
    """
    # Построение словаря
    vocab = build_vocab(X_train, max_vocab_size)
    
    # Создание датасетов
    train_dataset = TextDataset(X_train, y_train, vocab, max_len)
    val_dataset = TextDataset(X_val, y_val, vocab, max_len)
    test_dataset = TextDataset(X_test, y_test, vocab, max_len)
    
    # Создание даталоадеров
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, vocab

def prepare_data_for_bert(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32, max_len=128, model_name="bert-base-multilingual-cased"):
    """
    Подготовка данных для BERT модели
    """
    # Загрузка токенизатора
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Создание датасетов
    train_dataset = BertDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = BertDataset(X_val, y_val, tokenizer, max_len)
    test_dataset = BertDataset(X_test, y_test, tokenizer, max_len)
    
    # Создание даталоадеров
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, tokenizer

if __name__ == "__main__":
    # Примеры использования (закомментированы)
    """
    # Пример подготовки данных для LSTM
    train_loader, val_loader, test_loader, vocab = prepare_data_for_lstm(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Пример подготовки данных для BERT
    train_loader, val_loader, test_loader, tokenizer = prepare_data_for_bert(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    """
    pass
