"""
preprocessing.py
Модуль для препроцесингу тексту для класифікації емоцій
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch

# Налаштування відтворюваності
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Завантаження стоп-слів та лематизатора
nltk.download('stopwords')
nltk.download('wordnet')

# Створення списку українських стоп-слів
try:
    stop_words = set(stopwords.words('ukrainian'))
except OSError:
    # Если стоп-слова недоступны, используем наш собственный список
    from ukrainian_stopwords import UKRAINIAN_STOPWORDS
    stop_words = set(UKRAINIAN_STOPWORDS)

def normalize_text(text):
    """
    Нормалізація тексту: приведення до нижнього регістру, 
    заміна емотиконів, видалення URL, HTML-тегів, спеціальних символів
    
    Args:
        text: Текст для нормалізації
        
    Returns:
        Нормалізований текст
    """
    # Перевірка на None або порожній рядок
    if text is None or text == '':
        return ''
    
    # Приведення до нижнього регістру
    text = text.lower()
    
    # Заміна емотиконів на текстові мітки
    text = re.sub(r':\)', ' емоція_радість ', text)
    text = re.sub(r':\(', ' емоція_смуток ', text)
    text = re.sub(r':D', ' емоція_радість ', text)
    text = re.sub(r':\|', ' емоція_нейтральний ', text)
    
    # Видалення URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Видалення HTML-тегів
    text = re.sub(r'<.*?>', '', text)
    
    # Видалення спеціальних символів та цифр
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Видалення зайвих пробілів
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_and_remove_stopwords(text):
    """
    Токенізація тексту та видалення стоп-слів
    
    Args:
        text: Текст для обробки
        
    Returns:
        Список токенів без стоп-слів
    """
    # Токенізація
    tokens = text.split()
    
    # Видалення стоп-слів
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def lemmatize_text(tokens):
    """
    Лематизація токенів
    
    Args:
        tokens: Список токенів
        
    Returns:
        Текст з лематизованими токенами
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def preprocess_text(text):
    """
    Повний процес препроцесингу тексту
    
    Args:
        text: Текст для обробки
        
    Returns:
        Оброблений текст
    """
    # Нормалізація тексту
    normalized_text = normalize_text(text)
    
    # Токенізація та видалення стоп-слів
    tokens = tokenize_and_remove_stopwords(normalized_text)
    
    # Лематизація
    lemmatized_text = lemmatize_text(tokens)
    
    return lemmatized_text

def load_and_preprocess_data(file_path):
    """
    Завантаження та препроцесинг даних
    
    Args:
        file_path: Шлях до CSV файлу з даними
        
    Returns:
        Кортеж з трьома DataFrame: train_df, val_df, test_df
    """
    # Завантаження даних
    df = pd.read_csv(file_path)
    
    # Перевірка наявності необхідних стовпців
    if 'text' not in df.columns or 'emotion' not in df.columns:
        raise ValueError("CSV файл повинен містити стовпці 'text' та 'emotion'")
    
    # Препроцесинг тексту
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Розподіл на навчальну, валідаційну та тестову вибірки
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['emotion'], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=SEED)
    
    print(f"Розмір навчальної вибірки: {len(train_df)}")
    print(f"Розмір валідаційної вибірки: {len(val_df)}")
    print(f"Розмір тестової вибірки: {len(test_df)}")
    
    return train_df, val_df, test_df

class VocabBuilder:
    """
    Клас для побудови словника з текстових даних
    """
    def __init__(self, min_freq=2):
        """
        Ініціалізація побудовника словника
        
        Args:
            min_freq: Мінімальна частота слова для включення в словник
        """
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.frequencies = {}
        self.min_freq = min_freq
        
    def add_word(self, word):
        """
        Додавання слова до словника та підрахунок частоти
        
        Args:
            word: Слово для додавання
        """
        if word not in self.frequencies:
            self.frequencies[word] = 1
        else:
            self.frequencies[word] += 1
            
    def build_vocab(self):
        """
        Побудова словника на основі частоти слів
        
        Returns:
            Кортеж з двох словників: word2idx та idx2word
        """
        idx = 2  # Починаємо з 2 (0 і 1 зарезервовані для pad і unk)
        for word, freq in self.frequencies.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        return self.word2idx, self.idx2word

def build_vocab(texts, min_freq=2):
    """
    Побудова словника з текстів
    
    Args:
        texts: Список текстів
        min_freq: Мінімальна частота слова для включення в словник
        
    Returns:
        Словник відображення слів у індекси
    """
    vocab_builder = VocabBuilder(min_freq=min_freq)
    
    for text in texts:
        for word in text.split():
            vocab_builder.add_word(word)
    
    word2idx, _ = vocab_builder.build_vocab()
    return word2idx
