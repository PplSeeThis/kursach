import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from tqdm import tqdm

# Загрузка необходимых моделей NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Список стоп-слов для украинского языка
STOP_WORDS = set(stopwords.words('ukrainian'))

# Словарь для замены эмотиконов
EMOTICONS = {
    ':)': 'радість',
    ':(': 'смуток',
    ':D': 'радість',
    ':-D': 'радість',
    ':-(': 'смуток',
    ':-)': 'радість',
    ';)': 'радість',
    ':-P': 'радість',
    ':P': 'радість',
    ':/': 'розчарування',
    ':-/': 'розчарування',
    ':|': 'нейтральний',
    ':-|': 'нейтральний',
    ':O': 'здивування',
    ':-O': 'здивування',
    ':(': 'смуток',
    ':-(': 'смуток',
    '>:(': 'гнів',
    '>:-(': 'гнів',
    ':\'(': 'смуток',
    '♥': 'радість',
    '❤': 'радість',
}

def normalize_text(text):
    """
    Нормализация текста:
    - приведение к нижнему регистру
    - замена эмотиконов
    - удаление URL, HTML тегов, специальных символов, лишних пробелов
    """
    if not isinstance(text, str):
        return ""
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Замена эмотиконов
    for emoticon, replacement in EMOTICONS.items():
        text = text.replace(emoticon, f" {replacement} ")
    
    # Удаление URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Удаление HTML тегов
    text = re.sub(r'<.*?>', ' ', text)
    
    # Удаление специальных символов (оставляем только буквы, цифры и базовую пунктуацию)
    text = re.sub(r'[^\w\s\.,!?;:\(\)-]', ' ', text)
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """
    Токенизация текста с помощью NLTK
    """
    return word_tokenize(text, language='ukrainian')

def remove_stopwords(tokens):
    """
    Удаление стоп-слов из токенизированного текста
    """
    return [token for token in tokens if token.lower() not in STOP_WORDS]

def lemmatize_text(tokens):
    """
    Заглушка для лемматизации украинского текста.
    В реальном проекте здесь следует использовать специализированный лемматизатор для украинского языка.
    """
    # В данной реализации просто возвращаем исходные токены
    # В реальном проекте можно использовать, например, pymorphy2 с украинским словарем
    return tokens

def preprocess_text(text):
    """
    Полный пайплайн предобработки текста
    """
    text = normalize_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return ' '.join(tokens)

def prepare_data(data_path, random_state=42):
    """
    Загрузка и подготовка данных из CSV файла:
    - предобработка текста
    - разделение на тренировочную, валидационную и тестовую выборки
    """
    # Загрузка данных
    df = pd.read_csv(data_path)
    
    # Предобработка текста
    tqdm.pandas(desc="Preprocessing texts")
    df['processed_text'] = df['text'].progress_apply(preprocess_text)
    
    # Определение меток классов
    label_mapping = {
        'радість': 0,
        'смуток': 1,
        'гнів': 2,
        'страх': 3,
        'відраза': 4,
        'здивування': 5,
        'нейтральний': 6
    }
    df['label_id'] = df['emotion'].map(label_mapping)
    
    # Разделение на тренировочную, валидационную и тестовую выборки
    X = df['processed_text']
    y = df['label_id']
    
    # Сначала разделяем на train и temporary (тест + валидация)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    
    # Затем разделяем temporary на validation и test (по 15% каждый)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)
    
    # Проверка распределения классов
    print("Train set class distribution:")
    print(y_train.value_counts().sort_index())
    print("\nValidation set class distribution:")
    print(y_val.value_counts().sort_index())
    print("\nTest set class distribution:")
    print(y_test.value_counts().sort_index())
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_mapping

def plot_class_distribution(y_train, y_val, y_test, label_mapping, save_path=None):
    """
    Построение графика распределения классов в наборах данных
    """
    # Инвертирование label_mapping для получения имен классов
    id_to_label = {v: k for k, v in label_mapping.items()}
    
    # Подсчет экземпляров каждого класса
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    test_counts = Counter(y_test)
    
    # Создание DataFrame для построения графика
    df_counts = pd.DataFrame({
        'Тренировочная выборка': [train_counts.get(i, 0) for i in range(len(id_to_label))],
        'Валидационная выборка': [val_counts.get(i, 0) for i in range(len(id_to_label))],
        'Тестовая выборка': [test_counts.get(i, 0) for i in range(len(id_to_label))]
    }, index=[id_to_label[i] for i in range(len(id_to_label))])
    
    # Построение графика
    plt.figure(figsize=(12, 8))
    df_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Распределение классов в наборах данных', fontsize=14)
    plt.xlabel('Эмоция', fontsize=12)
    plt.ylabel('Количество экземпляров', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_datasets_with_different_sizes(X_train, y_train, sizes=[0.1, 0.25, 0.5, 0.75, 1.0], random_state=42):
    """
    Создание подвыборок разного размера из тренировочного набора данных
    """
    datasets = {}
    
    for size in sizes:
        if size < 1.0:
            X_subset, _, y_subset, _ = train_test_split(
                X_train, y_train, 
                train_size=size, 
                random_state=random_state, 
                stratify=y_train
            )
        else:
            X_subset, y_subset = X_train, y_train
        
        datasets[size] = (X_subset, y_subset)
        print(f"Dataset with {size*100}% of training data created: {len(X_subset)} samples")
    
    return datasets

if __name__ == "__main__":
    # Пример использования (закомментирован, так как файла данных может не быть)
    # X_train, X_val, X_test, y_train, y_val, y_test, label_mapping = prepare_data("emotion_dataset.csv")
    # plot_class_distribution(y_train, y_val, y_test, label_mapping, "class_distribution.png")
    pass
