#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стилей для графиков
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

class EmotionClassifier:
    """
    Класс для классификации эмоций в тексте с использованием LSTM или BERT-lite моделей.
    """
    
    def __init__(self, model_type, model_path, preprocessing_func, device=None):
        """
        Инициализирует классификатор эмоций.
        
        Args:
            model_type: Тип модели ('LSTM' или 'BERT-lite')
            model_path: Путь к сохраненной модели
            preprocessing_func: Функция предобработки текста
            device: Устройство для инференса (если None, будет выбрано автоматически)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.preprocessing_func = preprocessing_func
        
        # Определяем устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Используется устройство: {self.device}")
        
        # Загружаем модель
        self._load_model()
    
    def _load_model(self):
        """
        Загружает модель в зависимости от указанного типа.
        """
        if self.model_type == 'LSTM':
            from lstm_model import LSTMClassifier
            from preprocessing import create_vocab
            
            # Загружаем словарь и метки
            self.TEXT, self.LABEL = create_vocab()
            
            # Параметры модели
            INPUT_DIM = len(self.TEXT.vocab)
            EMBEDDING_DIM = 300
            HIDDEN_DIM = 256
            OUTPUT_DIM = len(self.LABEL.vocab)
            N_LAYERS = 2
            BIDIRECTIONAL = True
            DROPOUT = 0.3
            PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]
            
            # Инициализация модели
            self.model = LSTMClassifier(
                INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX
            )
            
            # Загрузка весов
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
            # Эмоции
            self.emotion_labels = self.LABEL.vocab.itos
            
        elif self.model_type == 'BERT-lite':
            from bert_lite_model import BERTLiteClassifier
            from transformers import BertConfig
            from bert_data_preparation import get_tokenizer, get_label_encoder
            
            # Загружаем токенизатор и кодировщик меток
            self.tokenizer = get_tokenizer()
            self.label_encoder = get_label_encoder()
            
            # Параметры модели
            config = BertConfig(
                vocab_size=30522,
                hidden_size=512,
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=2048,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                num_labels=len(self.label_encoder.classes_)
            )
            
            # Инициализация модели
            self.model = BERTLiteClassifier(config)
            
            # Загрузка весов
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
            # Эмоции
            self.emotion_labels = self.label_encoder.classes_
        
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}. Поддерживаются только 'LSTM' и 'BERT-lite'.")
        
        # Переводим модель в режим оценки и перемещаем на устройство
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print(f"Модель {self.model_type} успешно загружена.")
    
    def classify(self, text, return_probs=False):
        """
        Классифицирует текст по эмоциям.
        
        Args:
            text: Текст для классификации
            return_probs: Возвращать вероятности по всем классам (True) или только метку (False)
            
        Returns:
            str или tuple: Предсказанная эмоция или кортеж (эмоция, вероятности)
        """
        if self.model_type == 'LSTM':
            # Предобработка текста для LSTM
            processed_text = self.preprocessing_func(text)
            
            # Преобразование в тензор
            indexed = [self.TEXT.vocab.stoi[t] for t in processed_text]
            length = len(indexed)
            
            # Добавляем паддинг, если текст слишком короткий
            if length < 5:
                indexed.extend([self.TEXT.vocab.stoi[self.TEXT.pad_token]] * (5 - length))
                length = 5
            
            tensor = torch.LongTensor(indexed).unsqueeze(1).to(self.device)
            tensor_length = torch.LongTensor([length]).to(self.device)
            
            # Инференс
            with torch.no_grad():
                predictions = self.model(tensor, tensor_length).squeeze(1)
                probs = torch.softmax(predictions, dim=1).cpu().numpy()[0]
                pred_class = predictions.argmax(dim=1).item()
            
            # Получение предсказанной эмоции
            predicted_emotion = self.emotion_labels[pred_class]
            
        elif self.model_type == 'BERT-lite':
            # Предобработка текста для BERT-lite
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Перемещение на устройство
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Инференс
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class = logits.argmax(dim=1).item()
            
            # Получение предсказанной эмоции
            predicted_emotion = self.emotion_labels[pred_class]
        
        # Возвращаем результат
        if return_probs:
            return predicted_emotion, probs
        else:
            return predicted_emotion
    
    def classify_batch(self, texts, batch_size=32):
        """
        Классифицирует пакет текстов.
        
        Args:
            texts: Список текстов для классификации
            batch_size: Размер батча
            
        Returns:
            list: Список предсказанных эмоций
        """
        results = []
        
        # Обрабатываем тексты батчами
        for i in tqdm(range(0, len(texts), batch_size), desc="Классификация текстов"):
            batch_texts = texts[i:i+batch_size]
            batch_results = [self.classify(text) for text in batch_texts]
            results.extend(batch_results)
        
        return results
    
    def visualize_emotions(self, text):
        """
        Визуализирует вероятности эмоций для данного текста.
        
        Args:
            text: Текст для классификации
            
        Returns:
            None
        """
        # Получаем предсказание и вероятности
        emotion, probs = self.classify(text, return_probs=True)
        
        # Сортируем эмоции по вероятности
        sorted_indices = np.argsort(probs)[::-1]
        sorted_emotions = [self.emotion_labels[i] for i in sorted_indices]
        sorted_probs = [probs[i] for i in sorted_indices]
        
        # Создаем горизонтальный бар-плот
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_emotions, sorted_probs)
        plt.xlabel('Вероятность')
        plt.ylabel('Эмоция')
        plt.title(f'Распределение вероятностей эмоций для текста:\n"{text}"')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show()
        
        print(f"Предсказанная эмоция: {emotion} с вероятностью {probs[self.emotion_labels.index(emotion)]:.4f}")

def preprocess_text_for_lstm(text):
    """
    Предобработка текста для LSTM модели.
    
    Args:
        text: Текст для предобработки
        
    Returns:
        list: Список токенов
    """
    from preprocessing import preprocess_text, tokenize_text
    
    # Применяем предобработку текста
    processed_text = preprocess_text(text)
    
    # Токенизируем текст
    tokens = tokenize_text(processed_text)
    
    return tokens

def preprocess_text_for_bert(text):
    """
    Предобработка текста для BERT-lite модели.
    
    Args:
        text: Текст для предобработки
        
    Returns:
        str: Предобработанный текст
    """
    from preprocessing import preprocess_text
    
    # Применяем только базовую предобработку
    processed_text = preprocess_text(text)
    
    return processed_text

def load_emotion_classifier(model_type='LSTM', model_path=None):
    """
    Загружает классификатор эмоций.
    
    Args:
        model_type: Тип модели ('LSTM' или 'BERT-lite')
        model_path: Путь к сохраненной модели (если None, будет использован путь по умолчанию)
        
    Returns:
        EmotionClassifier: Загруженный классификатор эмоций
    """
    # Определяем пути к моделям по умолчанию
    default_paths = {
        'LSTM': 'models/best_lstm_model.pt',
        'BERT-lite': 'models/best_bert_model.pt'
    }
    
    # Используем путь по умолчанию, если не указан
    if model_path is None:
        model_path = default_paths[model_type]
    
    # Определяем функцию предобработки в зависимости от типа модели
    if model_type == 'LSTM':
        preprocessing_func = preprocess_text_for_lstm
    else:  # BERT-lite
        preprocessing_func = preprocess_text_for_bert
    
    # Создаем и возвращаем классификатор
    return EmotionClassifier(model_type, model_path, preprocessing_func)

# Пример использования
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify emotions in text using LSTM or BERT-lite models.')
    parser.add_argument('text', type=str, help='Text to classify')
    parser.add_argument('--model', type=str, choices=['LSTM', 'BERT-lite'], default='BERT-lite',
                        help='Model type to use (default: BERT-lite)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model (default: models/best_lstm_model.pt or models/best_bert_model.pt)')
    parser.add_argument('--visualize', action='store_true', help='Visualize emotion probabilities')
    
    args = parser.parse_args()
    
    # Загружаем классификатор
    classifier = load_emotion_classifier(args.model, args.model_path)
    
    # Классифицируем текст
    if args.visualize:
        classifier.visualize_emotions(args.text)
    else:
        emotion = classifier.classify(args.text)
        print(f"Предсказанная эмоция: {emotion}")
