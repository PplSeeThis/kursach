"""
emotion_classifier.py
Модуль для класифікації емоцій у тексті
"""

import torch
from transformers import BertTokenizer
import os
import json

# Імпортуємо наші модулі
from lstm_model import LSTMModel, preprocess_for_lstm
from bert_lite_model import BERTLiteModel, preprocess_for_bert

class EmotionClassifier:
    """
    Клас для класифікації емоцій у тексті
    """
    def __init__(self, model_type, model_path, config_path, device='cuda'):
        """
        Ініціалізація класифікатора
        
        Args:
            model_type: Тип моделі ('lstm' або 'bert')
            model_path: Шлях до ваг моделі
            config_path: Шлях до конфігурації моделі
            device: Пристрій для обчислень
        """
        self.model_type = model_type
        self.device = device
        
        # Завантаження конфігурації
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Ініціалізація моделі в залежності від типу
        if model_type == 'lstm':
            self.model = self._init_lstm_model()
            # Підготовка препроцесору
            self.preprocess = lambda text: preprocess_for_lstm(text, self.config['word2idx'])
        elif model_type == 'bert':
            self.model = self._init_bert_model()
            # Ініціалізація токенізатора BERT
            self.tokenizer = BertTokenizer.from_pretrained(self.config['tokenizer'])
            self.preprocess = lambda text: preprocess_for_bert(text, self.tokenizer)
        else:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}")
        
        # Завантаження ваг моделі
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        
        # Словник міток
        self.idx2label = self.config['idx2label']
        if isinstance(self.idx2label, dict):
            # Конвертуємо ключі з рядків у числа (для JSON)
            self.idx2label = {int(k): v for k, v in self.idx2label.items()}
    
    def _init_lstm_model(self):
        """
        Ініціалізація LSTM моделі
        
        Returns:
            LSTM модель
        """
        model = LSTMModel(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim'],
            n_layers=self.config['n_layers'],
            bidirectional=self.config['bidirectional'],
            dropout=self.config['dropout'],
            pad_idx=self.config['pad_idx']
        )
        return model
    
    def _init_bert_model(self):
        """
        Ініціалізація BERT-lite моделі
        
        Returns:
            BERT-lite модель
        """
        model = BERTLiteModel(
            num_classes=self.config['num_classes'],
            hidden_size=self.config['hidden_size'],
            num_hidden_layers=self.config['num_hidden_layers'],
            num_attention_heads=self.config['num_attention_heads'],
            intermediate_size=self.config['intermediate_size'],
            dropout=self.config['dropout']
        )
        return model
    
    def predict(self, text):
        """
        Передбачення емоції для тексту
        
        Args:
            text: Вхідний текст
            
        Returns:
            Кортеж з передбаченою емоцією, ймовірністю та всіма ймовірностями
        """
        # Препроцесинг тексту
        inputs = self.preprocess(text)
        
        # Переміщення входів на пристрій
        inputs = [tensor.to(self.device) for tensor in inputs]
        
        # Передбачення
        with torch.no_grad():
            if self.model_type == 'lstm':
                outputs = self.model(inputs[0], inputs[1])
            else:  # bert
                outputs = self.model(inputs[0], inputs[1])
        
        # Отримання ймовірностей
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Отримання індексу найбільш ймовірного класу
        predicted_idx = probabilities.argmax().item()
        
        # Отримання емоції та ймовірності
        emotion = self.idx2label[predicted_idx]
        probability = probabilities[predicted_idx].item()
        
        return emotion, probability, probabilities.cpu().numpy()
    
    def get_all_emotions(self):
        """
        Отримання списку всіх емоцій
        
        Returns:
            Список емоцій
        """
        return list(set(self.idx2label.values()))

def load_emotion_classifier(model_type, models_dir='models', device='cuda'):
    """
    Завантаження класифікатора емоцій
    
    Args:
        model_type: Тип моделі ('lstm' або 'bert')
        models_dir: Директорія з моделями
        device: Пристрій для обчислень
        
    Returns:
        Класифікатор емоцій
    """
    if model_type == 'lstm':
        model_path = os.path.join(models_dir, 'lstm_emotion_model.pt')
        config_path = os.path.join(models_dir, 'lstm_emotion_config.json')
    elif model_type == 'bert':
        model_path = os.path.join(models_dir, 'bert_emotion_model.pt')
        config_path = os.path.join(models_dir, 'bert_emotion_config.json')
    else:
        raise ValueError(f"Непідтримуваний тип моделі: {model_type}")
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Модель {model_type} не знайдена в директорії {models_dir}")
    
    return EmotionClassifier(model_type, model_path, config_path, device)

if __name__ == "__main__":
    # Приклад використання
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Завантаження моделей
    try:
        lstm_classifier = load_emotion_classifier('lstm', device=device)
        print("LSTM модель завантажено успішно")
    except FileNotFoundError:
        print("LSTM модель не знайдена")
        lstm_classifier = None
    
    try:
        bert_classifier = load_emotion_classifier('bert', device=device)
        print("BERT-lite модель завантажено успішно")
    except FileNotFoundError:
        print("BERT-lite модель не знайдена")
        bert_classifier = None
    
    # Приклади текстів для класифікації
    texts = [
        "Я дуже рада, що ми нарешті зустрілися!",
        "Мені так сумно через цю новину...",
        "Я просто в шоці, не можу повірити, що це сталося!",
        "Ненавиджу цю роботу, вона мене дратує!",
        "Мені так страшно, що буде далі..."
    ]
    
    # Класифікація текстів
    for text in texts:
        print(f"\nТекст: {text}")
        
        if lstm_classifier:
            emotion, prob, _ = lstm_classifier.predict(text)
            print(f"LSTM: {emotion} (ймовірність: {prob:.4f})")
        
        if bert_classifier:
            emotion, prob, _ = bert_classifier.predict(text)
            print(f"BERT-lite: {emotion} (ймовірність: {prob:.4f})")