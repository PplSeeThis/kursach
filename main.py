"""
main.py
Основний файл для класифікації емоцій у тексті за допомогою LSTM та BERT-lite
"""
import os
import warnings

# Подавление предупреждений TensorFlow/CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Подавляет все сообщения TensorFlow
warnings.filterwarnings('ignore')  # Подавляет прочие предупреждения Python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
import argparse
import os
import matplotlib.pyplot as plt

# Імпортуємо наші модулі
from preprocessing import load_and_preprocess_data
from lstm_model import create_dataloaders_for_lstm, initialize_lstm_model, train_lstm_model, preprocess_for_lstm
from bert_lite_model import create_bert_lite_model, prepare_data_for_bert, train_bert_lite_model, preprocess_for_bert
from evaluation import (evaluate_model_detailed, plot_training_history, plot_confusion_matrix, plot_f1_per_class,
                      plot_models_comparison, plot_data_size_comparison, save_model, predict_emotion)

def main():
    """
    Основна функція для тренування та оцінки моделей класифікації емоцій
    """
    # Парсинг аргументів командного рядка
    parser = argparse.ArgumentParser(description='Класифікація емоцій у тексті')
    parser.add_argument('--data_path', type=str, default='emotions_dataset.csv', help='Шлях до файлу з даними')
    parser.add_argument('--train_lstm', action='store_true', help='Тренувати LSTM модель')
    parser.add_argument('--train_bert', action='store_true', help='Тренувати BERT-lite модель')
    parser.add_argument('--compare', action='store_true', help='Порівняти моделі')
    parser.add_argument('--predict', type=str, default=None, help='Передбачити емоцію для вказаного тексту')
    parser.add_argument('--save_dir', type=str, default='models', help='Директорія для збереження моделей')
    args = parser.parse_args()
    
    # Створення директорії для збереження моделей
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Визначення пристрою для обчислень
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Використовується пристрій: {device}")
    
    # Завантаження та препроцесинг даних
    if args.train_lstm or args.train_bert or args.compare:
        print("Завантаження та препроцесинг даних...")
        train_df, val_df, test_df = load_and_preprocess_data(args.data_path)
    
    # Тренування та оцінка LSTM моделі
    if args.train_lstm:
        print("Підготовка даних для LSTM...")
        train_dataloader_lstm, val_dataloader_lstm, test_dataloader_lstm, \
        word2idx, idx2word, label2idx, idx2label, vocab_size = create_dataloaders_for_lstm(
            train_df, val_df, test_df
        )
        
        print("Ініціалізація LSTM моделі...")
        lstm_model = initialize_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            output_dim=len(label2idx),
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        print("Тренування LSTM моделі...")
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        lstm_model, lstm_history = train_lstm_model(
            lstm_model, train_dataloader_lstm, val_dataloader_lstm,
            optimizer, criterion, 20, 5, device
        )
        
        # Візуалізація історії навчання
        plot_training_history(lstm_history, 'Історія навчання LSTM')
        
        # Оцінка LSTM моделі
        print("Оцінка LSTM моделі...")
        lstm_metrics = evaluate_model_detailed(
            lstm_model, test_dataloader_lstm, idx2label, device, 'lstm'
        )
        
        print("\nРезультати для LSTM:")
        print(f"Accuracy: {lstm_metrics['accuracy']:.4f}")
        print(f"Precision: {lstm_metrics['precision']:.4f}")
        print(f"Recall: {lstm_metrics['recall']:.4f}")
        print(f"F1-score: {lstm_metrics['f1']:.4f}")
        print(f"Час інференсу: {lstm_metrics['inference_time']:.2f} мс/зразок")
        
        # Візуалізація матриці помилок
        plot_confusion_matrix(
            lstm_metrics['y_true'], lstm_metrics['y_pred'],
            list(idx2label.values()), 'Матриця помилок LSTM'
        )
        
        # Візуалізація F1-міри для кожного класу
        plot_f1_per_class(
            lstm_metrics['y_true'], lstm_metrics['y_pred'],
            list(idx2label.values()), 'F1-міра за класами (LSTM)'
        )
        
        # Збереження LSTM моделі
        lstm_metadata = {
            'vocab_size': vocab_size,
            'embedding_dim': lstm_model.embedding.embedding_dim,
            'hidden_dim': lstm_model.lstm.hidden_size,
            'output_dim': lstm_model.fc2.out_features,
            'n_layers': lstm_model.lstm.num_layers,
            'bidirectional': lstm_model.lstm.bidirectional,
            'dropout': lstm_model.dropout.p,
            'pad_idx': 0,
            'label2idx': label2idx,
            'idx2label': idx2label,
            'word2idx': word2idx,
            'metrics': {
                'accuracy': lstm_metrics['accuracy'],
                'precision': lstm_metrics['precision'],
                'recall': lstm_metrics['recall'],
                'f1': lstm_metrics['f1'],
                'inference_time': lstm_metrics['inference_time']
            }
        }
        
        save_model(
            lstm_model, f'{args.save_dir}/lstm_emotion_model.pt',
            f'{args.save_dir}/lstm_emotion_config.json', lstm_metadata
        )
    
    # Тренування та оцінка BERT-lite моделі
    if args.train_bert:
        print("Підготовка даних для BERT-lite...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        train_dataloader_bert, val_dataloader_bert, test_dataloader_bert, \
        label2idx_bert, idx2label_bert, num_classes = prepare_data_for_bert(
            train_df, val_df, test_df, tokenizer
        )
        
        print("Ініціалізація BERT-lite моделі...")
        bert_model = create_bert_lite_model(num_classes, device=device)
        
        print("Тренування BERT-lite моделі...")
        optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        bert_model, bert_history = train_bert_lite_model(
            bert_model, train_dataloader_bert, val_dataloader_bert,
            optimizer, criterion, 10, 3, device
        )
        
        # Візуалізація історії навчання
        plot_training_history(bert_history, 'Історія навчання BERT-lite')
        
        # Оцінка BERT-lite моделі
        print("Оцінка BERT-lite моделі...")
        bert_metrics = evaluate_model_detailed(
            bert_model, test_dataloader_bert, idx2label_bert, device, 'bert'
        )
        
        print("\nРезультати для BERT-lite:")
        print(f"Accuracy: {bert_metrics['accuracy']:.4f}")
        print(f"Precision: {bert_metrics['precision']:.4f}")
        print(f"Recall: {bert_metrics['recall']:.4f}")
        print(f"F1-score: {bert_metrics['f1']:.4f}")
        print(f"Час інференсу: {bert_metrics['inference_time']:.2f} мс/зразок")
        
        # Візуалізація матриці помилок
        plot_confusion_matrix(
            bert_metrics['y_true'], bert_metrics['y_pred'],
            list(idx2label_bert.values()), 'Матриця помилок BERT-lite'
        )
        
        # Візуалізація F1-міри для кожного класу
        plot_f1_per_class(
            bert_metrics['y_true'], bert_metrics['y_pred'],
            list(idx2label_bert.values()), 'F1-міра за класами (BERT-lite)'
        )
        
        # Збереження BERT-lite моделі
        bert_metadata = {
            'num_classes': bert_model.classifier.out_features,
            'hidden_size': bert_model.config.hidden_size,
            'num_hidden_layers': bert_model.config.num_hidden_layers,
            'num_attention_heads': bert_model.config.num_attention_heads,
            'intermediate_size': bert_model.config.intermediate_size,
            'dropout': bert_model.dropout.p,
            'label2idx': label2idx_bert,
            'idx2label': idx2label_bert,
            'tokenizer': 'bert-base-multilingual-cased',
            'metrics': {
                'accuracy': bert_metrics['accuracy'],
                'precision': bert_metrics['precision'],
                'recall': bert_metrics['recall'],
                'f1': bert_metrics['f1'],
                'inference_time': bert_metrics['inference_time']
            }
        }
        
        save_model(
            bert_model, f'{args.save_dir}/bert_emotion_model.pt',
            f'{args.save_dir}/bert_emotion_config.json', bert_metadata
        )
    
    # Порівняння моделей
    if args.compare and args.train_lstm and args.train_bert:
        print("\nПорівняння моделей:")
        
        # Порівняння загальних метрик
        plot_models_comparison(
            lstm_metrics, bert_metrics, 'general',
            title='Порівняння загальних метрик'
        )
        
        # Порівняння F1-міри за класами
        f1_per_class_lstm = [lstm_metrics['f1_per_class'][label] for label in idx2label.values()]
        f1_per_class_bert = [bert_metrics['f1_per_class'][label] for label in idx2label_bert.values()]
        
        plot_models_comparison(
            f1_per_class_lstm, f1_per_class_bert, 'f1_per_class',
            list(idx2label.values()), 'Порівняння F1-міри за класами'
        )
        
        # Порівняння на різних обсягах даних
        print("\nПорівняння моделей на різних обсягах даних...")
        
        from evaluation import compare_models_on_different_data_sizes
        
        results = compare_models_on_different_data_sizes(
            lstm_model, bert_model, 
            train_dataloader_lstm.dataset, val_dataloader_lstm, test_dataloader_lstm,
            [0.1, 0.25, 0.5, 0.75, 1.0],
            device
        )
        
        # Візуалізація результатів
        plot_data_size_comparison(
            results['data_sizes'], results['lstm_f1s'], results['bert_f1s'],
            'Залежність F1-міри від обсягу даних'
        )
    
    # Передбачення емоції для вказаного тексту
    if args.predict:
        from evaluation import load_lstm_model, load_bert_model
        
        print(f"\nПередбачення емоції для тексту: {args.predict}")
        
        # Завантаження LSTM моделі
        if os.path.exists(f'{args.save_dir}/lstm_emotion_model.pt'):
            lstm_model, lstm_config = load_lstm_model(
                f'{args.save_dir}/lstm_emotion_model.pt',
                f'{args.save_dir}/lstm_emotion_config.json',
                device
            )
            
            # Передбачення емоції за допомогою LSTM
            word2idx = lstm_config['word2idx']
            idx2label = lstm_config['idx2label']
            
            lstm_preproc = lambda text: preprocess_for_lstm(text, word2idx)
            emotion_lstm, prob_lstm, _ = predict_emotion(args.predict, lstm_model, lstm_preproc, idx2label, device)
            
            print(f"LSTM: {emotion_lstm} (ймовірність: {prob_lstm:.4f})")
        
        # Завантаження BERT-lite моделі
        if os.path.exists(f'{args.save_dir}/bert_emotion_model.pt'):
            bert_model, bert_config = load_bert_model(
                f'{args.save_dir}/bert_emotion_model.pt',
                f'{args.save_dir}/bert_emotion_config.json',
                device
            )
            
            # Передбачення емоції за допомогою BERT-lite
            idx2label_bert = bert_config['idx2label']
            tokenizer = BertTokenizer.from_pretrained(bert_config['tokenizer'])
            
            bert_preproc = lambda text: preprocess_for_bert(text, tokenizer)
            emotion_bert, prob_bert, _ = predict_emotion(args.predict, bert_model, bert_preproc, idx2label_bert, device)
            
            print(f"BERT-lite: {emotion_bert} (ймовірність: {prob_bert:.4f})")

def interactive_mode():
    """
    Інтерактивний режим для класифікації емоцій
    """
    from evaluation import load_lstm_model, load_bert_model
    
    # Визначення пристрою для обчислень
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Завантаження моделей
    models = {}
    
    # Завантаження LSTM моделі
    if os.path.exists('models/lstm_emotion_model.pt'):
        print("Завантаження LSTM моделі...")
        lstm_model, lstm_config = load_lstm_model(
            'models/lstm_emotion_model.pt',
            'models/lstm_emotion_config.json',
            device
        )
        
        word2idx = lstm_config['word2idx']
        idx2label = lstm_config['idx2label']
        
        models['lstm'] = {
            'model': lstm_model,
            'preproc': lambda text: preprocess_for_lstm(text, word2idx),
            'idx2label': idx2label
        }
    
    # Завантаження BERT-lite моделі
    if os.path.exists('models/bert_emotion_model.pt'):
        print("Завантаження BERT-lite моделі...")
        bert_model, bert_config = load_bert_model(
            'models/bert_emotion_model.pt',
            'models/bert_emotion_config.json',
            device
        )
        
        idx2label_bert = bert_config['idx2label']
        tokenizer = BertTokenizer.from_pretrained(bert_config['tokenizer'])
        
        models['bert'] = {
            'model': bert_model,
            'preproc': lambda text: preprocess_for_bert(text, tokenizer),
            'idx2label': idx2label_bert
        }
    
    if not models:
        print("Не знайдено жодної моделі. Будь ласка, спочатку тренуйте моделі.")
        return
    
    print("\n=== Інтерактивний режим класифікації емоцій ===")
    print("Введіть текст для аналізу або 'exit' для виходу")
    
    while True:
        text = input("\nВведіть текст: ")
        
        if text.lower() == 'exit':
            break
        
        print(f"Аналіз тексту: {text}")
        
        # Аналіз кожною моделлю
        for model_name, model_data in models.items():
            emotion, probability, _ = predict_emotion(
                text, model_data['model'], model_data['preproc'], model_data['idx2label'], device
            )
            
            print(f"{model_name.upper()}: {emotion} (ймовірність: {probability:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Класифікація емоцій у тексті')
    parser.add_argument('--interactive', action='store_true', help='Запустити в інтерактивному режимі')
    args, unknown = parser.parse_known_args()
    
    if args.interactive:
        interactive_mode()
    else:
        main()
