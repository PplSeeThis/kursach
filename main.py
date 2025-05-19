import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AdamW, get_linear_schedule_with_warmup

# Импорт собственных модулей
from preprocessing import prepare_data, create_datasets_with_different_sizes
from bert_data_preparation import prepare_data_for_lstm, prepare_data_for_bert
from lstm_model import create_lstm_model
from bert_lite_model import create_bert_lite_model
from training import (train_lstm_model, train_bert_model, test_model,
                     plot_training_curves, plot_confusion_matrix,
                     plot_f1_scores_by_emotion, compare_models_f1,
                     plot_data_size_comparison)

def set_seed(seed_value=42):
    """
    Установка seed для воспроизводимости результатов
    
    Args:
        seed_value: значение seed
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_lstm_experiments(X_train, X_val, X_test, y_train, y_val, y_test, label_mapping, output_dir):
    """
    Запуск экспериментов с LSTM моделью
    
    Args:
        X_train, X_val, X_test: тренировочные, валидационные и тестовые тексты
        y_train, y_val, y_test: тренировочные, валидационные и тестовые метки
        label_mapping: соответствие меток и классов
        output_dir: директория для сохранения результатов
    
    Returns:
        lstm_results: результаты экспериментов с LSTM
    """
    print("Preparing data for LSTM...")
    train_loader, val_loader, test_loader, vocab = prepare_data_for_lstm(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("Creating LSTM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    lstm_model = create_lstm_model(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_dim=256,
        output_dim=len(label_mapping),
        n_layers=2,
        bidirectional=True,
        dropout=0.3,
        pad_idx=vocab["<PAD>"]
    )
    
    # Настройка оптимизатора и функции потерь
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Обучение модели
    print("Training LSTM model...")
    lstm_save_path = os.path.join(output_dir, 'lstm_model.pt')
    lstm_history, lstm_model, lstm_training_time = train_lstm_model(
        lstm_model, train_loader, val_loader, optimizer, criterion, 
        n_epochs=30, device=device, scheduler=scheduler, 
        patience=5, model_save_path=lstm_save_path
    )
    
    # Построение графиков обучения
    lstm_curves_path = os.path.join(output_dir, 'lstm_training_curves.png')
    plot_training_curves(lstm_history, save_path=lstm_curves_path)
    
    # Тестирование модели
    print("Testing LSTM model...")
    lstm_acc, lstm_f1_macro, lstm_f1_weighted, lstm_precision, lstm_recall, lstm_conf_matrix, lstm_inference_time = test_model(
        lstm_model, test_loader, criterion, device, model_type='lstm'
    )
    
    # Построение матрицы ошибок
    id_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]
    
    lstm_cm_path = os.path.join(output_dir, 'lstm_confusion_matrix.png')
    plot_confusion_matrix(
        lstm_conf_matrix, class_names, model_name='LSTM', 
        normalize=True, save_path=lstm_cm_path
    )
    
    # Построение графика метрик для каждой эмоции
    lstm_metrics_path = os.path.join(output_dir, 'lstm_metrics_by_emotion.png')
    plot_f1_scores_by_emotion(
        lstm_precision, lstm_recall, lstm_f1_macro, class_names, 
        model_name='LSTM', save_path=lstm_metrics_path
    )
    
    # Сохранение результатов
    lstm_results = {
        'model': lstm_model,
        'history': lstm_history,
        'accuracy': lstm_acc,
        'f1_macro': lstm_f1_macro,
        'f1_weighted': lstm_f1_weighted,
        'precision': lstm_precision,
        'recall': lstm_recall,
        'conf_matrix': lstm_conf_matrix,
        'training_time': lstm_training_time,
        'inference_time': lstm_inference_time
    }
    
    return lstm_results

def run_bert_experiments(X_train, X_val, X_test, y_train, y_val, y_test, label_mapping, output_dir):
    """
    Запуск экспериментов с BERT-lite моделью
    
    Args:
        X_train, X_val, X_test: тренировочные, валидационные и тестовые тексты
        y_train, y_val, y_test: тренировочные, валидационные и тестовые метки
        label_mapping: соответствие меток и классов
        output_dir: директория для сохранения результатов
    
    Returns:
        bert_results: результаты экспериментов с BERT-lite
    """
    print("Preparing data for BERT-lite...")
    train_loader, val_loader, test_loader, tokenizer = prepare_data_for_bert(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )
    
    print("Creating BERT-lite model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    bert_model = create_bert_lite_model(
        num_classes=len(label_mapping),
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.3,
        pretrained_model_name="bert-base-multilingual-cased"
    )
    
    # Настройка оптимизатора и функции потерь
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Создание планировщика скорости обучения с разогревом
    total_steps = len(train_loader) * 10  # 10 эпох
    warmup_steps = int(total_steps * 0.1)  # 10% от общего количества шагов
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Обучение модели
    print("Training BERT-lite model...")
    bert_save_path = os.path.join(output_dir, 'bert_lite_model.pt')
    bert_history, bert_model, bert_training_time = train_bert_model(
        bert_model, train_loader, val_loader, optimizer, criterion, 
        n_epochs=10, device=device, scheduler=None, 
        patience=3, model_save_path=bert_save_path
    )
    
    # Построение графиков обучения
    bert_curves_path = os.path.join(output_dir, 'bert_training_curves.png')
    plot_training_curves(bert_history, save_path=bert_curves_path)
    
    # Тестирование модели
    print("Testing BERT-lite model...")
    bert_acc, bert_f1_macro, bert_f1_weighted, bert_precision, bert_recall, bert_conf_matrix, bert_inference_time = test_model(
        bert_model, test_loader, criterion, device, model_type='bert'
    )
    
    # Построение матрицы ошибок
    id_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]
    
    bert_cm_path = os.path.join(output_dir, 'bert_confusion_matrix.png')
    plot_confusion_matrix(
        bert_conf_matrix, class_names, model_name='BERT-lite', 
        normalize=True, save_path=bert_cm_path
    )
    
    # Построение графика метрик для каждой эмоции
    bert_metrics_path = os.path.join(output_dir, 'bert_metrics_by_emotion.png')
    plot_f1_scores_by_emotion(
        bert_precision, bert_recall, bert_f1_macro, class_names, 
        model_name='BERT-lite', save_path=bert_metrics_path
    )
    
    # Сохранение результатов
    bert_results = {
        'model': bert_model,
        'history': bert_history,
        'accuracy': bert_acc,
        'f1_macro': bert_f1_macro,
        'f1_weighted': bert_f1_weighted,
        'precision': bert_precision,
        'recall': bert_recall,
        'conf_matrix': bert_conf_matrix,
        'training_time': bert_training_time,
        'inference_time': bert_inference_time
    }
    
    return bert_results

def compare_models(lstm_results, bert_results, label_mapping, output_dir):
    """
    Сравнение результатов экспериментов с LSTM и BERT-lite моделями
    
    Args:
        lstm_results: результаты экспериментов с LSTM
        bert_results: результаты экспериментов с BERT-lite
        label_mapping: соответствие меток и классов
        output_dir: директория для сохранения результатов
    """
    # Подготовка данных для сравнения
    id_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]
    
    # Создание таблицы сравнения
    comparison_data = {
        'Метрика': ['Точность (Accuracy)', 'F1-мера (Macro)', 'F1-мера (Weighted)', 
                   'Время обучения (сек)', 'Время инференса (мс/образец)'],
        'LSTM': [lstm_results['accuracy'], lstm_results['f1_macro'], lstm_results['f1_weighted'],
                lstm_results['training_time'], lstm_results['inference_time'] * 1000],
        'BERT-lite': [bert_results['accuracy'], bert_results['f1_macro'], bert_results['f1_weighted'],
                    bert_results['training_time'], bert_results['inference_time'] * 1000]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Сохранение таблицы сравнения
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    # Сравнение F1-мер для каждой эмоции
    comparison_path = os.path.join(output_dir, 'emotion_comparison.png')
    compare_models_f1(
        f1_scores_lstm=lstm_results['f1_macro'], 
        f1_scores_bert=bert_results['f1_macro'],
        class_names=class_names,
        save_path=comparison_path
    )
    
    # Дополнительное сравнение метрик для каждой эмоции
    emotion_metrics = pd.DataFrame({
        'Эмоция': class_names,
        'LSTM_precision': lstm_results['precision'],
        'LSTM_recall': lstm_results['recall'],
        'LSTM_f1': lstm_results['f1_macro'],
        'BERT_precision': bert_results['precision'],
        'BERT_recall': bert_results['recall'],
        'BERT_f1': bert_results['f1_macro']
    })
    
    print("\nMetrics by Emotion:")
    print(emotion_metrics.to_string(index=False))
    
    # Сохранение метрик для каждой эмоции
    emotion_metrics_path = os.path.join(output_dir, 'emotion_metrics.csv')
    emotion_metrics.to_csv(emotion_metrics_path, index=False)

def run_data_size_experiments(X_train, X_val, X_test, y_train, y_val, y_test, label_mapping, output_dir):
    """
    Эксперименты с размером обучающей выборки
    
    Args:
        X_train, X_val, X_test: тренировочные, валидационные и тестовые тексты
        y_train, y_val, y_test: тренировочные, валидационные и тестовые метки
        label_mapping: соответствие меток и классов
        output_dir: директория для сохранения результатов
    """
    print("Running data size experiments...")
    
    # Создание подвыборок разного размера
    data_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    datasets = create_datasets_with_different_sizes(X_train, y_train, sizes=data_sizes)
    
    lstm_f1_scores = []
    bert_f1_scores = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for size in data_sizes:
        print(f"\nTraining with {size*100}% of data")
        X_subset, y_subset = datasets[size]
        
        # LSTM
        print("Preparing data for LSTM...")
        train_loader_lstm, val_loader_lstm, test_loader_lstm, vocab = prepare_data_for_lstm(
            X_subset, X_val, X_test, y_subset, y_val, y_test
        )
        
        print("Training LSTM model...")
        lstm_model = create_lstm_model(
            vocab_size=len(vocab),
            embedding_dim=300,
            hidden_dim=256,
            output_dim=len(label_mapping),
            n_layers=2,
            bidirectional=True,
            dropout=0.3,
            pad_idx=vocab["<PAD>"]
        )
        
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        lstm_save_path = os.path.join(output_dir, f'lstm_model_{int(size*100)}pct.pt')
        _, lstm_model, _ = train_lstm_model(
            lstm_model, train_loader_lstm, val_loader_lstm, optimizer, criterion, 
            n_epochs=20, device=device, scheduler=None, 
            patience=3, model_save_path=lstm_save_path
        )
        
        # Тестирование LSTM модели
        _, lstm_f1, _, _, _, _, _ = test_model(
            lstm_model, test_loader_lstm, criterion, device, model_type='lstm'
        )
        lstm_f1_scores.append(lstm_f1)
        
        # BERT-lite
        print("Preparing data for BERT-lite...")
        train_loader_bert, val_loader_bert, test_loader_bert, tokenizer = prepare_data_for_bert(
            X_subset, X_val, X_test, y_subset, y_val, y_test, batch_size=32
        )
        
        print("Training BERT-lite model...")
        bert_model = create_bert_lite_model(
            num_classes=len(label_mapping),
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            dropout=0.3,
            pretrained_model_name="bert-base-multilingual-cased"
        )
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        bert_save_path = os.path.join(output_dir, f'bert_model_{int(size*100)}pct.pt')
        _, bert_model, _ = train_bert_model(
            bert_model, train_loader_bert, val_loader_bert, optimizer, criterion, 
            n_epochs=6, device=device, scheduler=None, 
            patience=2, model_save_path=bert_save_path
        )
        
        # Тестирование BERT модели
        _, bert_f1, _, _, _, _, _ = test_model(
            bert_model, test_loader_bert, criterion, device, model_type='bert'
        )
        bert_f1_scores.append(bert_f1)
    
    # Построение графика зависимости F1-меры от размера обучающей выборки
    data_sizes_pct = [size * 100 for size in data_sizes]  # Convert to percentages
    data_size_path = os.path.join(output_dir, 'data_size_comparison.png')
    plot_data_size_comparison(
        data_sizes_pct, lstm_f1_scores, bert_f1_scores, save_path=data_size_path
    )
    
    # Сохранение результатов в CSV
    data_size_results = pd.DataFrame({
        'Data_Size_Pct': data_sizes_pct,
        'LSTM_F1': lstm_f1_scores,
        'BERT_F1': bert_f1_scores
    })
    
    data_size_csv_path = os.path.join(output_dir, 'data_size_results.csv')
    data_size_results.to_csv(data_size_csv_path, index=False)
    
    print("\nData Size Experiment Results:")
    print(data_size_results.to_string(index=False))

def main():
    """
    Основная функция для запуска экспериментов
    """
    # Установка seed для воспроизводимости
    set_seed(42)
    
    # Создание директории для результатов
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка и подготовка данных
    print("Loading and preparing data...")
    # В реальной курсовой нужно заменить на загрузку реальных данных
    # Здесь используем синтетические данные для иллюстрации
    
    # Создание синтетического датасета
    import pandas as pd
    import numpy as np
    
    # Функция для генерации случайного украинского текста
    def generate_random_ukrainian_text(length=50):
        ukr_alphabet = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
        return ' '.join(''.join(random.choice(ukr_alphabet) for _ in range(random.randint(3, 10)))
                       for _ in range(length))
    
    # Создание синтетического датасета
    n_samples = 10000  # Небольшой датасет для демонстрации
    
    emotions = ["радість", "смуток", "гнів", "страх", "відраза", "здивування", "нейтральний"]
    df = pd.DataFrame({
        'text': [generate_random_ukrainian_text() for _ in range(n_samples)],
        'emotion': [random.choice(emotions) for _ in range(n_samples)]
    })
    
    # Сохранение синтетического датасета
    df.to_csv(os.path.join(output_dir, 'synthetic_dataset.csv'), index=False)
    
    # Подготовка данных
    X_train, X_val, X_test, y_train, y_val, y_test, label_mapping = prepare_data(
        os.path.join(output_dir, 'synthetic_dataset.csv')
    )
    
    # Запуск экспериментов с LSTM
    lstm_results = run_lstm_experiments(
        X_train, X_val, X_test, y_train, y_val, y_test, label_mapping, output_dir
    )
    
    # Запуск экспериментов с BERT-lite
    bert_results = run_bert_experiments(
        X_train, X_val, X_test, y_train, y_val, y_test, label_mapping, output_dir
    )
    
    # Сравнение моделей
    compare_models(lstm_results, bert_results, label_mapping, output_dir)
    
    # Запуск экспериментов с размером обучающей выборки
    run_data_size_experiments(
        X_train, X_val, X_test, y_train, y_val, y_test, label_mapping, output_dir
    )
    
    print("\nAll experiments completed successfully!")
    print(f"Results saved to directory: {output_dir}")

if __name__ == "__main__":
    main()
