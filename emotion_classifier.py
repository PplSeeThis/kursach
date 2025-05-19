import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from preprocessing import preprocess_text

def predict_emotion(text, model, vocab=None, tokenizer=None, label_mapping=None, device='cpu', model_type='lstm'):
    """
    Предсказание эмоции для текста
    
    Args:
        text: текст для классификации
        model: модель (LSTM или BERT-lite)
        vocab: словарь для LSTM (опционально)
        tokenizer: токенизатор для BERT (опционально)
        label_mapping: соответствие меток и классов
        device: устройство (cpu или cuda)
        model_type: тип модели ('lstm' или 'bert')
        
    Returns:
        predicted_emotion: предсказанная эмоция
        probability: вероятность предсказания
    """
    # Инвертирование label_mapping для получения названий эмоций
    id_to_label = {v: k for k, v in label_mapping.items()}
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Предобработка текста
    preprocessed_text = preprocess_text(text)
    
    # Предобработка текста в зависимости от типа модели
    if model_type == 'lstm':
        if vocab is None:
            raise ValueError("vocab must be provided for LSTM model")
        
        # Токенизация и преобразование в индексы
        tokens = preprocessed_text.split()
        indexes = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        
        # Ограничение длины
        max_len = 128
        if len(indexes) > max_len:
            indexes = indexes[:max_len]
        
        # Создание тензора и переход на устройство
        text_tensor = torch.tensor(indexes, dtype=torch.long).unsqueeze(0).to(device)
        text_length = torch.tensor([len(indexes)], dtype=torch.long).to(device)
        
        # Предсказание
        with torch.no_grad():
            predictions = model(text_tensor, text_length)
    else:  # bert
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for BERT model")
        
        # Токенизация с помощью BERT-токенизатора
        encoding = tokenizer.encode_plus(
            preprocessed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Переход на устройство
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Предсказание
        with torch.no_grad():
            predictions = model(input_ids, attention_mask)
    
    # Получение предсказанного класса и вероятности
    probabilities = F.softmax(predictions, dim=1)
    predicted_idx = torch.argmax(predictions, dim=1).item()
    probability = probabilities[0][predicted_idx].item()
    
    # Получение названия эмоции
    predicted_emotion = id_to_label[predicted_idx]
    
    return predicted_emotion, probability

def visualize_attention(text, model, tokenizer, device='cpu'):
    """
    Визуализация внимания BERT-модели
    
    Args:
        text: текст для визуализации
        model: BERT-модель
        tokenizer: токенизатор BERT
        device: устройство (cpu или cuda)
    """
    # Токенизация текста
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Переход на устройство
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Получение выходов BERT-модели
    model.eval()
    with torch.no_grad():
        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
    
    # Получение матрицы внимания
    attentions = outputs.attentions  # Размер: (batch_size, num_heads, seq_len, seq_len)
    
    # Преобразование индексов в токены
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Удаление паддинга из списка токенов
    actual_tokens = []
    for token, mask in zip(tokens, attention_mask[0]):
        if mask:
            actual_tokens.append(token)
        else:
            break
    
    # Выбор слоя и головки внимания для визуализации
    layer_idx = -1  # Последний слой
    head_idx = 0    # Первая головка
    
    # Получение матрицы внимания для выбранного слоя и головки
    attention_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()
    
    # Обрезка матрицы внимания до фактической длины последовательности
    attention_matrix = attention_matrix[:len(actual_tokens), :len(actual_tokens)]
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, annot=False, cmap='viridis', xticklabels=actual_tokens, yticklabels=actual_tokens)
    plt.title(f'BERT Attention - Layer {layer_idx+1}, Head {head_idx+1}')
    plt.xlabel('Token (Key)')
    plt.ylabel('Token (Query)')
    plt.tight_layout()
    plt.show()

def analyze_errors(model, test_loader, criterion, device, class_names, model_type='lstm'):
    """
    Анализ ошибок модели
    
    Args:
        model: модель (LSTM или BERT-lite)
        test_loader: даталоадер с тестовыми данными
        criterion: функция потерь
        device: устройство (cpu или cuda)
        class_names: названия классов
        model_type: тип модели ('lstm' или 'bert')
        
    Returns:
        error_samples: примеры с ошибками
        conf_matrix: матрица ошибок
    """
    # Переводим модель в режим оценки
    model.eval()
    
    all_preds = []
    all_labels = []
    error_samples = []
    
    # Отключение вычисления градиентов для ускорения оценки
    with torch.no_grad():
        for batch in test_loader:
            # Извлечение данных из батча в зависимости от типа модели
            if model_type == 'lstm':
                text = batch['text'].to(device)
                text_lengths = batch['length'].to(device)
                labels = batch['label'].to(device)
                
                # Прямой проход
                predictions = model(text, text_lengths)
            else:  # bert
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Прямой проход
                predictions = model(input_ids, attention_mask)
            
            # Получение предсказанных классов
            preds = torch.argmax(predictions, dim=1)
            
            # Сохранение предсказаний и истинных меток
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Поиск ошибок
            errors = preds != labels
            error_indices = torch.nonzero(errors).squeeze(-1)
            
            # Сохранение примеров с ошибками
            for idx in error_indices:
                idx = idx.item()
                if model_type == 'lstm':
                    error_samples.append({
                        'text_idx': batch['text'][idx].cpu().numpy(),
                        'true_label': labels[idx].item(),
                        'pred_label': preds[idx].item()
                    })
                else:  # bert
                    error_samples.append({
                        'input_ids': batch['input_ids'][idx].cpu().numpy(),
                        'true_label': labels[idx].item(),
                        'pred_label': preds[idx].item()
                    })
    
    # Вычисление матрицы ошибок
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Визуализация матрицы ошибок
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return error_samples, conf_matrix

def classify_text_batch(texts, model, vocab=None, tokenizer=None, label_mapping=None, device='cpu', model_type='lstm'):
    """
    Классификация пакета текстов
    
    Args:
        texts: список текстов для классификации
        model: модель (LSTM или BERT-lite)
        vocab: словарь для LSTM (опционально)
        tokenizer: токенизатор для BERT (опционально)
        label_mapping: соответствие меток и классов
        device: устройство (cpu или cuda)
        model_type: тип модели ('lstm' или 'bert')
        
    Returns:
        results: список кортежей (предсказанная эмоция, вероятность) для каждого текста
    """
    # Инвертирование label_mapping для получения названий эмоций
    id_to_label = {v: k for k, v in label_mapping.items()}
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Предобработка текстов
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Предобработка текста в зависимости от типа модели
    if model_type == 'lstm':
        if vocab is None:
            raise ValueError("vocab must be provided for LSTM model")
        
        # Токенизация и преобразование в индексы
        indexed_texts = []
        lengths = []
        
        for text in preprocessed_texts:
            tokens = text.split()
            indexes = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
            
            # Ограничение длины
            max_len = 128
            if len(indexes) > max_len:
                indexes = indexes[:max_len]
            
            # Дополнение до максимальной длины
            padding_length = max_len - len(indexes)
            if padding_length > 0:
                indexes = indexes + [vocab["<PAD>"]] * padding_length
            
            indexed_texts.append(indexes)
            lengths.append(min(len(tokens), max_len))
        
        # Создание тензора и переход на устройство
        text_tensor = torch.tensor(indexed_texts, dtype=torch.long).to(device)
        text_lengths = torch.tensor(lengths, dtype=torch.long).to(device)
        
        # Предсказание
        with torch.no_grad():
            predictions = model(text_tensor, text_lengths)
    else:  # bert
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for BERT model")
        
        # Токенизация с помощью BERT-токенизатора
        encodings = tokenizer.batch_encode_plus(
            preprocessed_texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Переход на устройство
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Предсказание
        with torch.no_grad():
            predictions = model(input_ids, attention_mask)
    
    # Получение предсказанных классов и вероятностей
    probabilities = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1).cpu().numpy()
    
    # Формирование результатов
    results = []
    for i, idx in enumerate(predicted_indices):
        emotion = id_to_label[idx]
        probability = probabilities[i, idx].item()
        results.append((emotion, probability))
    
    return results

def save_model(model, vocab=None, tokenizer=None, label_mapping=None, model_type='lstm', save_path='emotion_classifier_model'):
    """
    Сохранение модели и дополнительных данных
    
    Args:
        model: модель (LSTM или BERT-lite)
        vocab: словарь для LSTM (опционально)
        tokenizer: токенизатор для BERT (опционально)
        label_mapping: соответствие меток и классов
        model_type: тип модели ('lstm' или 'bert')
        save_path: путь для сохранения модели
    """
    # Создание словаря с данными для сохранения
    save_data = {
        'model_state': model.state_dict(),
        'label_mapping': label_mapping,
        'model_type': model_type
    }
    
    # Добавление vocab или tokenizer в зависимости от типа модели
    if model_type == 'lstm' and vocab is not None:
        save_data['vocab'] = vocab
    elif model_type == 'bert' and tokenizer is not None:
        save_data['tokenizer'] = tokenizer
    
    # Сохранение данных
    torch.save(save_data, save_path)
    print(f"Model saved to {save_path}")

def load_model(load_path, device='cpu'):
    """
    Загрузка модели и дополнительных данных
    
    Args:
        load_path: путь для загрузки модели
        device: устройство (cpu или cuda)
        
    Returns:
        model: загруженная модель
        vocab: словарь для LSTM (может быть None)
        tokenizer: токенизатор для BERT (может быть None)
        label_mapping: соответствие меток и классов
        model_type: тип модели ('lstm' или 'bert')
    """
    # Загрузка данных
    save_data = torch.load(load_path, map_location=device)
    
    # Извлечение данных
    model_state = save_data['model_state']
    label_mapping = save_data['label_mapping']
    model_type = save_data['model_type']
    
    # Создание модели в зависимости от типа
    if model_type == 'lstm':
        from lstm_model import create_lstm_model
        
        # Извлечение словаря
        vocab = save_data.get('vocab', None)
        if vocab is None:
            raise ValueError("vocab not found in saved data")
        
        # Создание модели
        model = create_lstm_model(
            vocab_size=len(vocab),
            embedding_dim=300,
            hidden_dim=256,
            output_dim=len(label_mapping),
            n_layers=2,
            bidirectional=True,
            dropout=0.3,
            pad_idx=vocab["<PAD>"]
        )
        
        # Загрузка весов
        model.load_state_dict(model_state)
        model = model.to(device)
        
        return model, vocab, None, label_mapping, model_type
    else:  # bert
        from bert_lite_model import create_bert_lite_model
        from transformers import BertTokenizer
        
        # Извлечение токенизатора
        tokenizer = save_data.get('tokenizer', None)
        if tokenizer is None:
            # Если токенизатор не сохранен, загружаем стандартный
            tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        
        # Создание модели
        model = create_bert_lite_model(
            num_classes=len(label_mapping),
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            dropout=0.3
        )
        
        # Загрузка весов
        model.load_state_dict(model_state)
        model = model.to(device)
        
        return model, None, tokenizer, label_mapping, model_type

if __name__ == "__main__":
    # Пример использования
    # Эти примеры предполагают, что модель, словарь/токенизатор и label_mapping уже загружены
    
    # Пример текста для классификации
    sample_text = "Я дуже щаслива сьогодні!"
    
    # Классификация текста с помощью LSTM
    # emotion, probability = predict_emotion(sample_text, lstm_model, vocab=vocab, label_mapping=label_mapping, model_type='lstm')
    # print(f"LSTM: {emotion} (Вероятность: {probability:.4f})")
    
    # Классификация текста с помощью BERT-lite
    # emotion, probability = predict_emotion(sample_text, bert_model, tokenizer=tokenizer, label_mapping=label_mapping, model_type='bert')
    # print(f"BERT-lite: {emotion} (Вероятность: {probability:.4f})")
    
    # Визуализация внимания BERT
    # visualize_attention(sample_text, bert_model, tokenizer)
    
    # Классификация пакета текстов
    # sample_texts = ["Я дуже щаслива сьогодні!", "Мені дуже сумно.", "Я розлючений!"]
    # results = classify_text_batch(sample_texts, bert_model, tokenizer=tokenizer, label_mapping=label_mapping, model_type='bert')
    # for text, (emotion, probability) in zip(sample_texts, results):
    #     print(f"Текст: {text}")
    #     print(f"Эмоция: {emotion} (Вероятность: {probability:.4f})")
    #     print()
