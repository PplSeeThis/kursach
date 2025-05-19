import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    """
    LSTM модель для классификации эмоций в тексте
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, pretrained_embeddings=None):
        """
        Инициализация модели
        
        Args:
            vocab_size: размер словаря
            embedding_dim: размерность вложения слов
            hidden_dim: размерность скрытого состояния LSTM
            output_dim: размерность выходного слоя (количество классов)
            n_layers: количество слоев LSTM
            bidirectional: использовать ли двунаправленный LSTM
            dropout: вероятность dropout
            pad_idx: индекс паддинга в словаре
            pretrained_embeddings: предобученные вложения слов (опционально)
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Инициализация слоя вложения предобученными вложениями (если есть)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        
        # Определяем размерность выхода LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Linear(lstm_output_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(128, output_dim)
        
    def forward(self, text, text_lengths):
        """
        Прямой проход
        
        Args:
            text: тензор [batch_size, seq_len] с индексами слов
            text_lengths: тензор [batch_size] с длинами последовательностей
            
        Returns:
            output: тензор [batch_size, output_dim] с логитами
        """
        # text = [batch size, seq len]
        
        embedded = self.embedding(text)
        # embedded = [batch size, seq len, emb dim]
        
        # Упаковка последовательностей для эффективной обработки
        # Сортировка последовательностей по длине (требуется для pack_padded_sequence)
        text_lengths, sort_idx = text_lengths.sort(descending=True)
        embedded = embedded[sort_idx]
        
        # Упаковка последовательностей
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Распаковка выхода
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        # output = [batch size, seq len, hidden dim * num directions]
        
        # Глобальный max-pooling
        output = output.permute(0, 2, 1)
        # output = [batch size, hidden dim * num directions, seq len]
        
        pooled = F.adaptive_max_pool1d(output, 1).squeeze(2)
        # pooled = [batch size, hidden dim * num directions]
        
        # Восстановление исходного порядка
        _, unsort_idx = sort_idx.sort()
        pooled = pooled[unsort_idx]
        
        # Полносвязные слои
        dense = self.fc(pooled)
        dense = F.relu(dense)
        dense = self.dropout(dense)
        
        output = self.output(dense)
        
        return output

def create_lstm_model(vocab_size, embedding_dim=300, hidden_dim=256, output_dim=7, 
                     n_layers=2, bidirectional=True, dropout=0.3, pad_idx=0,
                     pretrained_embeddings=None):
    """
    Создание LSTM модели
    
    Args:
        vocab_size: размер словаря
        embedding_dim: размерность вложения слов
        hidden_dim: размерность скрытого состояния LSTM
        output_dim: размерность выходного слоя (количество классов)
        n_layers: количество слоев LSTM
        bidirectional: использовать ли двунаправленный LSTM
        dropout: вероятность dropout
        pad_idx: индекс паддинга в словаре
        pretrained_embeddings: предобученные вложения слов (опционально)
    
    Returns:
        model: инициализированная LSTM модель
    """
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        pad_idx=pad_idx,
        pretrained_embeddings=pretrained_embeddings
    )
    
    # Подсчет количества параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"LSTM model created with {total_params:,} total parameters, {trainable_params:,} trainable")
    
    return model

if __name__ == "__main__":
    # Пример использования
    vocab_size = 50000
    model = create_lstm_model(vocab_size)
    
    # Тест на случайных входных данных
    batch_size = 16
    seq_len = 128
    text = torch.randint(0, vocab_size, (batch_size, seq_len))
    text_lengths = torch.randint(1, seq_len+1, (batch_size,))
    
    output = model(text, text_lengths)
    print(f"Output shape: {output.shape}")  # Должно быть [batch_size, 7]
