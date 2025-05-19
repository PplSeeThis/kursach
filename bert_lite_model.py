import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERTLiteModel(nn.Module):
    """
    BERT-lite модель для классификации эмоций в тексте
    """
    def __init__(self, num_classes, hidden_size=512, num_hidden_layers=6, 
                 num_attention_heads=8, intermediate_size=2048, dropout=0.3,
                 pretrained_model_name=None):
        """
        Инициализация модели
        
        Args:
            num_classes: количество классов (эмоций)
            hidden_size: размерность скрытого состояния
            num_hidden_layers: количество слоев трансформера
            num_attention_heads: количество головок внимания
            intermediate_size: размерность промежуточного слоя в feed-forward сети
            dropout: вероятность dropout
            pretrained_model_name: имя предобученной модели (опционально)
        """
        super().__init__()
        
        if pretrained_model_name:
            # Загрузка предобученной модели
            self.bert = BertModel.from_pretrained(pretrained_model_name)
            
            # Опционально: изменение количества слоев для создания lite версии
            if num_hidden_layers < len(self.bert.encoder.layer):
                self.bert.encoder.layer = self.bert.encoder.layer[:num_hidden_layers]
        else:
            # Создание конфигурации BERT-lite
            config = BertConfig(
                vocab_size=119547,  # Размер словаря для bert-base-multilingual-cased
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                max_position_embeddings=512,
                type_vocab_size=2
            )
            
            # Инициализация модели с конфигурацией
            self.bert = BertModel(config)
        
        # Классификационная надстройка
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Прямой проход
        
        Args:
            input_ids: тензор [batch_size, seq_len] с индексами токенов
            attention_mask: тензор [batch_size, seq_len] с маской внимания
            
        Returns:
            output: тензор [batch_size, num_classes] с логитами
        """
        # input_ids = [batch size, seq len]
        # attention_mask = [batch size, seq len]
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Используем CLS токен для классификации
        pooled_output = outputs.pooler_output
        # pooled_output = [batch size, hidden size]
        
        pooled_output = self.dropout(pooled_output)
        hidden = self.fc(pooled_output)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.classifier(hidden)
        
        return output

def create_bert_lite_model(num_classes=7, hidden_size=512, num_hidden_layers=6, 
                          num_attention_heads=8, intermediate_size=2048, dropout=0.3,
                          pretrained_model_name="bert-base-multilingual-cased"):
    """
    Создание BERT-lite модели
    
    Args:
        num_classes: количество классов (эмоций)
        hidden_size: размерность скрытого состояния
        num_hidden_layers: количество слоев трансформера
        num_attention_heads: количество головок внимания
        intermediate_size: размерность промежуточного слоя в feed-forward сети
        dropout: вероятность dropout
        pretrained_model_name: имя предобученной модели
    
    Returns:
        model: инициализированная BERT-lite модель
    """
    model = BERTLiteModel(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        dropout=dropout,
        pretrained_model_name=pretrained_model_name
    )
    
    # Подсчет количества параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"BERT-lite model created with {total_params:,} total parameters, {trainable_params:,} trainable")
    
    return model

if __name__ == "__main__":
    # Пример использования
    model = create_bert_lite_model()
    
    # Тест на случайных входных данных
    batch_size = 16
    seq_len = 128
    input_ids = torch.randint(0, 100000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    
    output = model(input_ids, attention_mask)
    print(f"Output shape: {output.shape}")  # Должно быть [batch_size, 7]
