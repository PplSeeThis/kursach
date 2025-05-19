#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT-lite Hyperparameter Experiments
Script to conduct experiments with BERT-lite model hyperparameters
for emotion classification in Ukrainian text.

This script implements experiments with various configurations of BERT-lite:
- Different number of transformer layers (4, 6, 8)
- Different number of attention heads (4, 8, 12)
- Different hidden state dimensions (384, 512, 768)
- Different learning rates (5e-6, 2e-5, 5e-5)
- Different batch sizes (16, 32, 64)
"""

import os
import time
import random
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

# Set seed for reproducibility
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# Configuration class for experiments
class ExperimentConfig:
    def __init__(
        self,
        num_layers=6,
        num_attention_heads=8,
        hidden_size=512,
        learning_rate=2e-5,
        batch_size=32,
        max_seq_length=128,
        num_epochs=10,
        warmup_ratio=0.1,
        weight_decay=0.01,
        dropout=0.3
    ):
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.dropout = dropout
        
    def __str__(self):
        return (f"BERT-lite - Layers: {self.num_layers}, "
                f"Heads: {self.num_attention_heads}, "
                f"Hidden: {self.hidden_size}, "
                f"LR: {self.learning_rate}, "
                f"Batch: {self.batch_size}")
    
    def to_dict(self):
        return {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "num_epochs": self.num_epochs,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout
        }

# Class for loading and preprocessing data for BERT
class BertDataProcessor:
    def __init__(self, tokenizer, max_seq_length=128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, texts, labels, batch_size):
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Tokenize and encode texts
        input_ids = []
        attention_masks = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            encoded_text = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded_text['input_ids'])
            attention_masks.append(encoded_text['attention_mask'])
        
        # Convert to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(encoded_labels)
        
        # Create dataset
        dataset = TensorDataset(input_ids, attention_masks, labels)
        
        # Create dataloaders
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        
        return dataloader, len(self.label_encoder.classes_)
    
    def prepare_validation_data(self, texts, labels, batch_size):
        # Use existing label encoder
        encoded_labels = self.label_encoder.transform(labels)
        
        # Tokenize and encode texts
        input_ids = []
        attention_masks = []
        
        for text in tqdm(texts, desc="Tokenizing validation"):
            encoded_text = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded_text['input_ids'])
            attention_masks.append(encoded_text['attention_mask'])
        
        # Convert to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(encoded_labels)
        
        # Create dataset
        dataset = TensorDataset(input_ids, attention_masks, labels)
        
        # Create dataloaders
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        
        return dataloader
    
    def get_label_mapping(self):
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}

# Function to create a custom BERT-lite model
def create_bert_lite_model(config, num_labels):
    # Create custom BERT config
    bert_config = BertConfig(
        vocab_size=119547,  # For bert-base-multilingual-cased
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.hidden_size * 4,  # Typically 4x hidden_size
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        max_position_embeddings=512,
        type_vocab_size=2
    )
    
    # Create model with the custom config
    model = BertForSequenceClassification(
        config=bert_config,
        num_labels=num_labels
    )
    
    return model

# Function to initialize the model with pretrained weights
def initialize_from_pretrained(model, pretrained_model_name="bert-base-multilingual-cased"):
    # Load pretrained model
    pretrained_model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=model.config.num_labels
    )
    
    # Copy embeddings and first layers weights
    with torch.no_grad():
        # Copy embeddings
        model.bert.embeddings.load_state_dict(pretrained_model.bert.embeddings.state_dict())
        
        # Copy as many encoder layers as possible
        num_layers_to_copy = min(model.config.num_hidden_layers, pretrained_model.config.num_hidden_layers)
        for i in range(num_layers_to_copy):
            model.bert.encoder.layer[i].load_state_dict(pretrained_model.bert.encoder.layer[i].state_dict())
    
    return model

# Training function
def train_model(model, train_dataloader, val_dataloader, config, device, output_dir):
    # Prepare optimizer and schedule
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config.num_epochs
    
    # Create learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # Statistics to track
    training_stats = []
    best_val_f1 = 0
    best_model_path = None
    
    # Loop through epochs
    for epoch_i in range(config.num_epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {config.num_epochs} ========')
        print('Training...')
        
        # Track time and loss
        t0 = time.time()
        total_train_loss = 0
        model.train()
        
        # Training loop
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Extract batch and move to device
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Clear gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_attention_mask,
                labels=b_labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
        
        # Calculate average loss over all batches
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Training time
        training_time = time.time() - t0
        
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time:.2f}s")
        
        # Validation
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        
        # Tracking variables
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        # Validation loop
        for batch in tqdm(val_dataloader, desc="Validation"):
            # Extract batch and move to device
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # No gradients
            with torch.no_grad():
                # Forward pass
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_val_loss += loss.item()
                
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                # Calculate predictions
                preds = np.argmax(logits, axis=1)
                
                # Append to lists
                all_preds.extend(preds)
                all_labels.extend(label_ids)
        
        # Calculate average loss
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        # Calculate F1 score
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        val_f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        # Validation time
        validation_time = time.time() - t0
        
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  Macro F1: {val_f1_macro:.4f}")
        print(f"  Weighted F1: {val_f1_weighted:.4f}")
        print(f"  Validation took: {validation_time:.2f}s")
        
        # Save statistics
        training_stats.append({
            'epoch': epoch_i + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_f1_macro': val_f1_macro,
            'val_f1_weighted': val_f1_weighted,
            'training_time': training_time,
            'validation_time': validation_time
        })
        
        # Save model if this is the best F1 score
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            
            # Create directory if not exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Define model path
            model_name = f"bert_lite_l{config.num_layers}_h{config.num_attention_heads}_d{config.hidden_size}_lr{config.learning_rate}_bs{config.batch_size}"
            model_path = os.path.join(output_dir, model_name)
            
            # Save model
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
            
            print(f"  New best model saved to {model_path}!")
            
            # Save confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(output_dir, f"cm_{model_name}.png"))
            plt.close()
    
    return training_stats, best_model_path, best_val_f1

# Evaluation function
def evaluate_model(model, test_dataloader, device):
    model.eval()
    
    # Tracking variables
    all_preds = []
    all_labels = []
    
    # Test loop
    for batch in tqdm(test_dataloader, desc="Testing"):
        # Extract batch and move to device
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # No gradients
        with torch.no_grad():
            # Forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_attention_mask
            )
            
            logits = outputs.logits
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Calculate predictions
            preds = np.argmax(logits, axis=1)
            
            # Append to lists
            all_preds.extend(preds)
            all_labels.extend(label_ids)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': all_preds,
        'true_labels': all_labels
    }
    
    return results

# Main function to run experiments
def run_experiments(data_file, output_dir="bert_experiments"):
    # Set seed for reproducibility
    set_seed(42)
    
    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU.")
    
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_file)
    
    # Split data
    train_data = data[data['split'] == 'train']
    val_data = data[data['split'] == 'val']
    test_data = data[data['split'] == 'test']
    
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define experiments
    experiments = []
    
    # Varying number of layers (4, 6, 8)
    for num_layers in [4, 6, 8]:
        experiments.append(ExperimentConfig(num_layers=num_layers))
    
    # Varying number of attention heads (4, 8, 12)
    for num_heads in [4, 8, 12]:
        if num_heads != 8:  # Skip the default which is already included
            experiments.append(ExperimentConfig(num_attention_heads=num_heads))
    
    # Varying hidden state dimensions (384, 512, 768)
    for hidden_size in [384, 768]:  # Skip 512 which is the default
        experiments.append(ExperimentConfig(hidden_size=hidden_size))
    
    # Varying learning rates (5e-6, 2e-5, 5e-5)
    for lr in [5e-6, 5e-5]:  # Skip 2e-5 which is the default
        experiments.append(ExperimentConfig(learning_rate=lr))
    
    # Varying batch sizes (16, 32, 64)
    for batch_size in [16, 64]:  # Skip 32 which is the default
        experiments.append(ExperimentConfig(batch_size=batch_size))
    
    # Results storage
    experiment_results = []
    
    # Run each experiment
    for i, config in enumerate(experiments):
        print(f"\n========== Experiment {i+1}/{len(experiments)} ==========")
        print(f"Configuration: {config}")
        
        # Initialize data processor with the current batch size
        data_processor = BertDataProcessor(tokenizer, config.max_seq_length)
        
        # Prepare data loaders
        train_dataloader, num_labels = data_processor.prepare_data(
            train_data['text'].tolist(), 
            train_data['emotion'].tolist(), 
            config.batch_size
        )
        
        val_dataloader = data_processor.prepare_validation_data(
            val_data['text'].tolist(), 
            val_data['emotion'].tolist(), 
            config.batch_size
        )
        
        test_dataloader = data_processor.prepare_validation_data(
            test_data['text'].tolist(), 
            test_data['emotion'].tolist(), 
            config.batch_size
        )
        
        # Create model
        model = create_bert_lite_model(config, num_labels)
        
        # Initialize with pretrained weights if possible
        try:
            model = initialize_from_pretrained(model)
            print("Model initialized with pretrained weights.")
        except Exception as e:
            print(f"Could not initialize with pretrained weights: {e}")
            print("Training from scratch.")
        
        # Move model to device
        model.to(device)
        
        # Train model
        exp_output_dir = os.path.join(output_dir, f"experiment_{i+1}")
        training_stats, best_model_path, best_val_f1 = train_model(
            model, train_dataloader, val_dataloader, config, device, exp_output_dir
        )
        
        # Load best model for evaluation
        if best_model_path:
            model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate on test set
        test_results = evaluate_model(model, test_dataloader, device)
        
        # Save results
        result = {
            'experiment_id': i + 1,
            'config': config.to_dict(),
            'best_val_f1': best_val_f1,
            'test_accuracy': test_results['accuracy'],
            'test_f1_macro': test_results['f1_macro'],
            'test_f1_weighted': test_results['f1_weighted'],
            'training_stats': training_stats
        }
        
        experiment_results.append(result)
        
        # Save experiment results
        with open(os.path.join(exp_output_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
        # Create confusion matrix
        cm = confusion_matrix(test_results['true_labels'], test_results['predictions'])
        
        # Get label mapping
        label_mapping = data_processor.get_label_mapping()
        
        # Plot with proper labels
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[label_mapping[i] for i in range(num_labels)],
                   yticklabels=[label_mapping[i] for i in range(num_labels)])
        plt.title(f'Test Confusion Matrix - Experiment {i+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_output_dir, "test_confusion_matrix.png"))
        plt.close()
        
        # Plot training and validation loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot([stat['epoch'] for stat in training_stats], 
                 [stat['train_loss'] for stat in training_stats], 'b-o', label='Training')
        plt.plot([stat['epoch'] for stat in training_stats], 
                 [stat['val_loss'] for stat in training_stats], 'r-o', label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot([stat['epoch'] for stat in training_stats], 
                 [stat['val_accuracy'] for stat in training_stats], 'g-o', label='Accuracy')
        plt.plot([stat['epoch'] for stat in training_stats], 
                 [stat['val_f1_macro'] for stat in training_stats], 'm-o', label='F1 Macro')
        plt.title('Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(exp_output_dir, "training_curves.png"))
        plt.close()
    
    # Compile and save all results
    all_results_path = os.path.join(output_dir, 'all_experiments_results.json')
    with open(all_results_path, 'w') as f:
        json.dump(experiment_results, f, indent=4)
    
    # Create summary visualization
    plt.figure(figsize=(15, 10))
    
    # Group experiments by type
    layer_exps = [exp for exp in experiment_results if exp['config']['num_layers'] != 6]
    layer_exps.append(next(exp for exp in experiment_results if exp['config']['num_layers'] == 6 and 
                          exp['config']['num_attention_heads'] == 8 and
                          exp['config']['hidden_size'] == 512))
    
    head_exps = [exp for exp in experiment_results if exp['config']['num_attention_heads'] != 8]
    head_exps.append(next(exp for exp in experiment_results if exp['config']['num_layers'] == 6 and 
                          exp['config']['num_attention_heads'] == 8 and
                          exp['config']['hidden_size'] == 512))
    
    hidden_exps = [exp for exp in experiment_results if exp['config']['hidden_size'] != 512]
    hidden_exps.append(next(exp for exp in experiment_results if exp['config']['num_layers'] == 6 and 
                          exp['config']['num_attention_heads'] == 8 and
                          exp['config']['hidden_size'] == 512))
    
    # Sort experiments
    layer_exps.sort(key=lambda x: x['config']['num_layers'])
    head_exps.sort(key=lambda x: x['config']['num_attention_heads'])
    hidden_exps.sort(key=lambda x: x['config']['hidden_size'])
    
    # Plot by layers
    plt.subplot(3, 1, 1)
    plt.bar([f"Layers: {exp['config']['num_layers']}" for exp in layer_exps],
            [exp['test_f1_macro'] for exp in layer_exps])
    plt.title('Effect of Number of Layers on F1 Score')
    plt.ylabel('Macro F1 Score')
    plt.ylim(0.7, 0.9)
    
    # Plot by attention heads
    plt.subplot(3, 1, 2)
    plt.bar([f"Heads: {exp['config']['num_attention_heads']}" for exp in head_exps],
            [exp['test_f1_macro'] for exp in head_exps])
    plt.title('Effect of Number of Attention Heads on F1 Score')
    plt.ylabel('Macro F1 Score')
    plt.ylim(0.7, 0.9)
    
    # Plot by hidden size
    plt.subplot(3, 1, 3)
    plt.bar([f"Hidden: {exp['config']['hidden_size']}" for exp in hidden_exps],
            [exp['test_f1_macro'] for exp in hidden_exps])
    plt.title('Effect of Hidden Size on F1 Score')
    plt.ylabel('Macro F1 Score')
    plt.ylim(0.7, 0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hyperparameter_comparison.png"))
    plt.close()
    
    print("\nAll experiments completed!")
    print(f"Results saved to {all_results_path}")
    
    # Return results for further analysis
    return experiment_results

# If run as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run BERT-lite hyperparameter experiments')
    parser.add_argument('--data_file', type=str, required=True, 
                        help='Path to data file (CSV with text, emotion, and split columns)')
    parser.add_argument('--output_dir', type=str, default='bert_experiments',
                        help='Directory to save experiment results')
    
    args = parser.parse_args()
    
    run_experiments(args.data_file, args.output_dir)