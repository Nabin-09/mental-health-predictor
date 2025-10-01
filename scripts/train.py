#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import torch.nn as nn
from datetime import datetime
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import load_and_prepare_data
from models.bigru_attention import BiGRUAttentionModel
from models.trainer import ModelTrainer

def main(config_path: str = 'config/config.yaml'):
    """Main training function"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("🚀 Starting Mental Health Prediction Model Training")
    print(f"Configuration loaded from: {config_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    train_loader, val_loader, test_loader, label_encoder = load_and_prepare_data(config)
    
    print(f"✅ Data loaded successfully!")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    print(f"   - Classes: {label_encoder.classes_}")
    
    # Initialize model
    model = BiGRUAttentionModel(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_classes=len(label_encoder.classes_),
        dropout=config['model']['dropout']
    )
    
    # Print model info
    model_info = model.get_model_info()
    print(f"\n🧠 Model Architecture:")
    print(f"   - Total parameters: {model_info['total_parameters']:,}")
    print(f"   - Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"   - Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        label_encoder=label_encoder
    )
    
    # Start training
    print(f"\n🎯 Starting training for {config['training']['num_epochs']} epochs...")
    best_val_acc = trainer.train()
    
    print(f"\n🏆 Training completed!")
    print(f"   - Best validation accuracy: {best_val_acc:.4f}")
    
    # Test final model
    print("\n🧪 Evaluating on test set...")
    test_accuracy = trainer.evaluate(test_loader)
    print(f"   - Test accuracy: {test_accuracy:.4f}")
    
    print("\n✅ All done! Check the results/ directory for outputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Mental Health Prediction Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)
