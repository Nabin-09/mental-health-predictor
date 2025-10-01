import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, config, device, label_encoder):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.label_encoder = label_encoder
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0
        
        # Create results directory
        os.makedirs('results/logs', exist_ok=True)
        os.makedirs('models/saved_models', exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, attention_weights = self.model(input_ids)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['grad_clip']
            )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'    Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].squeeze().to(self.device)
                
                logits, _ = self.model(input_ids)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop"""
        patience_counter = 0
        max_patience = self.config['training']['early_stopping_patience']
        
        print(f"Starting training for {self.config['training']['num_epochs']} epochs...")
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["training"]["num_epochs"]}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('models/saved_models/best_model.pth')
                patience_counter = 0
                print(f'✅ New best model saved! Val Acc: {val_acc:.4f}')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        return self.best_val_acc
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].squeeze()
                
                logits, _ = self.model(input_ids)
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Print detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(
            all_labels, all_preds, 
            target_names=self.label_encoder.classes_
        ))
        
        return accuracy
    
    def save_model(self, filepath):
        """Save model state dict"""
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath):
        """Load model state dict"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
