import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List
import yaml

class MentalHealthDataset(Dataset):
    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.LongTensor(self.sequences[idx]),
            'labels': torch.LongTensor([self.labels[idx]])
        }

def create_sample_data(size: int = 1000) -> pd.DataFrame:
    """Create sample dataset for testing"""
    import random
    
    depression_texts = [
        "I feel so hopeless and empty inside",
        "Everything feels meaningless and dark",
        "I can't find joy in anything anymore",
        "Life feels like a burden every day",
        "I feel worthless and unwanted"
    ]
    
    anxiety_texts = [
        "I'm constantly worried about everything",
        "My heart races and I can't breathe",
        "I feel panic attacks coming frequently",
        "I'm always on edge and nervous",
        "Anxiety consumes my daily thoughts"
    ]
    
    bipolar_texts = [
        "My mood swings are out of control",
        "I go from extreme highs to deep lows",
        "My energy levels fluctuate drastically",
        "Sometimes I'm euphoric then suddenly depressed",
        "My emotions are like a roller coaster"
    ]
    
    control_texts = [
        "I feel balanced and content with life",
        "My mood is stable and positive",
        "I enjoy my daily activities and hobbies",
        "Life is good and I feel optimistic",
        "I have a healthy relationship with my emotions"
    ]
    
    texts = []
    labels = []
    
    for _ in range(size // 4):
        texts.extend([
            random.choice(depression_texts),
            random.choice(anxiety_texts), 
            random.choice(bipolar_texts),
            random.choice(control_texts)
        ])
        labels.extend(['Depression', 'Anxiety', 'Bipolar', 'Control'])
    
    return pd.DataFrame({'text': texts, 'label': labels})

def load_and_prepare_data(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """Load and prepare data for training"""
    # Load data (replace with your actual data loading)
    df = create_sample_data(2000)  # Replace with: pd.read_csv('data/raw/your_data.csv')
    
    # Initialize preprocessor
    from .preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor(
        min_word_freq=config['data']['min_word_freq'],
        max_vocab_size=config['model']['vocab_size']
    )
    
    # Clean texts
    df['clean_text'] = df['text'].apply(preprocessor.clean_text)
    
    # Build vocabulary
    preprocessor.build_vocabulary(df['clean_text'].tolist())
    
    # Convert to sequences
    max_length = config['model']['max_seq_length']
    df['sequence'] = df['clean_text'].apply(
        lambda x: preprocessor.text_to_sequence(x, max_length)
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    # Split data
    train_size = config['data']['train_split']
    val_size = config['data']['val_split']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['sequence'].tolist(), df['label_encoded'].tolist(),
        train_size=train_size, random_state=42, stratify=df['label_encoded']
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = MentalHealthDataset(X_train, y_train)
    val_dataset = MentalHealthDataset(X_val, y_val)
    test_dataset = MentalHealthDataset(X_test, y_test)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Save preprocessor
    preprocessor.save_vocabulary('models/vocabulary.pkl')
    
    return train_loader, val_loader, test_loader, label_encoder
