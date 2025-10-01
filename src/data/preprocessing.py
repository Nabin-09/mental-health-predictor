import re
import nltk
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple
import pickle

class TextPreprocessor:
    def __init__(self, min_word_freq: int = 2, max_vocab_size: int = 10000):
        self.min_word_freq = min_word_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts"""
        # Tokenize and count words
        word_freq = Counter()
        for text in texts:
            tokens = text.split()
            word_freq.update(tokens)
        
        # Create vocabulary
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for word, freq in word_freq.most_common(self.max_vocab_size - 4):
            if freq >= self.min_word_freq:
                self.word2idx[word] = len(self.word2idx)
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def text_to_sequence(self, text: str, max_length: int = 128) -> List[int]:
        """Convert text to sequence of token indices"""
        tokens = text.split()[:max_length]
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) 
                   for token in tokens]
        
        # Pad sequence
        if len(sequence) < max_length:
            sequence.extend([self.word2idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'min_word_freq': self.min_word_freq,
            'max_vocab_size': self.max_vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.min_word_freq = vocab_data['min_word_freq']
        self.max_vocab_size = vocab_data['max_vocab_size']
