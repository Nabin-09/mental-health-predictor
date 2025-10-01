import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_linear = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, gru_output):
        # gru_output: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = gru_output.size()
        
        # Calculate attention scores
        attention_scores = self.attention_linear(gru_output)  # (batch, seq, 1)
        attention_weights = F.softmax(attention_scores.squeeze(2), dim=1)  # (batch, seq)
        
        # Apply attention
        attended_output = torch.bmm(
            attention_weights.unsqueeze(1), gru_output
        )  # (batch, 1, hidden)
        attended_output = attended_output.squeeze(1)  # (batch, hidden)
        
        return attended_output, attention_weights

class BiGRUAttentionModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        pretrained_embeddings=None
    ):
        super(BiGRUAttentionModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))
            
        # BiGRU layer
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Classification layers
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq, embed)
        
        # BiGRU
        gru_output, _ = self.gru(embedded)  # (batch, seq, hidden*2)
        
        # Attention
        attended_output, attention_weights = self.attention(gru_output)
        
        # Classification
        output = self.dropout1(attended_output)
        output = F.relu(self.fc1(output))
        output = self.dropout2(output)
        logits = self.fc2(output)
        
        return logits, attention_weights
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
