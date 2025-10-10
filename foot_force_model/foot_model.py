import torch
import torch.nn as nn
import torch.nn.functional as F

class FootForceCNN(nn.Module):
    def __init__(self, input_channels=4, num_classes=5):
        super(FootForceCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * (90 // 2), 128)   # After pooling halves sequence length
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, channels=features, seq_len)
        
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 45)
        x = self.dropout(x)
        
        x = x.flatten(start_dim=1)            # (batch, 64*45)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
