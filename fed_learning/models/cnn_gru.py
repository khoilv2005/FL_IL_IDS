"""
CNN-GRU Model for Federated Learning

Based on DeepFed paper (IEEE TII 2020):
- CNN: Conv -> BN -> MaxPool (no dropout in CNN)
- GRU: Two identical GRU layers (100 units each)
- MLP: FC1 -> FC2 -> Dropout -> Softmax (2 FC layers, 1 dropout before output)
"""

import torch
import torch.nn as nn


class CNN_GRU_Model(nn.Module):
    """
    Hybrid CNN-GRU model for sequence classification.
    Architecture follows DeepFed paper (IEEE TII 2020).
    """
    
    def __init__(self, input_shape, num_classes: int = 34):
        super().__init__()
        # Handle input_shape: can be (seq_length,) or (seq_length, num_features)
        if isinstance(input_shape, tuple):
            seq_length = input_shape[0]
            num_features = input_shape[1] if len(input_shape) > 1 else 1
        else:
            seq_length = int(input_shape)
            num_features = 1

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_features = num_features

        # CNN blocks: Conv -> BN -> MaxPool (NO Dropout per DeepFed paper)
        self.conv1 = nn.Conv1d(num_features, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        # Calculate CNN output size
        cnn_len = seq_length
        for _ in range(3):
            cnn_len = cnn_len // 2
        self.cnn_output_size = 256 * cnn_len

        # GRU: Two identical GRU layers (100 units each per DeepFed paper)
        # Using num_layers=2 for CUDA optimization
        self.gru = nn.GRU(num_features, 100, num_layers=2, batch_first=True)
        self.gru_output_size = 100

        # MLP: FC1 -> FC2 -> Dropout -> Softmax (per DeepFed paper Eq.3)
        # Paper: "two fully connected layers" with "a dropout layer" (singular)
        concat_size = self.cnn_output_size + self.gru_output_size
        self.fc1 = nn.Linear(concat_size, 256)  # FC1
        self.fc2 = nn.Linear(256, num_classes)  # FC2 (also output layer)
        self.dropout = nn.Dropout(0.5)  # Single dropout before softmax
        self.relu = nn.ReLU()
    
    def get_fused_representation(self, x):
        """
        Get fused CNN+GRU representation (activations before MLP head).
        
        This is the representation space R^t from CGoFed paper (eq. 2):
        R^t = F(Θ^t, X^t) - intermediate representations before classifier.
        
        Returns:
            Tensor of shape [batch_size, cnn_output_size + gru_output_size]
        """
        # Handle input shape: [batch, seq_len] or [batch, seq_len, features]
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len] -> [batch, seq_len, 1]
        batch_size = x.size(0)

        # CNN pathway: Conv -> BN -> ReLU -> MaxPool (no dropout)
        x_cnn = x.permute(0, 2, 1)  # [batch, seq_len, features] -> [batch, features, seq_len]
        x_cnn = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.pool2(self.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.pool3(self.relu(self.bn3(self.conv3(x_cnn))))
        cnn_output = x_cnn.view(batch_size, -1)

        # GRU pathway: Two identical GRU layers -> last hidden state
        x_gru, _ = self.gru(x)
        gru_output = x_gru[:, -1, :]  # Take last timestep

        # Fused representation (before MLP)
        return torch.cat([cnn_output, gru_output], dim=1)

    def forward(self, x):
        """
        Forward pass: CNN+GRU fusion followed by MLP classifier.
        
        DeepFed Eq(3): FC1 -> FC2 -> Dropout -> Softmax
        """
        # Get fused representation
        z = self.get_fused_representation(x)
        
        # MLP per DeepFed Eq(3): h'1 -> h'2 -> tau -> Softmax
        z = self.relu(self.fc1(z))  # h'1 = ReLU(FC1(z))
        z = self.fc2(z)              # h'2 = FC2(h'1) - no ReLU per paper
        z = self.dropout(z)          # tau = Dropout(h'2) - Eq(3)
        
        return z  # Softmax handled by CrossEntropyLoss
