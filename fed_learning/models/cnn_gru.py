"""
CNN-GRU Model for Federated Learning
"""

import torch
import torch.nn as nn


class CNN_GRU_Model(nn.Module):
    """
    Hybrid CNN-GRU model for sequence classification.
    Combines convolutional feature extraction with recurrent processing.
    """
    
    def __init__(self, input_shape, num_classes: int = 34):
        super().__init__()
        if isinstance(input_shape, tuple):
            seq_length = input_shape[0]
        else:
            seq_length = int(input_shape)

        self.input_shape = input_shape
        self.num_classes = num_classes

        # CNN blocks
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout_cnn1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout_cnn2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout_cnn3 = nn.Dropout(0.3)

        # Calculate CNN output size
        cnn_len = seq_length
        for _ in range(3):
            cnn_len = cnn_len // 2
        self.cnn_output_size = 256 * cnn_len

        # GRU
        self.gru1 = nn.GRU(1, 128, batch_first=True)
        self.gru2 = nn.GRU(128, 64, batch_first=True)
        self.gru_output_size = 64

        # MLP
        concat_size = self.cnn_output_size + self.gru_output_size
        self.dense1 = nn.Linear(concat_size, 256)
        self.bn_mlp1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(256, 128)
        self.bn_mlp2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)

        # CNN
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.dropout_cnn1(x_cnn)
        x_cnn = self.pool2(self.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.dropout_cnn2(x_cnn)
        x_cnn = self.pool3(self.relu(self.bn3(self.conv3(x_cnn))))
        x_cnn = self.dropout_cnn3(x_cnn)
        cnn_output = x_cnn.view(batch_size, -1)

        # GRU
        x_gru = x
        x_gru, _ = self.gru1(x_gru)
        x_gru, _ = self.gru2(x_gru)
        gru_output = x_gru[:, -1, :]

        # Concat & MLP
        z = torch.cat([cnn_output, gru_output], dim=1)
        z = self.dense1(z)
        if z.size(0) > 1:
            z = self.bn_mlp1(z)
        z = self.relu(z)
        z = self.dropout1(z)
        z = self.dense2(z)
        if z.size(0) > 1:
            z = self.bn_mlp2(z)
        z = self.relu(z)
        z = self.dropout2(z)
        return self.output(z)
