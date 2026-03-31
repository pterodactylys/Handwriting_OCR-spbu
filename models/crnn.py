from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Compact CRNN baseline for line-level OCR with CTC."""

    def __init__(self, num_classes: int, rnn_hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=(2, 1), padding=0),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.1,
        )
        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        features = self.cnn(x)  # [B, C, H', W']
        if features.size(2) != 1:
            # Keep temporal axis (W') and collapse vertical axis (H').
            features = features.mean(dim=2)
        else:
            features = features.squeeze(2)
        features = features.permute(2, 0, 1).contiguous()  # [T, B, C]
        recurrent, _ = self.rnn(features)  # [T, B, 2H]
        logits = self.classifier(recurrent)  # [T, B, num_classes]
        return logits
