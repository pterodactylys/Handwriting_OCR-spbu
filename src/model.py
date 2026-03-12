from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, 16, W/2]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 128, 8, W/4]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [B, 256, 4, W/4]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [B, 256, 2, W/4]
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [B, 512, 1, W/4]
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    @staticmethod
    def compute_output_lengths(image_widths: torch.Tensor) -> torch.Tensor:
        output_lengths = image_widths.clone()
        output_lengths = torch.div(output_lengths, 2, rounding_mode="floor")
        output_lengths = torch.div(output_lengths, 2, rounding_mode="floor")
        return torch.clamp(output_lengths, min=1)

    def forward(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # images: [B, 1, H, W]
        features = self.cnn(images)
        if features.shape[2] != 1:
            raise RuntimeError(
                f"CRNN expects CNN height 1 before the LSTM, got {features.shape[2]}."
            )

        # [B, 512, 1, W'] -> [W', B, 512]
        sequence = features.squeeze(2).permute(2, 0, 1)
        recurrent_features, _ = self.rnn(sequence)
        logits = self.classifier(recurrent_features)  # [T, B, C]
        log_probs = F.log_softmax(logits, dim=-1)
        output_lengths = self.compute_output_lengths(image_widths)
        return log_probs, output_lengths
