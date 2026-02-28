import torch.nn as nn

class detector_model_mmse(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(32, output_dim),
            nn.LogSoftmax(dim=1),  # match torch.nn.NLLLoss()
        )

    def forward(self, x):
        return self.net(x)