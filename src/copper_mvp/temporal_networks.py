"""Small causal GRU/LSTM/TCN encoders with a shared current-context head."""

from copper_mvp.neural_runtime import load_tensor_runtime

torch = load_tensor_runtime()


class CausalBlock(torch.nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel - 1) * dilation
        self.first = torch.nn.Conv1d(inputs, outputs, kernel, dilation=dilation)
        self.second = torch.nn.Conv1d(outputs, outputs, kernel, dilation=dilation)
        self.residual = torch.nn.Conv1d(inputs, outputs, 1) if inputs != outputs else torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, X):
        hidden = self.dropout(torch.relu(self.first(torch.nn.functional.pad(X, (self.left_padding, 0)))))
        hidden = self.dropout(torch.relu(self.second(torch.nn.functional.pad(hidden, (self.left_padding, 0)))))
        return torch.relu(hidden + self.residual(X))


class TemporalNetwork(torch.nn.Module):
    def __init__(self, method, architecture, sequence_features=109, context_features=14):
        super().__init__()
        self.method = method
        if method in ("GRU", "LSTM"):
            hidden = architecture["hidden"]
            constructor = torch.nn.GRU if method == "GRU" else torch.nn.LSTM
            self.encoder = constructor(sequence_features, hidden, num_layers=1, batch_first=True)
        else:
            hidden = architecture["channels"]
            self.encoder = torch.nn.Sequential(
                *[
                    CausalBlock(
                        sequence_features if i == 0 else hidden,
                        hidden,
                        architecture["kernel_size"],
                        dilation,
                        architecture["dropout"],
                    )
                    for i, dilation in enumerate(architecture["dilations"])
                ]
            )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden + context_features, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )

    def encode(self, sequence):
        if self.method in ("GRU", "LSTM"):
            return self.encoder(sequence)[0]
        return self.encoder(sequence.transpose(1, 2)).transpose(1, 2)

    def forward(self, sequence, context):
        hidden = self.encode(sequence)[:, -1]
        return self.head(torch.cat((hidden, context), dim=1))
