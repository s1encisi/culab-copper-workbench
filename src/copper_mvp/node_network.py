"""Dense oblivious trees following the NODE authors' sparsemax/sparsemoid design.

Architecture reference: https://github.com/Qwicen/node (ODST and DenseBlock).
This implementation uses Torch-only, training-data initialization and two outputs.
"""

from copper_mvp.tabular_runtime import tabular_runtime

torch = tabular_runtime()
from entmax import sparsemax


class ObliviousLayer(torch.nn.Module):
    def __init__(self, features, trees, depth, outputs):
        super().__init__()
        self.trees, self.depth, self.outputs = trees, depth, outputs
        self.selection = torch.nn.Parameter(torch.empty(features, trees, depth).uniform_())
        self.thresholds = torch.nn.Parameter(torch.zeros(trees, depth))
        self.log_temperature = torch.nn.Parameter(torch.zeros(trees, depth))
        self.responses = torch.nn.Parameter(torch.randn(trees, outputs, 2**depth))
        bits = ((torch.arange(2**depth)[None, :] // (2 ** torch.arange(depth))[:, None]) % 2).bool()
        self.register_buffer("bits", bits)

    def selected_values(self, X):
        return torch.einsum("bi,itd->btd", X, sparsemax(self.selection, dim=0))

    def initialize(self, X):
        with torch.no_grad():
            values = self.selected_values(X)
            flat = values.flatten(1).sort(dim=0).values
            positions = torch.rand(flat.shape[1]) * (len(X) - 1)
            low = positions.floor().long()
            high = positions.ceil().long()
            column = torch.arange(flat.shape[1])
            fraction = positions - low
            thresholds = (flat[low, column] * (1 - fraction) + flat[high, column] * fraction).view(
                self.trees, self.depth
            )
            self.thresholds.copy_(thresholds)
            scale = (values - thresholds).abs().amax(dim=0) + 1e-6
            self.log_temperature.copy_(scale.log())

    def forward(self, X):
        logits = (self.selected_values(X) - self.thresholds) * torch.exp(-self.log_temperature)
        positive = (0.5 + 0.5 * logits).clamp(0.0, 1.0)
        matches = torch.where(self.bits[None, None, :, :], 1 - positive[:, :, :, None], positive[:, :, :, None])
        weights = matches.prod(dim=2)
        return torch.einsum("btl,tol->bto", weights, self.responses)


class NodeNetwork(torch.nn.Module):
    def __init__(self, features, layers, trees_per_layer, depth, tree_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                ObliviousLayer(features + i * trees_per_layer * tree_dim, trees_per_layer, depth, tree_dim)
                for i in range(layers)
            ]
        )

    def initialize(self, X):
        with torch.no_grad():
            for layer in self.layers:
                layer.initialize(X)
                X = torch.cat((X, layer(X).flatten(1)), dim=1)

    def forward(self, X):
        outputs = []
        for layer in self.layers:
            value = layer(X)
            outputs.append(value)
            X = torch.cat((X, value.flatten(1)), dim=1)
        return torch.cat(outputs, dim=1).mean(dim=1)
