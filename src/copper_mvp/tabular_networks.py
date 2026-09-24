"""Native TabNet/FT-Transformer and dense NODE behind a shared tensor interface."""

from copper_mvp.tabular_runtime import tabular_runtime

torch = tabular_runtime()
from copper_mvp.node_network import NodeNetwork


class TabularNetwork(torch.nn.Module):
    def __init__(self, method, architecture, numeric_features=220):
        super().__init__()
        self.method = method
        self.sparsity_weight = 0.0
        options = dict(architecture)
        if method == "TabNet":
            from pytorch_tabnet.tab_network import TabNet

            self.sparsity_weight = options.pop("lambda_sparse")
            options["cat_emb_dim"] = [options["cat_emb_dim"]] * 4
            total = numeric_features + 4
            self.core = TabNet(
                input_dim=total,
                output_dim=2,
                cat_idxs=list(range(numeric_features, total)),
                cat_dims=[3] * 4,
                group_attention_matrix=torch.eye(total),
                **options,
            )
        elif method == "FTTransformer":
            from rtdl_revisiting_models import FTTransformer

            self.core = FTTransformer(
                n_cont_features=numeric_features,
                cat_cardinalities=[3] * 4,
                d_out=2,
                ffn_d_hidden_multiplier=None,
                **options,
            )
        elif method == "NODE":
            for key in ("selector", "split", "threshold_initialization"):
                options.pop(key)
            self.core = NodeNetwork(features=numeric_features + 12, **options)
        elif method == "MLPControl":
            first, second = options["hidden"]
            self.core = torch.nn.Sequential(
                torch.nn.Linear(numeric_features + 12, first),
                torch.nn.ReLU(),
                torch.nn.Linear(first, second),
                torch.nn.ReLU(),
                torch.nn.Linear(second, 2),
            )
        else:
            raise ValueError("Unknown neural method")

    def node_input(self, numerical, categories):
        return torch.cat((numerical, torch.nn.functional.one_hot(categories, 3).flatten(1).float()), dim=1)

    def initialize(self, numerical, categories):
        if self.method == "NODE":
            self.core.initialize(self.node_input(numerical, categories))

    def forward(self, numerical, categories):
        if self.method == "TabNet":
            output, mask_loss = self.core(torch.cat((numerical, categories.float()), dim=1))
            return output, -self.sparsity_weight * mask_loss
        if self.method == "FTTransformer":
            return self.core(numerical, categories), numerical.new_zeros(())
        return self.core(self.node_input(numerical, categories)), numerical.new_zeros(())


def matched_mlp_architecture(parameter_budget, numeric_features=220):
    inputs = numeric_features + 12
    candidates = []
    for first in range(8, 513):
        estimate = (parameter_budget - (inputs + 1) * first - 2) / (first + 3)
        for second in (int(estimate), int(estimate) + 1):
            if 8 <= second <= 512:
                count = (inputs + 1) * first + (first + 3) * second + 2
                candidates.append((abs(count - parameter_budget), abs(first - second), first, second, count))
    _, _, first, second, count = min(candidates)
    return {"hidden": [first, second], "parameters": count}
