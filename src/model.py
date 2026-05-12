import torch
import torch.nn.functional as F
from torch.nn import (
    Module, Sequential, Linear, BatchNorm1d,
    ReLU, Dropout, ModuleList
)
from torch_geometric.nn import (
    GINConv, PNAConv, GlobalAttention, GraphNorm,
    global_add_pool, global_mean_pool
)

SEED = 42


class GINGraphOptimized(Module):
    """
    GIN Optimizado con pooling dual — Modelo V1.
    Arquitectura: Input → 4×GINConv → (Add⊕Mean)Pool → MLP → 1
    """
    def __init__(self, num_features: int = 9,
                 hidden_channels: int = 128, n_layers: int = 4):
        super().__init__()
        torch.manual_seed(SEED)
        self.convs = ModuleList()

        for i in range(n_layers):
            in_dim = num_features if i == 0 else hidden_channels
            mlp = Sequential(
                Linear(in_dim, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU()
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        self.lin1      = Linear(hidden_channels * 2, hidden_channels)
        self.lin2      = Linear(hidden_channels, hidden_channels // 2)
        self.lin_final = Linear(hidden_channels // 2, 1)
        self.dropout   = Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout(x)
        x_add  = global_add_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x = torch.cat([x_add, x_mean], dim=1)
        x = self.lin1(x).relu()
        x = self.dropout(x)
        x = self.lin2(x).relu()
        x = self.dropout(x)
        return self.lin_final(x).view(-1)


class PNAGraphRegressor(Module):
    """
    PNA Hybrid Regressor — Modelo V2 (modelo de producción).
    Arquitectura: Proyección → 5×(PNAConv + GraphNorm + Residual)
                  → GlobalAttention → MLP → 1
    """
    def __init__(self, num_features: int = 9, hidden_channels: int = 80,
                 n_layers: int = 5, dropout: float = 0.4, deg=None):
        super().__init__()
        torch.manual_seed(SEED)

        self.dropout_rate = dropout
        aggregators = ['mean', 'min', 'max', 'std']
        scalers     = ['identity', 'amplification', 'attenuation']

        self.input_net = Sequential(
            Linear(num_features, hidden_channels),
            ReLU()
        )

        self.convs = ModuleList()
        self.bns   = ModuleList()

        for _ in range(n_layers):
            conv = PNAConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=None,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False
            )
            self.convs.append(conv)
            self.bns.append(GraphNorm(hidden_channels))

        gate_nn   = Sequential(Linear(hidden_channels, 1), Linear(1, 1))
        self.pool = GlobalAttention(gate_nn, nn=None)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = self.input_net(x)

        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index)
            h = bn(h, batch)
            x = x + h
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x   = self.pool(x, batch)
        x   = self.lin1(x).relu()
        out = self.lin2(x)
        return out.view(-1)


def load_production_model(artifacts_dir: str = "artifacts"):
    """
    Carga el PNAGraphRegressor de producción desde los artefactos guardados.
    Retorna (model, config) listos para inferencia.
    """
    import json, os

    config_path = os.path.join(artifacts_dir, "model_config.json")
    model_path  = os.path.join(artifacts_dir, "best_model.pth")
    deg_path    = os.path.join(artifacts_dir, "deg.pt")

    with open(config_path) as f:
        config = json.load(f)

    deg   = torch.load(deg_path, map_location="cpu", weights_only=False)
    model = PNAGraphRegressor(
        num_features    = config["num_features"],
        hidden_channels = config["hidden_channels"],
        n_layers        = config["n_layers"],
        dropout         = config["dropout"],
        deg             = deg
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=False)
    )
    model.eval()
    return model, config