import torch
import pytest
from torch_geometric.data import Data
from src.model import PNAGraphRegressor, load_production_model


def make_dummy_deg(max_degree: int = 4) -> torch.Tensor:
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    deg[2] = 10
    deg[3] = 20
    deg[4] = 5
    return deg


def make_dummy_graph(n_atoms: int = 6) -> Data:
    x          = torch.randn(n_atoms, 9)
    edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]], dtype=torch.long)
    batch      = torch.zeros(n_atoms, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)


def test_model_instantiation():
    deg   = make_dummy_deg()
    model = PNAGraphRegressor(num_features=9, hidden_channels=80,
                               n_layers=5, dropout=0.4, deg=deg)
    assert model is not None


def test_output_shape():
    deg   = make_dummy_deg()
    model = PNAGraphRegressor(num_features=9, hidden_channels=80,
                               n_layers=5, dropout=0.4, deg=deg)
    model.eval()
    graph = make_dummy_graph(n_atoms=6)

    with torch.no_grad():
        out = model(graph)

    assert out.shape == (1,), f"Shape esperado (1,), obtenido {out.shape}"


def test_output_is_scalar():
    deg   = make_dummy_deg()
    model = PNAGraphRegressor(num_features=9, hidden_channels=80,
                               n_layers=5, dropout=0.4, deg=deg)
    model.eval()
    graph = make_dummy_graph(n_atoms=8)

    with torch.no_grad():
        out = model(graph)

    assert isinstance(out.item(), float), "La salida debe ser un escalar float"


def test_load_production_model():
    model, config = load_production_model("artifacts")
    assert model  is not None
    assert "model_class"   in config
    assert "best_val_rmse" in config
    assert config["model_class"] == "PNAGraphRegressor"