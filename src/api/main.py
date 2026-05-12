
import os
import json
import torch
from torch_geometric.data import Data
from fastapi import FastAPI, HTTPException

from src.model import load_production_model
from src.api.schemas import MoleculeInput, PredictionOutput

app = FastAPI(
    title="GNN Lipophilicity Predictor",
    description="Predice la lipofilicidad molecular (log D) usando PNAGraphRegressor",
    version="1.0.0"
)

# Carga única al arrancar — no se repite en cada request
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
_model, _config = load_production_model(ARTIFACTS_DIR)


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   _config["model_class"],
        "val_rmse": _config["best_val_rmse"]
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: MoleculeInput):
    try:
        x          = torch.tensor(data.node_features, dtype=torch.float)
        edge_index = torch.tensor(data.edge_index,    dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index)
        graph.batch = torch.zeros(x.size(0), dtype=torch.long)

        with torch.no_grad():
            prediction = _model(graph)

        return PredictionOutput(
            lipophilicity_logD = round(float(prediction.item()), 4),
            model              = _config["model_class"],
            num_atoms          = x.size(0)
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))