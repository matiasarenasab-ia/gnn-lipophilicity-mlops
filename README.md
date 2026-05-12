# GNN Lipophilicity Predictor - MLOps Project

Predicción de lipofilicidad molecular (log D) usando Graph Neural Networks.
Proyecto final de la asignatura MLOps - Máster en Deep Learning, UPM.

**Autor:** Matias Arenas

---

## Descripción

Pipeline completo de MLOps para predicción de lipofilicidad molecular sobre
el dataset del torneo GNN (4.200 moléculas, 9 features por átomo).
El modelo de producción es un **PNAGraphRegressor** (Principal Neighbourhood
Aggregation) con conexiones residuales y pooling por atención global,
que alcanza un Val RMSE de 0.6077.

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Modelo | PyTorch Geometric — PNAGraphRegressor |
| Experimentos | Weights & Biases |
| API de inferencia | FastAPI + Pydantic |
| Contenedorización | Docker |
| CI/CD | GitHub Actions |
| Despliegue | Render |

## Estructura del proyecto

gnn-lipophilicity-mlops/
├── artifacts/        # Artefactos del modelo (model.pth, deg.pt, config)
├── notebooks/        # Notebook de exploración (ModeloMLOps.ipynb)
├── src/
│   ├── model.py      # Arquitecturas GNN (GINGraphOptimized, PNAGraphRegressor)
│   ├── train.py      # Loop de entrenamiento con W&B
│   └── api/
│       ├── main.py   # FastAPI — endpoint /predict
│       └── schemas.py# Validación Pydantic de entrada/salida
├── tests/            # Tests automatizados
├── Dockerfile
└── requirements.txt

## Configuración del entorno

```bash
pip install -r requirements.txt
```

## Lanzar la API localmente

```bash
uvicorn src.api.main:app --reload --port 8000
```

La documentación interactiva queda disponible en `http://localhost:8000/docs`.

## Ejemplo de petición

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "node_features": [[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                             [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]],
           "edge_index": [[0,1],[1,0]]
         }'
```

Respuesta esperada:

```json
{
  "lipophilicity_logD": 0.3421,
  "model": "PNAGraphRegressor",
  "num_atoms": 2
}
```

## Enlaces

- **GitHub:** [enlace al repo]
- **W&B Project:** [enlace al proyecto]
- **Endpoint producción:** [enlace al endpoint]