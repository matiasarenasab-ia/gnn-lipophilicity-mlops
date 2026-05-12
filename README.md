# GNN Lipophilicity Predictor — MLOps Project

Predicción de lipofilicidad molecular (log D) usando Graph Neural Networks.
Proyecto final de la asignatura MLOps — Máster en Deep Learning, UPM.

**Autor:** Matias Arenas

---

## Descripción

Pipeline completo de MLOps para predicción de lipofilicidad molecular sobre
el dataset del Torneo GNN de la UPM (4.200 moléculas, 9 features por átomo).
El modelo de producción es un **PNAGraphRegressor** (Principal Neighbourhood
Aggregation) con conexiones residuales y pooling por atención global,
que alcanza un Val RMSE de 0.6047 y un Test RMSE de 0.6858.

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Modelo | PyTorch Geometric — PNAGraphRegressor |
| Experimentos | Weights & Biases |
| API de inferencia | FastAPI + Pydantic |
| Contenedorización | Docker |
| Despliegue | Hugging Face Spaces |

## Estructura del proyecto

gnn-lipophilicity-mlops/
├── artifacts/          # Artefactos del modelo
│   ├── best_model.pth  # Pesos del PNAGraphRegressor
│   ├── deg.pt          # Histograma de grados (requerido por PNAConv)
│   └── model_config.json
├── notebooks/
│   └── ModeloMLOps.ipynb  # Exploración y entrenamiento
├── src/
│   ├── model.py        # Arquitecturas GNN
│   └── api/
│       ├── main.py     # FastAPI — endpoint /predict
│       └── schemas.py  # Validación Pydantic
├── tests/              # Tests automatizados
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

## Contenedorización con Docker

El proyecto incluye un `Dockerfile` que empaqueta el servicio completo.
Las capas están ordenadas para maximizar el uso del caché de Docker:
primero las dependencias del sistema, luego `requirements.txt`, y
finalmente el código fuente.

Para construir y ejecutar el contenedor en un entorno con Docker disponible:

```bash
docker build -t gnn-lipophilicity .
docker run -p 8000:8000 gnn-lipophilicity
```

La API quedaría disponible en `http://localhost:8000`.

En este proyecto el despliegue en producción se realizó mediante
Hugging Face Spaces, que construye y ejecuta el contenedor Docker
directamente en su infraestructura cloud.

## Ejemplo de petición

```bash
curl -X POST "https://matiasluis-gnn-lipophilicity.hf.space/predict" \
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

## Tests

```bash
pytest tests/ -v
```

## Enlaces

- **GitHub:** https://github.com/matiasarenasab-ia/gnn-lipophilicity-mlops
- **W&B Report:** https://wandb.ai/matias-arenas-universidad-polit-cnica-de-madrid/gnn-lipophilicity-mlops/reports/An-lisis-de-Experimento-PNAGraphRegressor-para-Predicci-n-de-Lipofilicidad--VmlldzoxNjg1OTM2Mg
- **Endpoint producción:** https://matiasluis-gnn-lipophilicity.hf.space
