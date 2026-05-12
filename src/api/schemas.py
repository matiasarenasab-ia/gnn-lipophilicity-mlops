from pydantic import BaseModel, validator


class MoleculeInput(BaseModel):
    node_features: list[list[float]]  # [[f1..f9], ...] — un vector por átomo
    edge_index:    list[list[int]]    # [[src1,src2,...], [dst1,dst2,...]]

    @validator("node_features")
    def check_node_features(cls, v):
        if not v:
            raise ValueError("La molécula debe tener al menos un átomo")
        if any(len(row) != 9 for row in v):
            raise ValueError("Cada átomo debe tener exactamente 9 features")
        return v

    @validator("edge_index")
    def check_edge_index(cls, v):
        if len(v) != 2:
            raise ValueError("edge_index debe tener exactamente 2 listas: [sources, targets]")
        return v


class PredictionOutput(BaseModel):
    lipophilicity_logD: float
    model:              str
    num_atoms:          int