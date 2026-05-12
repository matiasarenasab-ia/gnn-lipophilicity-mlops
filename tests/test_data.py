import pytest
from pydantic import ValidationError
from src.api.schemas import MoleculeInput


def test_valid_input():
    mol = MoleculeInput(
        node_features=[[0.1]*9, [0.2]*9, [0.3]*9],
        edge_index=[[0,1,1,2],[1,0,2,1]]
    )
    assert len(mol.node_features) == 3
    assert len(mol.node_features[0]) == 9


def test_wrong_feature_dimension():
    with pytest.raises(ValidationError):
        MoleculeInput(
            node_features=[[0.1]*7],    # 7 features en vez de 9
            edge_index=[[],[]]
        )


def test_empty_molecule():
    with pytest.raises(ValidationError):
        MoleculeInput(
            node_features=[],           # sin átomos
            edge_index=[[],[]]
        )


def test_wrong_edge_index_format():
    with pytest.raises(ValidationError):
        MoleculeInput(
            node_features=[[0.1]*9],
            edge_index=[[0,1,2]]        # solo una lista
        )


def test_nine_features_accepted():
    mol = MoleculeInput(
        node_features=[[float(i) for i in range(9)]],
        edge_index=[[],[]]
    )
    assert mol.node_features[0] == list(range(9))