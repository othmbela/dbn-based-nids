import torch

from models import (
    MLP,
    DBN,
)


def load_model(model_name: str, params: dict) -> torch.nn.Module:
    """Load Model."""
    if model_name == "MLP":
        return MLP(**params)
    elif model_name == "DBN":
        return DBN(**params)
    else:
        raise NotImplementedError(f"The model {model_name} is not implemented")
