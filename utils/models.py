import torch

from models import (
    NN,
    RNN,
)


def load_model(model_name: str, params: dict) -> torch.nn.Module:
    """Load Model."""
    if model_name == "NN":
        return NN(**params)
    elif model_name == "RNN":
        return RNN(**params)
    else:
        raise NotImplementedError(f"The model {model_name} is not implemented")
